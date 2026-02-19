import json
import re
from typing import List, Optional

from app.services.summarizer import (
    groq_client,
    DEEPSEEK_R1_MODEL_ID,
    LFM_MODEL_ID,
    _extract_json_from_response,
    _generate_with_deepseek_r1,
    _generate_with_lfm,
)


def _repair_json_for_llm(text: str) -> str:
    """Fix common LLM JSON syntax errors (trailing commas, etc.) before parsing."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def _tasks_from_parsed(parsed) -> list:
    """Extract tasks list from parsed JSON (handles 'tasks' key or root list)."""
    if isinstance(parsed, list):
        return _filter_valid_tasks(parsed)
    if isinstance(parsed, dict):
        tasks = parsed.get("tasks") or parsed.get("task") or []
        return _filter_valid_tasks(tasks) if isinstance(tasks, list) else []
    return []


def _parse_tasks_json(response_text: str) -> list:
    """Parse JSON and return tasks list; try repair and truncation fallback on failure."""
    response_text = _repair_json_for_llm(response_text)
    try:
        parsed = json.loads(response_text)
        return _tasks_from_parsed(parsed)
    except json.JSONDecodeError:
        pass
    # Fallback: find the last balanced '}' and try parsing up to there (handles truncation)
    depth = 0
    last_good = -1
    for i, c in enumerate(response_text):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                last_good = i
    if last_good != -1:
        try:
            parsed = json.loads(_repair_json_for_llm(response_text[: last_good + 1]))
            return _tasks_from_parsed(parsed)
        except json.JSONDecodeError:
            pass
    # Try to find JSON array [...] if object parse failed (model might output only the array)
    start = response_text.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(response_text)):
            if response_text[i] == "[":
                depth += 1
            elif response_text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(_repair_json_for_llm(response_text[start : i + 1]))
                        return _tasks_from_parsed(parsed)
                    except json.JSONDecodeError:
                        break
                    break
    return []


def _parse_tasks_json_from_raw(raw_response: str) -> list:
    """Try to extract tasks when model outputs reasoning then JSON (find last '{\"tasks\"' block)."""
    for needle in ['{"tasks":', '"tasks":']:
        idx = raw_response.rfind(needle)
        if idx == -1:
            continue
        start = raw_response.rfind("{", 0, idx + 1)
        if start == -1:
            start = idx
        depth = 0
        for i in range(start, len(raw_response)):
            if raw_response[i] == "{":
                depth += 1
            elif raw_response[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = raw_response[start : i + 1]
                    tasks = _parse_tasks_json(chunk)
                    if tasks:
                        return tasks
                    break
    return []


def _filter_valid_tasks(tasks: list) -> list:
    """Keep tasks that have at least a title or description; skip only non-dict or completely empty."""
    result = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        title = (t.get("title") or "").strip()
        desc = (t.get("description") or "").strip()
        # Keep if there is any meaningful content (title or description)
        if title or desc:
            result.append(t)
        elif t.get("raw_message") or t.get("id"):
            # Keep if it has raw_message or id (might be a valid task with empty title)
            result.append(t)
    return result


def extract_tasks_from_messages(messages: List[dict], model: Optional[str] = None) -> dict:
    """Extract structured tasks (todos) from chat messages using the LLM.

    Returns a dict of the form:
    {
      "tasks": [
        {
          "id": "task-1",
          "title": "Short task title",
          "description": "Full task description",
          "assignee": "Alice" | null,
          "due_date": "2026-02-01" | null,
          "raw_message": "original message text",
          "message_id": "<source message_id>",
          "timestamp": "2026-01-27 10:30:00",
          "status": "todo" | "in_progress" | "done"
        },
        ...
      ]
    }
    """
    if not messages:
        return {"tasks": []}

    # Filter out system messages
    chat_messages = [m for m in messages if m.get("sender") != "System"]
    if not chat_messages:
        return {"tasks": []}

    # Format messages for the prompt
    chat_text = "\n".join(
        f"[{m['timestamp']}] {m['sender']}: {m['message']} (id={m['message_id']})"
        for m in chat_messages
    )

    prompt = f"""You are an expert assistant that reads chat conversations and extracts TASKS / TODOS.

Chat conversation:
{chat_text}

Your job is ONLY to return a JSON object with a list of tasks extracted from the conversation.
A "task" is something that someone should do in the future (work item, follow-up, bug to fix, document to write, meeting to schedule, etc.).

Return JSON with the following structure:
{{
  "tasks": [
    {{
      "id": "string - unique synthetic ID you generate (e.g. task-1, task-2)",
      "title": "short human-readable task title (max 10-12 words)",
      "description": "concise description of the task, including context from the chat",
      "assignee": "person responsible if clearly mentioned (by name or @handle), otherwise null",
      "due_date": "ISO date (YYYY-MM-DD) if an explicit deadline is mentioned, otherwise null",
      "raw_message": "exact original message text that contained the task",
      "message_id": "the message_id of the message that contained the task (from the chat text)",
      "timestamp": "timestamp of the message that contained the task",
      "status": "one of: 'todo', 'in_progress', 'done' (infer from wording if possible, otherwise 'todo')"
    }},
    ...
  ]
}}

Guidelines:
- Only include tasks that are clearly implied or stated.
- If no tasks are present, return {{"tasks": []}}.
- Do NOT include any explanation text, ONLY the JSON object.
- Do NOT wrap the JSON in markdown code fences.

Return the JSON now.
"""

    # Shorter, strict prompt for local models (DeepSeek R1, LFM) to reduce echoing and force JSON-only output
    prompt_local = f"""Extract tasks/todos from this conversation. Output ONLY a JSON object with a "tasks" array. Do not repeat or quote the conversation.

Conversation:
{chat_text}

Rules: A task is an actionable item (work to do, follow-up, deadline). Extract only real tasks. If none, return {{"tasks": []}}.
Each task must have: "id", "title", "description", "assignee" (or null), "due_date" (or null), "raw_message", "message_id", "timestamp", "status" ("todo"/"in_progress"/"done").

Output only the JSON, no other text:"""

    system_content = (
        "You are a precise JSON-producing task extraction engine. "
        "You MUST return only valid JSON with no extra text."
    )
    system_content_local = (
        "You output only valid JSON. No explanations, no repetition of the input, no markdown. Only a single JSON object."
    )

    try:
        if model == DEEPSEEK_R1_MODEL_ID:
            raw_response = _generate_with_deepseek_r1(
                prompt_local, system_content=system_content_local, max_new_tokens=2048
            )
            response_text = _extract_json_from_response(raw_response)
            tasks = _parse_tasks_json(response_text)
            if not tasks and raw_response.strip():
                tasks = _parse_tasks_json_from_raw(raw_response)
            return {"tasks": tasks}

        if model == LFM_MODEL_ID:
            raw_response = _generate_with_lfm(
                prompt_local, system_content=system_content_local, max_new_tokens=2048
            )
            response_text = _extract_json_from_response(raw_response)
            tasks = _parse_tasks_json(response_text)
            if not tasks and raw_response.strip():
                tasks = _parse_tasks_json_from_raw(raw_response)
            return {"tasks": tasks}

        if not groq_client.api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Please set it in your environment variables or .env file."
            )

        api_params = {
            "model": model or "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1500,
        }

        try:
            api_params["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        completion = groq_client.chat.completions.create(**api_params)
        response_text = completion.choices[0].message.content.strip()

        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()
        parsed = json.loads(response_text)

        tasks = parsed.get("tasks", [])
        if not isinstance(tasks, list):
            tasks = []

        return {"tasks": tasks}

    except Exception as e:  # pragma: no cover - defensive
        print(f"Error extracting tasks from LLM: {e}")
        return {"tasks": []}
