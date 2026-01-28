import json
from typing import List

from app.services.summarizer import groq_client  # reuse same LLM client


def extract_tasks_from_messages(messages: List[dict]) -> dict:
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

    if not groq_client.api_key:
        # Surface a clear error upstream
        raise ValueError(
            "GROQ_API_KEY not set. Please set it in your environment variables or .env file."
        )

    api_params = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise JSON-producing task extraction engine. "
                    "You MUST return only valid JSON with no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
    }

    try:
        # Try JSON mode if supported
        try:
            api_params["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        completion = groq_client.chat.completions.create(**api_params)
        response_text = completion.choices[0].message.content.strip()

        # Strip markdown fences if present
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
        # Fail soft: return empty list so API still works
        return {"tasks": []}
