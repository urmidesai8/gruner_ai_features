import json
import re
from typing import List, Optional, Set

from dotenv import load_dotenv
from groq import Groq

from app.models.schemas import chat_history
from app.core.config import settings


load_dotenv()

# Initialize Groq client (can be swapped for local LLM later if needed)
groq_client = Groq(api_key=settings.GROQ_API_KEY)


def _extract_json_from_response(response_text: str) -> str:
    """
    Extract a JSON object from model output that may include chain-of-thought
    or markdown fences (e.g. ```json ... ```). Returns the substring to pass to json.loads().
    """
    text = response_text.strip()
    # Remove markdown code block if present
    if "```json" in text:
        text = re.sub(r"^.*?```json\s*", "", text, flags=re.DOTALL)
    if "```" in text:
        text = re.sub(r"\s*```.*$", "", text, flags=re.DOTALL)
    text = text.strip()
    # If there is leading text before the first '{', take from first '{' to matching '}'
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Fallback: first { to last }
    last_brace = text.rfind("}")
    if last_brace != -1:
        return text[start : last_brace + 1]
    return text[start:]

# Local DeepSeek R1 Distill Qwen 1.5B (load once, use from cache)
DEEPSEEK_R1_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
_deepseek_tokenizer = None
_deepseek_model = None


def _get_deepseek_r1_model():
    """Lazy-load tokenizer and model for DeepSeek R1 Distill Qwen 1.5B."""
    global _deepseek_tokenizer, _deepseek_model
    if _deepseek_model is not None:
        return _deepseek_tokenizer, _deepseek_model

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    _deepseek_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_R1_MODEL_ID, trust_remote_code=True)
    _deepseek_model = AutoModelForCausalLM.from_pretrained(DEEPSEEK_R1_MODEL_ID, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _deepseek_model = _deepseek_model.to(device)
    _deepseek_model.eval()
    return _deepseek_tokenizer, _deepseek_model


def _generate_with_deepseek_r1(user_content: str, system_content: Optional[str] = None, max_new_tokens: int = 2000) -> str:
    """Run inference with DeepSeek R1; returns decoded generated text only."""
    import torch

    tokenizer, model = _get_deepseek_r1_model()
    if system_content:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [{"role": "user", "content": user_content}]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
    except Exception:
        # Fallback if system role not supported
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": (system_content or "") + "\n\n" + user_content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

    device = next(model.parameters()).device
    gen_kwargs = {"input_ids": inputs["input_ids"].to(device)}
    if inputs.get("attention_mask") is not None:
        gen_kwargs["attention_mask"] = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            **gen_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = gen_kwargs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


def generate_chat_summary(messages: List[dict], username: Optional[str] = None, total_messages: int = 100, model: str = None) -> dict:
    """
    Generate a comprehensive chat summary using Groq Llama 3.1 8B instant model.

    Includes:
    - Bullet point summary
    - Key decisions
    - Action items
    - "What did I miss?" summary for unread messages
    """
    if not messages:
        return {
            "summary": "No messages to summarize.",
            "bullet_points": [],
            "key_decisions": [],
            "action_items": [],
            "unread_summary": "No unread messages.",
            "total_messages": 0,
            "participants": [],
        }

    # Filter out system messages for summarization
    chat_messages = [msg for msg in messages if msg.get("sender") != "System"]

    if not chat_messages:
        return {
            "summary": "No chat messages to summarize.",
            "bullet_points": [],
            "key_decisions": [],
            "action_items": [],
            "unread_summary": "No unread chat messages.",
            "total_messages": 0,
            "participants": [],
        }

    # Limit messages to the last N messages if total_messages is specified
    if total_messages and total_messages > 0:
        chat_messages = chat_messages[-total_messages:]
    
    total_messages_count = len(chat_messages)
    participants: Set[str] = {msg["sender"] for msg in chat_messages}

    # Format messages for the prompt
    chat_text = "\n".join(
        f"[{msg['timestamp']}] {msg['sender']}: {msg['message']}"
        for msg in chat_messages
    )

    # Generate "What did I miss?" context
    unread_context = ""
    if username:
        unread_messages = chat_history.get_unread_messages(username)
        if unread_messages:
            unread_count = len(unread_messages)
            unread_context = (
                f"\n\nIMPORTANT: The user '{username}' has {unread_count} unread "
                "message(s). Please provide a 'What did I miss?' summary focusing on "
                "these unread messages."
            )

    # Create the prompt for the LLM
    prompt = f"""You are analyzing a chat conversation. Please provide a comprehensive summary in JSON format.

Chat Conversation:
{chat_text}
{unread_context}

Please analyze this conversation and provide a JSON response with the following structure:
{{
    "summary": "A brief 2-3 sentence overview of the entire conversation",
    "bullet_points": ["Key point 1", "Key point 2", "Key point 3", ...],
    "key_decisions": ["Decision 1 with context", "Decision 2 with context", ...],
    "action_items": ["Action item 1 with assignee if mentioned", "Action item 2", ...],
    "unread_summary": "A personalized summary for the user about what they missed (if unread messages exist, focus on those)"
}}

Guidelines:
- bullet_points: Extract 5-10 most important points from the conversation as clear bullet points
- key_decisions: Identify any decisions, agreements, or choices made during the conversation (include who made them and what was decided)
- action_items: Extract any tasks, todos, or action items mentioned (include who is responsible if mentioned)
- unread_summary: If there are unread messages, summarize what happened in those messages. If no unread messages, say "You're all caught up!"
- Be concise but informative
- If a category has no items, return an empty array []
- Return ONLY valid JSON, no additional text before or after

Return the JSON response now:"""

    raw_response_text: Optional[str] = None
    try:
        if model == DEEPSEEK_R1_MODEL_ID:
            system_content = (
                "You are a helpful assistant that analyzes chat conversations "
                "and provides structured summaries in JSON format. Always "
                "return valid JSON only, no markdown code blocks, no additional text."
            )
            raw_response_text = _generate_with_deepseek_r1(prompt, system_content=system_content, max_new_tokens=2000)
            response_text = _extract_json_from_response(raw_response_text)
            llm_summary = json.loads(response_text)
            result = {
                "summary": llm_summary.get(
                    "summary",
                    f"Chat summary: {total_messages_count} messages from "
                    f"{len(participants)} participant(s): {', '.join(participants)}",
                ),
                "bullet_points": llm_summary.get("bullet_points", []),
                "key_decisions": llm_summary.get("key_decisions", []),
                "action_items": llm_summary.get("action_items", []),
                "unread_summary": llm_summary.get("unread_summary", "Summary generated successfully."),
                "total_messages": total_messages_count,
                "participants": list(participants),
            }
            if not result["key_decisions"]:
                result["key_decisions"] = ["No explicit decisions identified in the conversation."]
            if not result["action_items"]:
                result["action_items"] = ["No action items identified in the conversation."]
            return result

        if not groq_client.api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Please set it in your environment variables or .env file."
            )

        api_params = {
            "model": model or "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that analyzes chat conversations "
                        "and provides structured summaries in JSON format. Always "
                        "return valid JSON only, no markdown code blocks, no additional "
                        "text."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        # Try to use JSON mode if supported (some Groq models support it)
        try:
            api_params["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        completion = groq_client.chat.completions.create(**api_params)

        # Parse the response (extract JSON in case of chain-of-thought or markdown)
        raw_response_text = completion.choices[0].message.content.strip()
        response_text = _extract_json_from_response(raw_response_text)
        llm_summary = json.loads(response_text)

        result = {
            "summary": llm_summary.get(
                "summary",
                f"Chat summary: {total_messages_count} messages from "
                f"{len(participants)} participant(s): {', '.join(participants)}",
            ),
            "bullet_points": llm_summary.get("bullet_points", []),
            "key_decisions": llm_summary.get("key_decisions", []),
            "action_items": llm_summary.get("action_items", []),
            "unread_summary": llm_summary.get(
                "unread_summary", "Summary generated successfully."
            ),
            "total_messages": total_messages_count,
            "participants": list(participants),
        }

        if not result["key_decisions"]:
            result["key_decisions"] = [
                "No explicit decisions identified in the conversation."
            ]
        if not result["action_items"]:
            result["action_items"] = [
                "No action items identified in the conversation."
            ]

        return result

    except json.JSONDecodeError as e:
        print(f"Error parsing LLM JSON response: {e}")
        print(f"Response was: {response_text}")
        # Retry by extracting JSON from raw response (e.g. chain-of-thought before ```json)
        if raw_response_text:
            try:
                extracted = _extract_json_from_response(raw_response_text)
                llm_summary = json.loads(extracted)
                result = {
                    "summary": llm_summary.get(
                        "summary",
                        f"Chat summary: {total_messages_count} messages from "
                        f"{len(participants)} participant(s): {', '.join(participants)}",
                    ),
                    "bullet_points": llm_summary.get("bullet_points", []),
                    "key_decisions": llm_summary.get("key_decisions", []),
                    "action_items": llm_summary.get("action_items", []),
                    "unread_summary": llm_summary.get("unread_summary", "Summary generated successfully."),
                    "total_messages": total_messages_count,
                    "participants": list(participants),
                }
                if not result["key_decisions"]:
                    result["key_decisions"] = ["No explicit decisions identified in the conversation."]
                if not result["action_items"]:
                    result["action_items"] = ["No action items identified in the conversation."]
                return result
            except (json.JSONDecodeError, TypeError):
                pass
        return {
            "summary": (
                f"Chat summary: {total_messages_count} messages from "
                f"{len(participants)} participant(s): {', '.join(participants)}"
            ),
            "bullet_points": [
                f"{msg['sender']}: {msg['message']}"
                for msg in chat_messages[:10]
            ],
            "key_decisions": ["Error parsing LLM response. Please try again."],
            "action_items": ["Error parsing LLM response. Please try again."],
            "unread_summary": "Error generating unread summary.",
            "total_messages": total_messages_count,
            "participants": list(participants),
        }
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return {
            "summary": (
                f"Chat summary: {total_messages_count} messages from "
                f"{len(participants)} participant(s): {', '.join(participants)}"
            ),
            "bullet_points": [
                f"{msg['sender']}: {msg['message'][:80]}..."
                for msg in chat_messages[:10]
            ],
            "key_decisions": [f"Error: {str(e)}. Please check your GROQ_API_KEY."],
            "action_items": [f"Error: {str(e)}. Please check your GROQ_API_KEY."],
            "unread_summary": f"Error generating summary: {str(e)}",
            "total_messages": total_messages_count,
            "participants": list(participants),
        }

def generate_text_summary(text: str, model: str = None) -> dict:
    """
    Generate a formatted summary structure from raw text (e.g. transcription).
    Reuses the structure of chat summary.
    """
    if not text:
        return {
            "summary": "No text to summarize.",
            "bullet_points": [],
            "key_decisions": [],
            "action_items": [],
            "unread_summary": "",
        }

    prompt = f"""You are analyzing a transcript. Please provide a comprehensive summary in JSON format.

Transcript:
{text}

Please analyze this text and provide a JSON response with the following structure:
{{
    "summary": "A brief 2-3 sentence overview",
    "bullet_points": ["Key point 1", "Key point 2", ...],
    "key_decisions": ["Decision 1", ...],
    "action_items": ["Action item 1", ...]
}}

Guidelines:
- bullet_points: Extract 5-10 most important points
- key_decisions: Identify decisions/agreements
- action_items: Extract tasks/todos
- Return ONLY valid JSON
"""

    try:
        if model == DEEPSEEK_R1_MODEL_ID:
            system_content = "You are a helpful assistant that analyzes text and provides structured JSON summaries."
            response_text = _generate_with_deepseek_r1(prompt, system_content=system_content, max_new_tokens=1500)
            response_text = _extract_json_from_response(response_text)
            llm_summary = json.loads(response_text)
            return {
                "summary": llm_summary.get("summary", "Summary generated."),
                "bullet_points": llm_summary.get("bullet_points", []),
                "key_decisions": llm_summary.get("key_decisions", []),
                "action_items": llm_summary.get("action_items", []),
                "unread_summary": "N/A for transcript",
            }

        if not groq_client.api_key:
             return {"summary": "Error: GROQ_API_KEY not set."}

        api_params = {
            "model": model or "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that analyzes text and provides structured JSON summaries."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }

        completion = groq_client.chat.completions.create(**api_params)
        response_text = completion.choices[0].message.content.strip()
        
        # JSON formatting safety
        if response_text.startswith("```json"): response_text = response_text[7:]
        if response_text.startswith("```"): response_text = response_text[3:]
        if response_text.endswith("```"): response_text = response_text[:-3]
        
        llm_summary = json.loads(response_text.strip())
        
        return {
            "summary": llm_summary.get("summary", "Summary generated."),
            "bullet_points": llm_summary.get("bullet_points", []),
            "key_decisions": llm_summary.get("key_decisions", []),
            "action_items": llm_summary.get("action_items", []),
            "unread_summary": "N/A for transcript"
        }

    except Exception as e:
        return {"summary": f"Error generating summary: {str(e)}"}
