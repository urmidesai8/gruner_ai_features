import json
from typing import List, Optional, Set

from dotenv import load_dotenv
from groq import Groq

from app.models.schemas import chat_history
from app.core.config import settings


load_dotenv()

# Initialize Groq client (can be swapped for local LLM later if needed)
groq_client = Groq(api_key=settings.GROQ_API_KEY)


def generate_chat_summary(messages: List[dict], username: Optional[str] = None, total_messages: int = 100) -> dict:
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

    try:
        if not groq_client.api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Please set it in your environment variables or .env file."
            )

        api_params = {
            "model": "llama-3.1-8b-instant",
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

        # Parse the response
        response_text = completion.choices[0].message.content.strip()

        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

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
        return {
            "summary": (
                f"Chat summary: {total_messages_count} messages from "
                f"{len(participants)} participant(s): {', '.join(participants)}"
            ),
            "bullet_points": [
                f"{msg['sender']}: {msg['message'][:80]}..."
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

def generate_text_summary(text: str) -> dict:
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
        if not groq_client.api_key:
             return {"summary": "Error: GROQ_API_KEY not set."}

        api_params = {
            "model": "llama-3.1-8b-instant",
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
