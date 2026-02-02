import json
from typing import Optional, Dict
from datetime import datetime, timedelta
import uuid

from app.services.summarizer import groq_client
from app.models.schemas import chat_history


def generate_context_based_suggestions(
    username: Optional[str] = None, context_window: Optional[int] = None, model: str = None
) -> Dict:
    """Generate context-based reminder suggestions from chat history.
    
    Args:
        username: Optional username to filter messages for personalized suggestions
        context_window: Optional number of recent messages to consider (default: all)
        model: Optional model name to use
    
    Returns:
        Dict with structure:
        {
            "suggestions": [
                {
                    "id": "suggestion-1",
                    "title": "Reminder title",
                    "description": "Context-aware description",
                    "suggested_due_date": "2026-02-01" | null,
                    "priority": "low" | "medium" | "high",
                    "context": "Relevant chat context",
                    "confidence": 0.85
                },
                ...
            ]
        }
    """
    try:
        # Get relevant messages (only AI-enabled messages)
        if username:
            all_messages = chat_history.get_unread_messages(username)
            if not all_messages:
                all_messages = chat_history.get_ai_enabled_messages()
        else:
            all_messages = chat_history.get_ai_enabled_messages()
        
        messages = all_messages
        
        # Apply context window if specified
        if context_window and context_window > 0:
            messages = messages[-context_window:]
        
        if not messages:
            return {"suggestions": []}
        
        # Format messages for the prompt
        chat_text = "\n".join(
            f"[{m['timestamp']}] {m['sender']}: {m['message']}"
            for m in messages
        )
        
        prompt = f"""You are an intelligent reminder assistant. Analyze the following chat conversation and suggest relevant reminders based on the context.

Chat conversation:
{chat_text}

Your job is to identify items that would benefit from reminders. These could be:
- Follow-up actions mentioned but not scheduled
- Deadlines or time-sensitive items
- Commitments or promises made
- Tasks that need to be done at a specific time
- Recurring items that should be tracked

Return a JSON object with the following structure:
{{
  "suggestions": [
    {{
      "id": "string - unique synthetic ID (e.g. suggestion-1)",
      "title": "short, actionable reminder title (max 8-10 words)",
      "description": "concise description explaining why this reminder is relevant based on the chat context",
      "suggested_due_date": "ISO date (YYYY-MM-DD) if a deadline is implied, otherwise null",
      "priority": "one of: 'low', 'medium', 'high' (based on urgency and importance)",
      "context": "brief excerpt from the chat that supports this suggestion",
      "confidence": 0.0-1.0 (how confident you are that this is a valid reminder suggestion)
    }},
    ...
  ]
}}

Guidelines:
- Only suggest reminders that are clearly implied or would be helpful based on the conversation
- Focus on actionable items, not just general topics
- Limit to 5-7 most relevant suggestions
- If no good suggestions exist, return {{"suggestions": []}}
- Do NOT include any explanation text, ONLY the JSON object
- Do NOT wrap the JSON in markdown code fences

Return the JSON now.
"""
        
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
                        "You are a precise JSON-producing reminder suggestion engine. "
                        "You MUST return only valid JSON with no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
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
            
            suggestions = parsed.get("suggestions", [])
            if not isinstance(suggestions, list):
                suggestions = []
            
            return {"suggestions": suggestions}
            
        except Exception as e:
            print(f"Error generating reminder suggestions from LLM: {e}")
            return {"suggestions": []}
            
    except Exception as e:
        print(f"Error in generate_context_based_suggestions: {e}")
        return {"suggestions": []}


def create_reminder_from_task(
    task_id: str,
    title: str,
    description: Optional[str] = None,
    due_date: Optional[str] = None,
    assignee: Optional[str] = None,
    reminder_time: Optional[str] = None,
) -> Dict:
    """Create a reminder from an action item/task with one-click creation.
    
    Args:
        task_id: ID of the source task/action item
        title: Reminder title
        description: Optional reminder description
        due_date: Optional due date in ISO format (YYYY-MM-DD)
        assignee: Optional assignee name
        reminder_time: Optional reminder time in ISO datetime format
    
    Returns:
        Dict with the created reminder:
        {
            "id": "reminder-id",
            "title": "...",
            "description": "...",
            "due_date": "...",
            "assignee": "...",
            "reminder_time": "...",
            "created_at": "...",
            "source_task_id": "...",
            "status": "pending"
        }
    """
    try:
        reminder_id = f"reminder-{uuid.uuid4().hex[:8]}"
        created_at = datetime.now().isoformat()
        
        # If reminder_time is not provided but due_date is, set reminder to 1 day before due date
        if not reminder_time and due_date:
            try:
                due_dt = datetime.fromisoformat(due_date.replace("Z", "+00:00"))
                reminder_dt = due_dt - timedelta(days=1)
                reminder_time = reminder_dt.isoformat()
            except Exception:
                pass
        
        reminder = {
            "id": reminder_id,
            "title": title,
            "description": description,
            "due_date": due_date,
            "assignee": assignee,
            "reminder_time": reminder_time,
            "created_at": created_at,
            "source_task_id": task_id,
            "status": "pending",
        }
        
        return reminder
        
    except Exception as e:
        print(f"Error creating reminder from task: {e}")
        raise ValueError(f"Failed to create reminder: {str(e)}")
