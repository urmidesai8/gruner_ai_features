"""
Chat, AI toggle, prioritization, moderation, smart replies, summarization,
tasks, translation, reminders, and chat search endpoints.
"""

import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...models.schemas import (
    AIAnalysisRequest,
    chat_history,
    SummarizeRequest,
    SmartRepliesRequest,
    DraftResponseRequest,
    ReminderSuggestionRequest,
    ReminderCreateRequest,
    TranslationRequest,
    TextTranslationRequest,
)
from ...services.ai_service import call_groq_ai
from ...services.summarizer import generate_chat_summary
from ...services.task_classifier import extract_tasks_from_messages
from ...services.translation_service import translate_messages_batch, translate_text
from ...services.reminder_service import (
    generate_context_based_suggestions,
    create_reminder_from_task,
)
from ...services.local_feature_service import (
    generate_smart_replies_local,
    analyze_prioritization_local,
    analyze_moderation_local,
    analyze_reminders_local,
)
from ...services.chat_search_service import search_chat_messages


class AIToggleRequest(BaseModel):
    enabled: bool


class ChatSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    min_score: Optional[float] = 0.3
    username: Optional[str] = None


router = APIRouter()


@router.get("/ai-status")
async def get_ai_status() -> JSONResponse:
    """Get the current AI enabled status."""
    return JSONResponse(content={"ai_enabled": chat_history.get_ai_enabled()})


@router.post("/ai-toggle")
async def toggle_ai(request: AIToggleRequest) -> JSONResponse:
    """Toggle AI features on/off globally."""
    chat_history.set_ai_enabled(request.enabled)
    return JSONResponse(content={
        "ai_enabled": chat_history.get_ai_enabled(),
        "message": f"AI features {'enabled' if request.enabled else 'disabled'}"
    })


@router.post("/prioritize")
async def prioritize_messages(request: AIAnalysisRequest) -> JSONResponse:
    """Classify priority for a list of messages."""
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    if not request.messages:
        return JSONResponse(content={})
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg["message_id"] for msg in ai_enabled_messages}
    filtered_messages = [m for m in request.messages if m.id in ai_enabled_ids]
    if not filtered_messages:
        return JSONResponse(content={})
    if request.model and "/" in request.model and "openai" not in request.model:
        results = analyze_prioritization_local(filtered_messages, request.model)
        return JSONResponse(content=results)
    prompt_items = [f"ID: {m.id} | Msg: {m.message}" for m in filtered_messages]
    prompt_text = "\n".join(prompt_items)
    prompt = f"""Analyze the priority of the following messages.
Return a JSON object where keys are IDs and values are one of: 'Low', 'Normal', 'High', 'Urgent'.
Messages:
{prompt_text}
Return ONLY valid JSON."""
    try:
        response_text = call_groq_ai(prompt, model_name=request.model)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        results = json.loads(response_text)
    except Exception:
        results = {m.id: "Normal" for m in filtered_messages}
    return JSONResponse(content=results)


@router.post("/moderate")
async def moderate_messages(request: AIAnalysisRequest) -> JSONResponse:
    """Check moderation status."""
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    if not request.messages:
        return JSONResponse(content={})
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg["message_id"] for msg in ai_enabled_messages}
    filtered_messages = [m for m in request.messages if m.id in ai_enabled_ids]
    if not filtered_messages:
        return JSONResponse(content={})
    if request.model and "/" in request.model and "openai" not in request.model:
        results = analyze_moderation_local(filtered_messages, request.model)
        return JSONResponse(content=results)
    prompt_items = [f"ID: {m.id} | Msg: {m.message}" for m in filtered_messages]
    prompt_text = "\n".join(prompt_items)
    prompt = f"""Check these messages for spam, scams, or abuse.
Return a JSON object where keys are IDs and values are objects like {{ "safe": true }} or {{ "safe": false, "reason": "spam" }}.
Messages:
{prompt_text}
Return ONLY valid JSON."""
    try:
        response_text = call_groq_ai(prompt, model_name=request.model)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        results = json.loads(response_text)
    except Exception:
        results = {m.id: {"safe": True} for m in filtered_messages}
    return JSONResponse(content=results)


@router.post("/smart-replies")
async def smart_replies(request: SmartRepliesRequest) -> JSONResponse:
    """Generate smart replies with specified tone."""
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    if not request.messages:
        return JSONResponse(content={"suggestions": []})
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg["message_id"] for msg in ai_enabled_messages}
    filtered_messages = [m for m in request.messages if m.id in ai_enabled_ids]
    if not filtered_messages:
        return JSONResponse(content={"suggestions": []})
    last_msg = filtered_messages[-1]
    tone = request.tone.lower()
    tone_instructions = {
        "auto": "Match the tone of the original message automatically.",
        "professional": "Use a professional, business-appropriate tone. Be formal, clear, and respectful.",
        "casual": "Use a casual, relaxed tone. Be friendly and conversational, like talking to a friend.",
        "friendly": "Use a warm, friendly tone. Be approachable, positive, and engaging.",
        "formal": "Use a formal, official tone. Be polite, structured, and maintain proper etiquette."
    }
    tone_instruction = tone_instructions.get(tone, tone_instructions["auto"])
    if request.model and "/" in request.model and "openai" not in request.model:
        suggestions = generate_smart_replies_local(filtered_messages, request.model)
        return JSONResponse(content={"suggestions": suggestions})
    prompt = f"""Generate 3 short, context-aware reply suggestions for the following message:
"{last_msg.message}"
Tone requirement: {tone_instruction}
The replies should be contextually appropriate, match the specified tone: {tone}, be concise (1-2 sentences each), and be natural and conversational.
Return a JSON object: {{ "suggestions": ["Reply 1", "Reply 2", "Reply 3"] }}
Return ONLY valid JSON."""
    try:
        response_text = call_groq_ai(prompt, model_name=request.model)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        result = json.loads(response_text)
    except Exception as e:
        print(f"Error in smart_replies: {e}")
        result = {"suggestions": []}
    return JSONResponse(content=result)


@router.post("/draft-response")
async def draft_response(request: DraftResponseRequest) -> JSONResponse:
    """
    Fix grammar and rewrite a single user message in a specified tone.

    Request body:
    {
      "message": "string - the original user message",
      "tone": "one of: professional, casual, friendly, formal, auto"
    }

    Response:
    {
      "original": "original message",
      "grammar_fixed": "grammatically correct version, same tone as original",
      "tone_rewritten": "rewritten version in the requested tone",
      "tone": "normalized tone value actually used"
    }
    """
    if not chat_history.get_ai_enabled():
        raise HTTPException(
            status_code=403,
            detail="AI features are currently disabled. Please enable AI to use this feature.",
        )

    text = (request.message or "").strip()
    if not text:
        return JSONResponse(
            content={
                "original": request.message or "",
                "grammar_fixed": "",
                "tone_rewritten": "",
                "tone": (request.tone or "auto").lower(),
            }
        )

    tone = (request.tone or "auto").lower()
    tone_instructions = {
        "auto": "Match the tone of the original message automatically.",
        "professional": "Use a professional, business-appropriate tone. Be formal, clear, and respectful.",
        "casual": "Use a casual, relaxed tone. Be friendly and conversational, like talking to a friend.",
        "friendly": "Use a warm, friendly tone. Be approachable, positive, and engaging.",
        "formal": "Use a formal, official tone. Be polite, structured, and maintain proper etiquette.",
    }
    tone_instruction = tone_instructions.get(tone, tone_instructions["auto"])

    primary_model = "llama-3.1-8b-instant"
    fallback_model = "llama-3.3-70b-versatile"

    system_prompt = (
        "You are a writing assistant that cleans up grammar and rewrites text in different tones.\n"
        "Given a single user message and a tone requirement, you MUST return ONLY valid JSON with this structure:\n"
        "{\n"
        '  "original": "the original input message",\n'
        '  "grammar_fixed": "same content with correct grammar, spelling, and punctuation, keeping the original tone",\n'
        '  "tone_rewritten": "a rewritten version that preserves meaning but matches the requested tone",\n'
        '  "tone": "the final tone you used (e.g. professional, casual, friendly, formal, auto)"\n'
        "}\n"
        "Guidelines:\n"
        "- Do NOT change the factual meaning of the message.\n"
        "- Be concise and natural.\n"
        "- Never add extra explanation or commentary outside of the JSON.\n"
    )

    user_prompt = (
        f"Original message:\n{text}\n\n"
        f"Tone requirement: {tone_instruction}\n"
        "Return the JSON object as specified."
    )

    def _call_model(model_name: str) -> dict:
        response_text = call_groq_ai(
            f"{system_prompt}\n\n{user_prompt}",
            model_name=model_name,
        )
        if not response_text:
            raise RuntimeError(f"Empty response from Groq model {model_name}")

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]

        response_text = response_text.strip()
        if not response_text:
            raise RuntimeError(f"Blank response from Groq model {model_name} after stripping")

        parsed = json.loads(response_text)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Groq model {model_name} returned non-dict JSON")

        return {
            "original": parsed.get("original", text),
            "grammar_fixed": parsed.get("grammar_fixed", text),
            "tone_rewritten": parsed.get("tone_rewritten", text),
            "tone": parsed.get("tone", tone),
        }

    for model_name in (primary_model, fallback_model):
        try:
            return JSONResponse(content=_call_model(model_name))
        except Exception as e:
            print(f"Error generating draft response with model {model_name}: {e}")
            continue

    return JSONResponse(
        content={
            "original": text,
            "grammar_fixed": text,
            "tone_rewritten": text,
            "tone": tone,
            "notice": "Draft response temporarily unavailable; returning original text.",
        }
    )

@router.post("/chat-summarize")
async def summarize_chat(request: SummarizeRequest) -> JSONResponse:
    """Generate chat summary. Uses all messages regardless of AI toggle."""
    try:
        if request.username:
            messages = chat_history.get_unread_messages(request.username)
            if not messages:
                messages = chat_history.get_all_messages_for_summary()
        else:
            messages = chat_history.get_all_messages_for_summary()
        summary = generate_chat_summary(
            messages,
            username=request.username,
            total_messages=request.total_messages,
            model=request.model
        )
        if request.username:
            chat_history.mark_as_read(request.username)
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}") from e


@router.get("/unread-messages")
async def get_messages(username: Optional[str] = None) -> JSONResponse:
    """Get all chat messages or unread messages for a user."""
    if username:
        return JSONResponse(content={
            "messages": chat_history.get_unread_messages(username),
            "unread_count": chat_history.get_unread_count(username),
        })
    all_messages = chat_history.get_all_messages()
    return JSONResponse(content={
        "messages": all_messages,
        "total_count": len(all_messages),
    })


@router.post("/tasks-classifier")
async def classify_tasks(username: Optional[str] = None, model: Optional[str] = None) -> JSONResponse:
    """Identify tasks/todos in chat messages."""
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    try:
        if username:
            all_messages = chat_history.get_unread_messages(username)
            if not all_messages:
                all_messages = chat_history.get_ai_enabled_messages()
        else:
            all_messages = chat_history.get_ai_enabled_messages()
        result = extract_tasks_from_messages(all_messages, model=model)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying tasks: {str(e)}") from e


@router.post("/translate")
async def translate_chat_messages(requests: List[TranslationRequest]) -> JSONResponse:
    """Translate chat messages into a target language."""
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    if not requests:
        return JSONResponse(content={"translations": {}})
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg["message_id"] for msg in ai_enabled_messages}
    filtered_requests = [r for r in requests if r.id in ai_enabled_ids]
    if not filtered_requests:
        return JSONResponse(content={"translations": {}})
    try:
        model = filtered_requests[0].model if filtered_requests else None
        payload = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in filtered_requests]
        result = translate_messages_batch(payload, model=model)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}") from e


@router.post("/translate-text")
async def translate_text_endpoint(request: TextTranslationRequest) -> JSONResponse:
    """Translate raw text into a target language."""
    try:
        result = translate_text(request.text, request.target_language, model=request.model)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}") from e


@router.post("/smart-reminders/suggestions")
async def get_reminder_suggestions(request: ReminderSuggestionRequest) -> JSONResponse:
    """Generate context-based reminder suggestions from chat history."""
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    try:
        if request.model and "/" in request.model and "openai" not in request.model:
            result = analyze_reminders_local(chat_history.get_ai_enabled_messages(), request.model)
            return JSONResponse(content=result)
        result = generate_context_based_suggestions(
            username=request.username, context_window=request.context_window, model=request.model
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating reminder suggestions: {str(e)}") from e


@router.post("/smart-reminders/create")
async def create_reminder(request: ReminderCreateRequest) -> JSONResponse:
    """Create a reminder from an action item."""
    try:
        reminder = create_reminder_from_task(
            task_id=request.task_id,
            title=request.title,
            description=request.description,
            due_date=request.due_date,
            assignee=request.assignee,
            reminder_time=request.reminder_time,
        )
        return JSONResponse(content=reminder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating reminder: {str(e)}") from e


@router.post("/chat-search")
async def chat_search(request: ChatSearchRequest) -> JSONResponse:
    """Perform semantic search over conversation history."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query is required.")
    try:
        results = search_chat_messages(
            query=request.query,
            limit=request.limit or 10,
            min_score=request.min_score or 0.3,
            username=request.username,
        )
        return JSONResponse(content={
            "results": results,
            "total_found": len(results),
            "query": request.query,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat search failed: {str(e)}") from e
