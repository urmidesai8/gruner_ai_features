from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List, Optional, Dict
from fastapi.responses import JSONResponse
import json
from pydantic import BaseModel
from ...models.schemas import (
    FeatureRequest,
    chat_history,
    SummarizeRequest,
    SmartRepliesRequest,
    ReminderSuggestionRequest,
    ReminderCreateRequest,
    TranslationRequest,
    TextTranslationRequest,
)
from ...services.ai_service import call_groq_ai, transcribe_audio
from ...services.summarizer import generate_chat_summary, generate_text_summary
from ...services.task_classifier import extract_tasks_from_messages
from ...services.translation_service import translate_messages_batch, translate_text
from ...services.reminder_service import (
    generate_context_based_suggestions,
    create_reminder_from_task,
)
from ...services.translation_service import translate_messages_batch

class TextSummaryRequest(BaseModel):
    text: str

class AIToggleRequest(BaseModel):
    enabled: bool

router = APIRouter()

@router.get("/ai-status")
async def get_ai_status() -> JSONResponse:
    """Get the current AI enabled status."""
    return JSONResponse(content={"ai_enabled": chat_history.get_ai_enabled()})

@router.post("/ai-toggle")
async def toggle_ai(request: AIToggleRequest) -> JSONResponse:
    """Toggle AI features on/off globally.
    
    When AI is OFF:
    - All AI features become unavailable
    - Messages sent during this time are not considered for AI analysis
    - Only messages sent when AI is ON will be used for AI features
    
    Exception: Chat Summary always uses all messages regardless of AI state.
    """
    chat_history.set_ai_enabled(request.enabled)
    return JSONResponse(content={
        "ai_enabled": chat_history.get_ai_enabled(),
        "message": f"AI features {'enabled' if request.enabled else 'disabled'}"
    })

@router.post("/prioritize")
async def prioritize_messages(messages: List[FeatureRequest]):
    """Classify priority for a list of messages.
    
    Note: Only processes messages that were created when AI was enabled.
    """
    # Check if AI is enabled
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    
    if not messages:
        return {}
    
    # Filter to only include messages created when AI was enabled
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg['message_id'] for msg in ai_enabled_messages}
    filtered_messages = [m for m in messages if m.id in ai_enabled_ids]
    
    if not filtered_messages:
        return {}
    
    prompt_items = [f"ID: {m.id} | Msg: {m.message}" for m in filtered_messages]
    prompt_text = "\n".join(prompt_items)
    
    prompt = f"""
    Analyze the priority of the following messages. 
    Return a JSON object where keys are IDs and values are one of: 'Low', 'Normal', 'High', 'Urgent'.
    
    Messages:
    {prompt_text}
    
    Return ONLY valid JSON.
    """
    
    try:
        response_text = call_groq_ai(prompt)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        results = json.loads(response_text)
    except Exception:
        results = {m.id: "Normal" for m in filtered_messages}
        
    return JSONResponse(content=results)

@router.post("/moderate")
async def moderate_messages(messages: List[FeatureRequest]):
    """Check moderation status.
    
    Note: Only processes messages that were created when AI was enabled.
    """
    # Check if AI is enabled
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    
    if not messages:
        return {}
    
    # Filter to only include messages created when AI was enabled
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg['message_id'] for msg in ai_enabled_messages}
    filtered_messages = [m for m in messages if m.id in ai_enabled_ids]
    
    if not filtered_messages:
        return {}
    
    prompt_items = [f"ID: {m.id} | Msg: {m.message}" for m in filtered_messages]
    prompt_text = "\n".join(prompt_items)
    
    prompt = f"""
    Check these messages for spam, scams, or abuse.
    Return a JSON object where keys are IDs and values are objects like {{ "safe": true }} or {{ "safe": false, "reason": "spam" }}.
    
    Messages:
    {prompt_text}
    
    Return ONLY valid JSON.
    """
    
    try:
        response_text = call_groq_ai(prompt)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        results = json.loads(response_text)
    except Exception:
        results = {m.id: {"safe": True} for m in filtered_messages}

    return JSONResponse(content=results)

@router.post("/smart-replies")
async def smart_replies(request: SmartRepliesRequest):
    """Generate smart replies with specified tone.
    
    Request body:
    - messages: List of messages to generate replies for
    - tone: Tone of the reply (auto, professional, casual, friendly, formal) - default: auto
    
    Note: Only processes messages that were created when AI was enabled.
    """
    # Check if AI is enabled
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    
    if not request.messages:
        return JSONResponse(content={"suggestions": []})
    
    # Filter to only include messages created when AI was enabled
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg['message_id'] for msg in ai_enabled_messages}
    filtered_messages = [m for m in request.messages if m.id in ai_enabled_ids]
    
    if not filtered_messages:
        return JSONResponse(content={"suggestions": []})
    
    last_msg = filtered_messages[-1]
    tone = request.tone.lower()
    
    # Define tone instructions
    tone_instructions = {
        "auto": "Match the tone of the original message automatically.",
        "professional": "Use a professional, business-appropriate tone. Be formal, clear, and respectful.",
        "casual": "Use a casual, relaxed tone. Be friendly and conversational, like talking to a friend.",
        "friendly": "Use a warm, friendly tone. Be approachable, positive, and engaging.",
        "formal": "Use a formal, official tone. Be polite, structured, and maintain proper etiquette."
    }
    
    tone_instruction = tone_instructions.get(tone, tone_instructions["auto"])
    
    prompt = f"""
    Generate 3 short, context-aware reply suggestions for the following message:
    "{last_msg.message}"
    
    Tone requirement: {tone_instruction}
    
    The replies should:
    - Be contextually appropriate
    - Match the specified tone: {tone}
    - Be concise (1-2 sentences each)
    - Be natural and conversational
    
    Return a JSON object: {{ "suggestions": ["Reply 1", "Reply 2", "Reply 3"] }}
    Return ONLY valid JSON.
    """
    
    try:
        response_text = call_groq_ai(prompt)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        result = json.loads(response_text)
    except Exception as e:
        print(f"Error in smart_replies: {e}")
        result = {"suggestions": []}

    return JSONResponse(content=result)


@router.post("/chat-summarize")
async def summarize_chat(request: SummarizeRequest) -> JSONResponse:
    """
    Generate chat summary.

    Request body:
    - username: Optional username to get personalized "What did I miss?" summary.
    - total_messages: Optional number of recent messages to consider (default: 100)
    
    Note: Chat Summary ALWAYS uses ALL messages regardless of AI toggle state.
    This is an exception to the AI filtering rule.
    """
    try:
        # Chat summary always uses all messages (exception to AI filtering)
        if request.username:
            messages = chat_history.get_unread_messages(request.username)
            if not messages:
                messages = chat_history.get_all_messages_for_summary()
        else:
            messages = chat_history.get_all_messages_for_summary()

        summary = generate_chat_summary(
            messages, 
            username=request.username, 
            total_messages=request.total_messages
        )

        if request.username:
            chat_history.mark_as_read(request.username)

        return JSONResponse(content=summary)

    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}") from e


@router.get("/unread-messages")
async def get_messages(username: Optional[str] = None) -> JSONResponse:
    """Get all chat messages or unread messages for a user."""
    if username:
        return JSONResponse(
            content={
                "messages": chat_history.get_unread_messages(username),
                "unread_count": chat_history.get_unread_count(username),
            }
        )

    return JSONResponse(
        content={
            "messages": chat_history.get_all_messages(),
            "total_count": len(chat_history.messages),
        }
    )


@router.post("/tasks-classifier")
async def classify_tasks(username: Optional[str] = None) -> JSONResponse:
    """Identify tasks/todos in chat messages and return them in a structured format.

    - If `username` is provided, prefer that user's unread messages; if none, use all.
    - Otherwise, run on the entire chat history.
    
    Note: Only processes messages that were created when AI was enabled.
    """
    # Check if AI is enabled
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    
    try:
        # Get AI-enabled messages only
        if username:
            all_messages = chat_history.get_unread_messages(username)
            if not all_messages:
                all_messages = chat_history.get_ai_enabled_messages()
        else:
            all_messages = chat_history.get_ai_enabled_messages()
        
        messages = all_messages

        result = extract_tasks_from_messages(messages)
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Error classifying tasks: {str(e)}"
        ) from e


@router.post("/translate")
async def translate_chat_messages(request: List[Dict]) -> JSONResponse:
    try:
        result = translate_messages_batch(request)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Translation failed: {str(e)}"
        ) from e


@router.post("/translate-text")
async def translate_text_endpoint(request: TextTranslationRequest) -> JSONResponse:
    """Translate raw text into a target language."""
    try:
        result = translate_text(request.text, request.target_language)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Translation failed: {str(e)}"
        ) from e


@router.post("/smart-reminders/suggestions")
async def get_reminder_suggestions(
    request: ReminderSuggestionRequest,
) -> JSONResponse:
    """Generate context-based reminder suggestions from chat history.

    Request body:
    - username: Optional username to get personalized suggestions
    - context_window: Optional number of recent messages to consider (default: all)
    
    Note: Only processes messages that were created when AI was enabled.

    Returns:
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
    # Check if AI is enabled
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    
    try:
        result = generate_context_based_suggestions(
            username=request.username, context_window=request.context_window
        )
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Error generating reminder suggestions: {str(e)}",
        ) from e


@router.post("/smart-reminders/create")
async def create_reminder(request: ReminderCreateRequest) -> JSONResponse:
    """Create a reminder from an action item with one-click creation.

    Request body:
    - task_id: ID of the source task/action item (required)
    - title: Reminder title (required)
    - description: Optional reminder description
    - due_date: Optional due date in ISO format (YYYY-MM-DD)
    - assignee: Optional assignee name
    - reminder_time: Optional reminder time in ISO datetime format

    Returns:
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
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Error creating reminder: {str(e)}"
        ) from e


@router.post("/translate")
async def translate_messages(requests: List[TranslationRequest]) -> JSONResponse:
    """Translate chat messages into a target language.

    Request body: List of objects
    
    Note: Only processes messages that were created when AI was enabled.
    [
      {
        "id": "message-id",
        "text": "Hello, world!",
        "target_language": "es"
      },
      ...
    ]

    Returns:
    {
      "translations": {
        "message-id": {
          "translated_text": "Hola, mundo!",
          "detected_language": "en"
        },
        ...
      }
    }
    """
    # Check if AI is enabled
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    
    if not requests:
        return JSONResponse(content={"translations": {}})
    
    # Filter to only include messages created when AI was enabled
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg['message_id'] for msg in ai_enabled_messages}
    filtered_requests = [r for r in requests if r.id in ai_enabled_ids]
    
    if not filtered_requests:
        return JSONResponse(content={"translations": {}})

    try:
        payload = [r.dict() for r in filtered_requests]
        result = translate_messages_batch(payload)
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Error translating messages: {str(e)}"
        ) from e

@router.post("/transcribe")
async def transcribe_voice_note(file: UploadFile = File(...)) -> JSONResponse:
    """
    Transcribe uploaded audio file.
    """
    try:
        # We need to pass the file-like object to the service along with its filename
        # so Groq/httpx knows the file type (e.g. "audio.mp3").
        # We pass a tuple (filename, file_obj) which is supported by the library.
        transcription_text = transcribe_audio((file.filename, file.file))
        
        if transcription_text.startswith("Error"):
             raise HTTPException(status_code=500, detail=transcription_text)

        return JSONResponse(content={"transcription": transcription_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}") from e

@router.post("/summarize-text")
async def summarize_text(request: TextSummaryRequest) -> JSONResponse:
    """
    Summarize raw text (e.g. from transcription).
    """
    try:
        result = generate_text_summary(request.text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}") from e
