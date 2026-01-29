from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List, Optional
from fastapi.responses import JSONResponse
import json
from pydantic import BaseModel
from ...models.schemas import (
    FeatureRequest,
    chat_history,
    SummarizeRequest,
    ReminderSuggestionRequest,
    ReminderCreateRequest,
    TranslationRequest,
)
from ...services.ai_service import call_groq_ai, transcribe_audio
from ...services.summarizer import generate_chat_summary, generate_text_summary
from ...services.task_classifier import extract_tasks_from_messages
from ...services.reminder_service import (
    generate_context_based_suggestions,
    create_reminder_from_task,
)
from ...services.translation_service import translate_messages_batch

class TextSummaryRequest(BaseModel):
    text: str

router = APIRouter()

@router.post("/prioritize")
async def prioritize_messages(messages: List[FeatureRequest]):
    """Classify priority for a list of messages"""
    if not messages:
        return {}
    
    prompt_items = [f"ID: {m.id} | Msg: {m.message}" for m in messages]
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
        results = {m.id: "Normal" for m in messages}
        
    return JSONResponse(content=results)

@router.post("/moderate")
async def moderate_messages(messages: List[FeatureRequest]):
    """Check moderation status"""
    if not messages:
        return {}
    
    prompt_items = [f"ID: {m.id} | Msg: {m.message}" for m in messages]
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
        results = {m.id: {"safe": True} for m in messages}

    return JSONResponse(content=results)

@router.post("/smart-replies")
async def smart_replies(messages: List[FeatureRequest]):
    """Generate smart replies"""
    if not messages:
        return {"suggestions": []}
    
    last_msg = messages[-1]
    
    prompt = f"""
    Generate 3 short, context-aware reply suggestions for the following message:
    "{last_msg.message}"
    
    Return a JSON object: {{ "suggestions": ["Yes", "No", "maybe"] }}
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
    """
    try:
        if request.username:
            messages = chat_history.get_unread_messages(request.username)
            if not messages:
                messages = chat_history.get_all_messages()
        else:
            messages = chat_history.get_all_messages()

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
    """
    try:
        if username:
            messages = chat_history.get_unread_messages(username)
            if not messages:
                messages = chat_history.get_all_messages()
        else:
            messages = chat_history.get_all_messages()

        result = extract_tasks_from_messages(messages)
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Error classifying tasks: {str(e)}"
        ) from e


@router.post("/smart-reminders/suggestions")
async def get_reminder_suggestions(
    request: ReminderSuggestionRequest,
) -> JSONResponse:
    """Generate context-based reminder suggestions from chat history.

    Request body:
    - username: Optional username to get personalized suggestions
    - context_window: Optional number of recent messages to consider (default: all)

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
    if not requests:
        return JSONResponse(content={"translations": {}})

    try:
        payload = [r.dict() for r in requests]
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
