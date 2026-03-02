from fastapi import APIRouter, BackgroundTasks, HTTPException, File, Form, UploadFile
from typing import List, Optional, Dict
from fastapi.responses import JSONResponse
import json
import shutil
import os
from pathlib import Path
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from ...models.schemas import (
    FeatureRequest,
    AIAnalysisRequest,
    chat_history,
    SummarizeRequest,
    SmartRepliesRequest,
    ReminderSuggestionRequest,
    ReminderCreateRequest,
    TranslationRequest,
    TextTranslationRequest,
)
from ...services.ai_service import (
    call_groq_ai,
    transcribe_audio,
    transcribe_audio_with_timestamps,
    transcribe_audio_whisper_local,
    transcribe_audio_vibevoice,
    transcribe_audio_seamless_m4t,
    transcribe_audio_whisper_local,
    format_meeting_transcription,
)
from ...services.summarizer import generate_chat_summary, generate_text_summary
from ...services.task_classifier import extract_tasks_from_messages
from ...services.translation_service import translate_messages_batch, translate_text
from ...services.memory_service import (
    upsert_individual_chat_memories,
    upsert_group_chat_memories,
    search_individual_memories,
    search_group_memories,
)
from ...services.reminder_service import (
    generate_context_based_suggestions,
    create_reminder_from_task,
)
from ...services.translation_service import translate_messages_batch
from ...services.local_feature_service import (
    generate_smart_replies_local,
    analyze_prioritization_local,
    analyze_moderation_local,
    analyze_reminders_local
)
from ...services.meeting_task_service import extract_meeting_tasks
from ...services.chat_search_service import search_chat_messages
from ...services.meeting_transcription_service import store_meeting_transcription, ask_meeting_question
from ...services.meeting_agenda_service import analyze_agenda_vs_discussion
from ...services.meeting_intelligence_service import generate_meeting_intelligence
from ...services.document_extraction_service import extract_document_text_and_tables
from ...services.document_extraction_qdrant_service import (
    store_document_extraction,
    get_document_extraction,
    get_document_extraction_by_original_name_and_user,
)

class TextSummaryRequest(BaseModel):
    text: str
    model: Optional[str] = None


class AudioFileRequest(BaseModel):
    filename: str


class MeetingRecordingSummaryRequest(BaseModel):
    """
    Request body for meeting recording post-processing.

    This is triggered from the dedicated meeting transcription UI
    *after* the raw transcription has been generated via /transcribe-file.
    """

    transcription: str
    model: Optional[str] = None


class MeetingTasksRequest(BaseModel):
    """
    Request body for meeting task extraction.
    
    Extracts tasks from meeting transcriptions using spaCy NER + rule-based detection.
    """
    text: str
    model: Optional[str] = None  # spaCy model name (e.g., "en_core_web_sm" or "en_core_web_trf")


class DocumentTextExtractionRequest(BaseModel):
    """
    Request body for document text extraction.

    This should be called after /upload-document, using the returned doc_id.
    Pass uploaded_user_id and doc_upload_time from the upload response to store them in Qdrant.
    """

    doc_id: str  # Filename (doc_id) as returned by /upload-document
    model: Optional[str] = None  # Optional LLM model for formatting/summary
    uploaded_user_id: Optional[str] = None  # From /upload-document; stored in Qdrant
    doc_upload_time: Optional[str] = None  # From /upload-document; stored in Qdrant
    original_name: Optional[str] = None  # Original filename; used for de-duplication


class DocumentQARequest(BaseModel):
    """
    Request body for document question-answering based on extracted text in Qdrant.
    """

    doc_id: str
    uploaded_user_id: str
    question: str
    model: Optional[str] = None


class ChatSearchRequest(BaseModel):
    """
    Request body for semantic chat search.
    
    Searches through conversation history to find relevant messages.
    """
    query: str
    limit: Optional[int] = 10
    min_score: Optional[float] = 0.3
    username: Optional[str] = None  # Filter by sender username


class MeetingAskRequest(BaseModel):
    """
    Request body for asking questions about meeting transcriptions.
    """
    query: str
    participant_id: str  # User ID asking the question (filters meetings they participated in)
    limit: Optional[int] = 3  # Maximum number of relevant meetings to consider
    score_threshold: Optional[float] = 0.3  # Minimum similarity score for relevant meetings
    model: Optional[str] = None  # Optional LLM model for generating answer


class MeetingAudioFileRequest(BaseModel):
    """
    Request body for meeting-specific transcription directly from an uploaded file.

    This is similar to AudioFileRequest used by /transcribe-file but will
    additionally format the transcript into a speaker-separated meeting view.
    """

    filename: str
    model: Optional[str] = None  # LLM model for speaker formatting
    asr_model: Optional[str] = "whisper-large-v3"  # ASR: whisper-large-v3 | openai/whisper-large-v3 (local) | microsoft/VibeVoice-ASR | facebook/seamless-m4t-medium
    participant_ids: Optional[List[str]] = None  # List of participant user IDs
    meeting_agenda: Optional[str] = None  # Meeting agenda/topic
    store_in_qdrant: Optional[bool] = True  # Whether to store transcription in Qdrant


class AgendaItem(BaseModel):
    """One planned agenda topic with allocated minutes."""
    title: str
    planned_minutes: int


class AgendaIntelligenceRequest(BaseModel):
    """
    Request body for Agenda vs Discussion Intelligence.
    Compares planned agenda vs actual discussion (actual meeting time is user input).
    """
    transcript: str
    agenda_items: List[AgendaItem]
    actual_meeting_minutes: int
    model: Optional[str] = None


class MeetingIntelligenceRequest(BaseModel):
    """
    Request body for Meeting Intelligence & Insights.
    Higher-level analytics: talk-time, sentiment/tension, decisions, blockers, dominant topics.
    """
    transcript: str
    segments: Optional[List[Dict]] = None  # Optional: [{start, end, text, speaker?}]
    model: Optional[str] = None


class AIToggleRequest(BaseModel):
    enabled: bool


class IndividualMemoryRefreshRequest(BaseModel):
    """
    Request to refresh AI memories for a 1:1 chat.

    NOTE: For now, this uses the global chat history as a single room.
    In a multi-room system, you would filter messages by chat_id.
    """

    user1_id: str
    user1_name: str
    user2_id: str
    user2_name: str
    model: Optional[str] = None


class GroupMemoryRefreshRequest(BaseModel):
    """
    Request to refresh AI memories for a group chat.
    """

    group_id: str
    group_name: str
    participants: List[Dict[str, str]]  # [{\"user_id\": ..., \"name\": ...}]
    model: Optional[str] = None


class IndividualMemorySearchRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 10


class GroupMemorySearchRequest(BaseModel):
    group_id: str
    query: str
    limit: int = 10

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


@router.post("/ai-memory/refresh-individual")
async def refresh_individual_memory(
    request: IndividualMemoryRefreshRequest,
) -> JSONResponse:
    """
    Refresh AI memories for a 1:1 chat and upsert into Qdrant `individual_chats` collection.

    For this POC, we use the entire chat history as the context. In a multi-room
    system you would filter messages belonging to a specific chat.
    """
    # Get all messages that were created when AI was enabled (respecting consent)
    messages = chat_history.get_ai_enabled_messages()

    count = upsert_individual_chat_memories(
        user1_id=request.user1_id,
        user1_name=request.user1_name,
        user2_id=request.user2_id,
        user2_name=request.user2_name,
        messages=messages,
        model=request.model,
    )

    return JSONResponse(
        content={
            "status": "ok",
            "memories_upserted": count,
            "collection": "individual_chats",
        }
    )


@router.post("/ai-memory/refresh-group")
async def refresh_group_memory(
    request: GroupMemoryRefreshRequest,
) -> JSONResponse:
    """
    Refresh AI memories for a group chat and upsert into Qdrant `group_chats` collection.

    For this POC, we use the entire chat history as the context. In a multi-room
    system you would filter messages belonging to a specific group/chat ID.
    """
    messages = chat_history.get_ai_enabled_messages()

    count = upsert_group_chat_memories(
        group_id=request.group_id,
        group_name=request.group_name,
        participants=request.participants,
        messages=messages,
        model=request.model,
    )

    return JSONResponse(
        content={
            "status": "ok",
            "memories_upserted": count,
            "collection": "group_chats",
        }
    )


@router.post("/ai-memory/search-individual")
async def search_individual_memory(
    request: IndividualMemorySearchRequest,
) -> JSONResponse:
    """
    Semantic search over individual chat memories for a given user.
    """
    results = search_individual_memories(
        user_id=request.user_id,
        query=request.query,
        limit=request.limit,
    )
    return JSONResponse(content={"results": results})


@router.post("/ai-memory/search-group")
async def search_group_memory(
    request: GroupMemorySearchRequest,
) -> JSONResponse:
    """
    Semantic search over group chat memories for a given group_id.
    """
    results = search_group_memories(
        group_id=request.group_id,
        query=request.query,
        limit=request.limit,
    )
    return JSONResponse(content={"results": results})

@router.post("/prioritize")
async def prioritize_messages(request: AIAnalysisRequest):
    """Classify priority for a list of messages.
    
    Note: Only processes messages that were created when AI was enabled.
    """
    # Check if AI is enabled
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    
    if not request.messages:
        return {}
    
    # Filter to only include messages created when AI was enabled
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg['message_id'] for msg in ai_enabled_messages}
    filtered_messages = [m for m in request.messages if m.id in ai_enabled_ids]
    
    if not filtered_messages:
        return {}
    
    # Check if local model requested (contains '/')
    if request.model and "/" in request.model and "openai" not in request.model:
        # Local execution using transformers
        results = analyze_prioritization_local(filtered_messages, request.model)
        return JSONResponse(content=results)

    # Fallback to Groq API (Original Logic)
    # prompt code maintained below...
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
        response_text = call_groq_ai(prompt, model_name=request.model)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        results = json.loads(response_text)
    except Exception:
        results = {m.id: "Normal" for m in filtered_messages}
        
    return JSONResponse(content=results)

@router.post("/moderate")
async def moderate_messages(request: AIAnalysisRequest):
    """Check moderation status.
    
    Note: Only processes messages that were created when AI was enabled.
    """
    # Check if AI is enabled
    if not chat_history.get_ai_enabled():
        raise HTTPException(status_code=403, detail="AI features are currently disabled. Please enable AI to use this feature.")
    
    if not request.messages:
        return {}
    
    # Filter to only include messages created when AI was enabled
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg['message_id'] for msg in ai_enabled_messages}
    filtered_messages = [m for m in request.messages if m.id in ai_enabled_ids]
    
    if not filtered_messages:
        return {}
    
    # Check if local model requested (contains '/')
    if request.model and "/" in request.model and "openai" not in request.model:
        # Local execution using transformers
        results = analyze_moderation_local(filtered_messages, request.model)
        return JSONResponse(content=results)

    # Fallback to Groq API (Original Logic)
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
        response_text = call_groq_ai(prompt, model_name=request.model)
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
    
    # Check if local model requested (contains '/')
    if request.model and "/" in request.model and "openai" not in request.model:
        # Local execution using transformers
        suggestions = generate_smart_replies_local(filtered_messages, request.model)
        return JSONResponse(content={"suggestions": suggestions})

    # Fallback to Groq API (Original Logic)
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
        response_text = call_groq_ai(prompt, model_name=request.model)
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
            total_messages=request.total_messages,
            model=request.model
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
async def classify_tasks(username: Optional[str] = None, model: Optional[str] = None) -> JSONResponse:
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

        result = extract_tasks_from_messages(messages, model=model)
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Error classifying tasks: {str(e)}"
        ) from e


@router.post("/translate")
async def translate_chat_messages(requests: List[TranslationRequest]) -> JSONResponse:
    """Translate chat messages into a target language.

    Request body: list of objects (each may include optional `model`).
    Note: Only processes messages that were created when AI was enabled.
    """
    # Check if AI is enabled
    if not chat_history.get_ai_enabled():
        raise HTTPException(
            status_code=403,
            detail="AI features are currently disabled. Please enable AI to use this feature.",
        )

    if not requests:
        return JSONResponse(content={"translations": {}})

    # Filter to only include messages created when AI was enabled
    ai_enabled_messages = chat_history.get_ai_enabled_messages()
    ai_enabled_ids = {msg["message_id"] for msg in ai_enabled_messages}
    filtered_requests = [r for r in requests if r.id in ai_enabled_ids]

    if not filtered_requests:
        return JSONResponse(content={"translations": {}})

    try:
        # Prefer per-item model if set; otherwise defaulting is handled in the service
        model = filtered_requests[0].model if filtered_requests else None
        payload = [r.dict() for r in filtered_requests]
        result = translate_messages_batch(payload, model=model)
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}") from e


@router.post("/translate-text")
async def translate_text_endpoint(request: TextTranslationRequest) -> JSONResponse:
    """Translate raw text into a target language."""
    try:
        result = translate_text(request.text, request.target_language, model=request.model)
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
        # Check if local model requested (contains '/')
        if request.model and "/" in request.model and "openai" not in request.model:
            # Local execution using transformers
            result = analyze_reminders_local(chat_history.get_ai_enabled_messages(), request.model)
            return JSONResponse(content=result)

        # Fallback to Groq API (Original Logic)
        result = generate_context_based_suggestions(
            username=request.username, context_window=request.context_window, model=request.model
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


UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload-audio")
async def upload_audio_file(file: UploadFile = File(...)) -> JSONResponse:
    try:
        # Generate unique filename to avoid collisions
        file_ext = os.path.splitext(file.filename)[1]
        if not file_ext:
            file_ext = ".webm" # Default for browser recording
            
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return JSONResponse(content={
            "url": f"/static/uploads/{unique_filename}",
            "filename": unique_filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e


ALLOWED_DOC_EXTENSIONS = {".pdf", ".docx"}


@router.post("/upload-document")
async def upload_document_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    uploaded_user_id: Optional[str] = Form(None),
) -> JSONResponse:
    """Accept a document file (pdf or docx), save to static/uploads, return immediately. Document-text-extraction runs in background."""
    ext = (os.path.splitext(file.filename or "")[1] or "").lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only .pdf and .docx are allowed. Got: {ext or 'no extension'}",
        )
    try:
        doc_upload_time = datetime.now(timezone.utc).isoformat()
        original_name = file.filename or ""
        unique_filename = f"{uuid.uuid4()}{ext}"
        file_path = UPLOAD_DIR / unique_filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Trigger document-text-extraction in background (so upload response returns immediately)
        background_tasks.add_task(
            _run_document_extraction_pipeline,
            file_path,
            unique_filename,
            uploaded_user_id,
            doc_upload_time,
            None,
            original_name,
        )

        return JSONResponse(content={
            "doc_id": unique_filename,
            "url": f"/static/uploads/{unique_filename}",
            "filename": unique_filename,
            "original_name": original_name or unique_filename,
            "uploaded_user_id": uploaded_user_id,
            "doc_upload_time": doc_upload_time,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e


def _run_document_extraction_pipeline(
    file_path: Path,
    doc_id: str,
    uploaded_user_id: Optional[str],
    doc_upload_time: Optional[str],
    model: Optional[str],
    original_name: Optional[str] = None,
) -> dict:
    """
    Run full document extraction: Docling -> LLM format -> summarize -> store in Qdrant.
    Returns dict with extraction_id, formatted_document_text, document_summary.
    """
    # De-duplication: if we already have extraction for this original_name, reuse it
    # and avoid recomputing vectors / document extraction.
    if original_name and uploaded_user_id:
        existing = get_document_extraction_by_original_name_and_user(
            original_name, uploaded_user_id
        )
        if existing:
            formatted_document_text = (existing.get("formatted_document_text") or "").strip()
            document_summary = (existing.get("document_summary") or "").strip() or "Summary generated."
            extraction_id = str(uuid.uuid4())
            return {
                "extraction_id": extraction_id,
                "formatted_document_text": formatted_document_text,
                "document_summary": document_summary,
            }

    clean_text, tables = extract_document_text_and_tables(file_path)

    format_prompt = f"""You are an expert document reconstruction assistant.

You are given:
- CLEAN_TEXT: linear text extracted from a document
- TABLES_JSON: structured table data (list of tables, with headers and rows)

Reconstruct a clean, well-formatted Markdown version of the document.

Requirements:
- Preserve sections and headings (use Markdown headings like #, ##, etc.).
- Preserve paragraph structure and lists (bullet/numbered).
- Recreate tables as Markdown tables using headers and rows from TABLES_JSON.
- Where the original document likely contained images or figures, infer and
  insert short placeholders like "[Image: description]" if mentioned in the text.
- Do NOT output JSON or explanations. Return ONLY the formatted document text.

CLEAN_TEXT:
\"\"\"{clean_text}\"\"\"

TABLES_JSON:
\"\"\"{json.dumps(tables, ensure_ascii=False)}\"\"\"
"""

    formatted_document_text = call_groq_ai(format_prompt, model_name=model)
    if not formatted_document_text or formatted_document_text.startswith("Error"):
        formatted_document_text = clean_text

    summary_result = generate_text_summary(formatted_document_text, model=model)
    document_summary = summary_result.get("summary", "Summary generated.")

    extraction_id = str(uuid.uuid4())

    try:
        store_document_extraction(
            doc_id=doc_id,
            document_summary=document_summary,
            uploaded_user_id=uploaded_user_id,
            doc_upload_time=doc_upload_time,
            formatted_document_text=formatted_document_text,
            original_name=original_name,
        )
    except Exception as e:
        print(f"Warning: Failed to store document extraction in Qdrant: {e}")

    return {
        "extraction_id": extraction_id,
        "formatted_document_text": formatted_document_text,
        "document_summary": document_summary,
    }


@router.post("/document-text-extraction")
async def document_text_extraction(
    request: DocumentTextExtractionRequest,
) -> JSONResponse:
    """
    Extract and format text from an uploaded document, then summarize it.

    Pipeline:
    1) Locate document in static/uploads using doc_id from /upload-document.
    2) Use Docling to extract clean text and structured tables.
    3) Use an LLM to produce a well-formatted document representation
       (Markdown) that integrates tables and preserves structure.
    4) Summarize the formatted document text and store in Qdrant.

    Note: /upload-document now triggers this pipeline automatically; use this
    endpoint when you need to re-run extraction for an already-uploaded document.
    """
    if not request.doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required.")

    file_path = UPLOAD_DIR / request.doc_id
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        result = _run_document_extraction_pipeline(
            file_path=file_path,
            doc_id=request.doc_id,
            uploaded_user_id=request.uploaded_user_id,
            doc_upload_time=request.doc_upload_time,
            model=request.model,
            original_name=request.original_name,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document extraction failed: {str(e)}",
        ) from e


@router.post("/document-qa")
async def document_qa(request: DocumentQARequest) -> JSONResponse:
    """
    Answer a question based on a specific document's extracted text in Qdrant.

    The document is uniquely identified by (doc_id, uploaded_user_id). Any user can
    ask questions about a document, but we always match both fields to ensure the
    correct document extraction is used.
    """
    if not request.doc_id or not request.uploaded_user_id:
        raise HTTPException(status_code=400, detail="doc_id and uploaded_user_id are required.")

    payload = get_document_extraction(
        doc_id=request.doc_id,
        uploaded_user_id=request.uploaded_user_id,
    )
    if not payload:
        raise HTTPException(status_code=404, detail="No extraction found for this document and user.")

    context_text = (
        (payload.get("formatted_document_text") or "").strip()
        or (payload.get("document_summary") or "").strip()
    )
    if not context_text:
        raise HTTPException(status_code=400, detail="Document extraction text is empty.")

    qa_prompt = f"""You are a helpful assistant answering questions based ONLY on the document below.

DOCUMENT:
\"\"\"{context_text}\"\"\"

Question: {request.question}

If the answer is clearly present, answer concisely.
If the answer is not present, say you cannot answer based on this document.
"""

    answer = call_groq_ai(qa_prompt, model_name=request.model)
    return JSONResponse(
        content={
            "doc_id": request.doc_id,
            "uploaded_user_id": request.uploaded_user_id,
            "question": request.question,
            "answer": answer,
        }
    )


@router.post("/transcribe-file")
async def transcribe_saved_file(request: AudioFileRequest) -> JSONResponse:
    try:
        file_path = UPLOAD_DIR / request.filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        # Transcribe
        with open(file_path, "rb") as audio_file:
            # Pass tuple (filename, file_object)
            transcription_text = transcribe_audio((request.filename, audio_file))
            
        return JSONResponse(content={"transcription": transcription_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}") from e


@router.post("/transcribe-meeting-file")
async def transcribe_meeting_file(request: MeetingAudioFileRequest) -> JSONResponse:
    """
    Transcribe an uploaded meeting recording and format it by speakers.

    Flow:
    - Accept filename of a previously uploaded audio file (same as /transcribe-file)
    - Use Groq Whisper STT to generate the raw transcription text
    - Post-process that text with an LLM to infer speaker turns and return
      a clean, speaker-separated transcript.
    """
    try:
        file_path = UPLOAD_DIR / request.filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # 1) Raw transcription: dispatch by selected ASR model
        asr_model = (request.asr_model or "whisper-large-v3").strip()
        if asr_model in ("facebook/seamless-m4t-medium", "facebook/hf-seamless-m4t-medium"):
            raw_transcription = await run_in_threadpool(transcribe_audio_seamless_m4t, str(file_path))
            if raw_transcription.startswith("Error"):
                with open(file_path, "rb") as audio_file:
                    raw_transcription = transcribe_audio((request.filename, audio_file))
        elif asr_model == "microsoft/VibeVoice-ASR":
            raw_transcription = await run_in_threadpool(transcribe_audio_vibevoice, str(file_path))
            if raw_transcription.startswith("Error"):
                with open(file_path, "rb") as audio_file:
                    raw_transcription = transcribe_audio((request.filename, audio_file))
            segments = []
        elif asr_model in ("openai/whisper-small"):
            # Local Whisper (Transformers): load once per model, then from cache
            raw_transcription, segments = await run_in_threadpool(
                transcribe_audio_whisper_local, str(file_path), asr_model
            )
        else:
            # Fallback: Groq Whisper
            with open(file_path, "rb") as audio_file:
                raw_transcription, segments = transcribe_audio_with_timestamps((request.filename, audio_file))

        if isinstance(raw_transcription, str) and raw_transcription.startswith("Error"):
            raise HTTPException(status_code=500, detail=raw_transcription)

        # 2) Meeting-style formatted transcription (speaker separated; with timestamps for whisper-large-v3)
        formatted_transcription = format_meeting_transcription(
            raw_transcription,
            model_name=request.model,
            segments=segments if segments else None,
        )

        if isinstance(formatted_transcription, str) and formatted_transcription.startswith("Error"):
            # Fall back to raw transcription if formatting fails
            return JSONResponse(
                content={
                    "transcription": raw_transcription,
                    "formatted_transcription": None,
                    "segments": segments if segments else None,
                    "notice": "Meeting formatting failed; returning raw transcription only.",
                }
            )

        response_data = {
            "transcription": raw_transcription,
            "formatted_transcription": formatted_transcription,
            "segments": segments if segments else None,
        }
        
        # Store transcription in Qdrant if requested and participant_ids provided
        if request.store_in_qdrant and request.participant_ids:
            try:
                storage_result = store_meeting_transcription(
                    transcription=raw_transcription,
                    participant_ids=request.participant_ids,
                    meeting_agenda=request.meeting_agenda,
                )
                response_data["meeting_id"] = storage_result["meeting_id"]
                response_data["stored_in_qdrant"] = True
                response_data["meeting_time"] = storage_result["meeting_time"]
            except Exception as e:
                # Don't fail the transcription if storage fails
                print(f"Warning: Failed to store transcription in Qdrant: {e}")
                response_data["stored_in_qdrant"] = False
                response_data["storage_error"] = str(e)
        else:
            response_data["stored_in_qdrant"] = False
        
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Meeting transcription failed: {str(e)}",
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
        result = generate_text_summary(request.text, model=request.model)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}") from e


@router.post("/meeting-recording/summary")
async def meeting_recording_summary(
    request: MeetingRecordingSummaryRequest,
) -> JSONResponse:
    """
    Post-process a full meeting transcription into a structured summary.

    This endpoint is designed to be called from the dedicated
    meeting transcription page once `/transcribe-file` has returned
    the raw transcript. It reuses the generic text summarization
    pipeline but is scoped specifically for meeting recordings.
    """
    if not request.transcription.strip():
        raise HTTPException(status_code=400, detail="Transcription text is required.")

    try:
        summary = generate_text_summary(
            request.transcription,
            model=request.model,
        )
        return JSONResponse(content=summary)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Meeting recording summary failed: {str(e)}",
        ) from e


@router.post("/meeting-tasks")
async def meeting_tasks(
    request: MeetingTasksRequest,
) -> JSONResponse:
    """
    Extract tasks from a meeting transcription using spaCy NER + rule-based detection.

    This endpoint uses spaCy to extract:
    - PERSON entities (assignees)
    - DATE/TIME entities (due dates)
    - Task trigger verbs (send, deliver, prepare, etc.)

    Returns tasks with structure:
    {
        "tasks": [
            {
                "task": "Send the proposal",
                "assignee": "John",
                "due_date": "Friday",
                "confidence": 0.82
            },
            ...
        ]
    }
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Transcription text is required.")

    try:
        # Use specified model or default to en_core_web_sm
        spacy_model = request.model or "en_core_web_sm"
        result = extract_meeting_tasks(request.text, model_name=spacy_model)
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Meeting task extraction failed: {str(e)}",
        ) from e


@router.post("/chat-search")
async def chat_search(
    request: ChatSearchRequest,
) -> JSONResponse:
    """
    Perform semantic search over conversation history.
    
    This endpoint searches through all chat messages using semantic similarity
    and keyword matching to find relevant conversations based on the user's query.
    
    Args:
        query: Search query text
        limit: Maximum number of results (default: 10)
        min_score: Minimum similarity score threshold 0.0-1.0 (default: 0.3)
        username: Optional username to filter messages by sender
    
    Returns:
        {
            "results": [
                {
                    "message_id": "...",
                    "sender": "...",
                    "message": "...",
                    "timestamp": "...",
                    "similarity_score": 0.85,
                    "keyword_score": 0.9,
                    "semantic_score": 0.7,
                    "relevance": "high"
                },
                ...
            ],
            "total_found": 5,
            "query": "..."
        }
    """
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
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Chat search failed: {str(e)}",
        ) from e


@router.post("/meeting_ask")
async def meeting_ask(
    request: MeetingAskRequest,
) -> JSONResponse:
    """
    Answer questions based on meeting transcriptions.
    
    This endpoint:
    1. Filters meetings where the participant_id is in the participant_ids list
    2. Searches those meetings for relevant information
    3. Generates an answer using LLM based on the relevant meeting transcriptions
    
    Args:
        query: User's question
        participant_id: User ID asking the question (filters meetings they participated in)
        limit: Maximum number of relevant meetings to consider (default: 3)
        score_threshold: Minimum similarity score for relevant meetings (default: 0.3)
        model: Optional LLM model for generating answer
    
    Returns:
        {
            "answer": "Generated answer based on meeting transcriptions",
            "relevant_meetings": [
                {
                    "meeting_id": "...",
                    "meeting_time": "...",
                    "meeting_agenda": "...",
                    "similarity_score": 0.85
                },
                ...
            ],
            "sources": [
                {
                    "meeting_id": "...",
                    "meeting_time": "...",
                    "meeting_agenda": "...",
                    "similarity_score": 0.85
                },
                ...
            ]
        }
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required.")
    
    if not request.participant_id or not request.participant_id.strip():
        raise HTTPException(status_code=400, detail="Participant ID is required.")
    
    try:
        result = ask_meeting_question(
            query=request.query,
            participant_id=request.participant_id,
            limit=request.limit or 3,
            score_threshold=request.score_threshold or 0.3,
            model=request.model,
        )
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Meeting Q&A failed: {str(e)}",
        ) from e


@router.post("/meeting/agenda-intelligence")
async def meeting_agenda_intelligence(
    request: AgendaIntelligenceRequest,
) -> JSONResponse:
    """
    Agenda vs Discussion Intelligence: compare planned agenda vs actual discussion.

    Allocates user-provided actual_meeting_minutes across agenda items and off-agenda
    topics, then returns overrun/underrun/missed insights.

    Request: transcript, agenda_items (list of {title, planned_minutes}), actual_meeting_minutes.
    Returns: agenda_items (with actual_minutes, status), off_agenda_topics, insights.
    """
    if not request.transcript or not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcription text is required.")
    if not request.agenda_items:
        raise HTTPException(status_code=400, detail="At least one agenda item is required.")
    if request.actual_meeting_minutes <= 0:
        raise HTTPException(status_code=400, detail="Actual meeting minutes must be greater than 0.")

    try:
        agenda_dicts = [{"title": item.title, "planned_minutes": item.planned_minutes} for item in request.agenda_items]
        result = analyze_agenda_vs_discussion(
            transcript=request.transcript,
            agenda_items=agenda_dicts,
            actual_meeting_minutes=request.actual_meeting_minutes,
            model=request.model,
        )
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Agenda intelligence failed: {str(e)}",
        ) from e


@router.post("/meeting/intelligence-insights")
async def meeting_intelligence_insights(
    request: MeetingIntelligenceRequest,
) -> JSONResponse:
    """
    Meeting Intelligence & Insights: higher-level analytics across people, topics, time, and outcomes.

    Produces: talk-time per participant, sentiment & tension detection, decision velocity,
    blockers/risks mentioned, dominant topics, and human-readable insight bullets.
    """
    if not request.transcript or not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcription text is required.")

    try:
        result = generate_meeting_intelligence(
            transcript=request.transcript,
            segments=request.segments,
            model=request.model,
        )
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Meeting intelligence failed: {str(e)}",
        ) from e
