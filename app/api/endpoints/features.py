from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List, Optional, Dict
from fastapi.responses import JSONResponse
import json
import shutil
import os
from pathlib import Path
import uuid
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
    analyze_reminders_local,
    generate_smart_replies_embedding,
    learn_new_reply,
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


class MeetingAudioFileRequest(BaseModel):
    """
    Request body for meeting-specific transcription directly from an uploaded file.

    This is similar to AudioFileRequest used by /transcribe-file but will
    additionally format the transcript into a speaker-separated meeting view.
    """

    filename: str
    model: Optional[str] = None  # LLM model for speaker formatting
    asr_model: Optional[str] = "whisper-large-v3"  # ASR: whisper-large-v3 | openai/whisper-large-v3 (local) | microsoft/VibeVoice-ASR | facebook/seamless-m4t-medium

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

@router.post("/sentiment")
async def analyze_sentiment(request: AIAnalysisRequest) -> JSONResponse:
    """Analyze sentiment of messages using VADER (Valence Aware Dictionary and sEntiment Reasoner).

    Returns per-message compound score in [-1.0, +1.0] and a label:
      - Positive  (compound >= 0.05)
      - Negative  (compound <= -0.05)
      - Neutral   (otherwise)

    VADER is rule-based and requires no GPU or model download.
    """
    if not chat_history.get_ai_enabled():
        raise HTTPException(
            status_code=403,
            detail="AI features are currently disabled. Please enable AI to use this feature.",
        )

    if not request.messages:
        return JSONResponse(content={})

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="vaderSentiment is not installed. Run: pip install vaderSentiment",
        )

    import time, json as _json
    analyzer = SentimentIntensityAnalyzer()
    results: dict = {}
    start_time = time.perf_counter()

    for msg in request.messages:
        text = msg.message or ""
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        results[msg.id] = {
            "label": label.capitalize(),
            "compound": round(compound, 4),
            "pos": round(scores["pos"], 4),
            "neu": round(scores["neu"], 4),
            "neg": round(scores["neg"], 4),
        }

        # --- Terminal Log ---
        log_entry = {
            "text": text,
            "sentiment": {
                "label": label,
                "compound": round(compound, 4),
                "positive": round(scores["pos"], 4),
                "neutral": round(scores["neu"], 4),
                "negative": round(scores["neg"], 4),
            }
        }
        print("\n--- VADER Sentiment ---")
        print(_json.dumps(log_entry, indent=2))
        print("-----------------------")

    elapsed = time.perf_counter() - start_time
    print(f"\n✅ Sentiment analysis complete: {len(results)} message(s) in {elapsed:.4f}s\n")

    return JSONResponse(content=results)


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
    
    import time
    start_time = time.time()
    should_learn = False
    original_model = request.model 

    # --- 1. Embedding / Vector Search (New Feature) ---
    if request.model and "sentence-transformers" in request.model:
        # Try retrieving from bank
        vector_result = generate_smart_replies_embedding(filtered_messages, request.model)
        
        if vector_result.get("source") in ["vector-bank", "faiss-cache"]:
            # Found in cache! Return immediately.
            duration = round(time.time() - start_time, 2)
            vector_result["execution_time"] = duration
            # Pass through the internal model time if available
            if "response_time" in vector_result:
                 vector_result["model_time"] = vector_result["response_time"]
            return JSONResponse(content=vector_result)
            
        # Missed in cache (Low confidence). Fallback to Generative AI.
        # We flag this to "Learn" the new result later.
        should_learn = True
        
        # Force fallback to Groq (or default cloud) by clearing the model ID 
        # so it skips the "Local execution" block below which expects a generative model.
        request.model = None 

    # --- 2. Local Generative Execution ---
    # Check if local model requested (contains '/') AND it's not the embedding model we just handled
    if request.model and "/" in request.model and "openai" not in request.model:
        # Local execution using transformers
        suggestions = generate_smart_replies_local(filtered_messages, request.model)
        duration = round(time.time() - start_time, 2)
        return JSONResponse(content={"suggestions": suggestions, "execution_time": duration})

    # --- 3. Cloud/Groq Fallback (Original Logic) ---
    # --- 3. Cloud/Groq Fallback (Original Logic) ---
    # Smart "Context-Aware" Prompt
    recent_msgs = filtered_messages[-3:] # Get last 3 messages for context
    conversation_context = "\n".join([f"- {m.message}" for m in recent_msgs])
    
    prompt = f"""
    You are a helpful AI assistant drafting quick replies for a chat user.
    
    Conversation Context (Last few messages):
    {conversation_context}
    
    Task: Generate 3 short, natural reply suggestions for the LAST message above.
    
    Tone requirement: {tone_instruction}
    
    Strict Rules:
    - Do NOT paraphrase the last message.
    - Do NOT continue the conversation as if you are the same speaker.
    - You are replying TO the last message.
    - Keep it under 15 words.
    
    Return a JSON object: {{ "suggestions": ["Reply 1", "Reply 2", "Reply 3"] }}
    Return ONLY valid JSON.
    """
    
    try:
        # Use default Groq model (llama3-70b usually) if request.model is None
        response_text = call_groq_ai(prompt, model_name=request.model)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            
        result = json.loads(response_text)
        
        # --- 4. Dynamic Learning (Post-Generation) ---
        if should_learn and result.get("suggestions"):
            # We generated new answers. Store them in the bank for next time.
            embedding_model_id = original_model
            for reply in result["suggestions"]:
                # Learn the pair: (User Query) -> (Bot Reply)
                # Note: We learn from the LAST message the user sent.
                learn_new_reply(last_msg.message, reply, embedding_model_id)
            
            # Mark source as "generated-and-learned" for UI clarity if needed
            result["source"] = "generative-llm (learned)"

        duration = round(time.time() - start_time, 2)
        result["execution_time"] = duration
        
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
            segments = []
        elif asr_model == "microsoft/VibeVoice-ASR":
            raw_transcription = await run_in_threadpool(transcribe_audio_vibevoice, str(file_path))
            if raw_transcription.startswith("Error"):
                with open(file_path, "rb") as audio_file:
                    raw_transcription = transcribe_audio((request.filename, audio_file))
            segments = []
        elif asr_model in ("whisper-large-v3", "openai/whisper-small"):
            # Local Whisper (Transformers): load once per model, then from cache
            raw_transcription, segments = await run_in_threadpool(
                transcribe_audio_whisper_local, str(file_path), "en"
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

        return JSONResponse(
            content={
                "transcription": raw_transcription,
                "formatted_transcription": formatted_transcription,
                "segments": segments if segments else None,
            }
        )
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
