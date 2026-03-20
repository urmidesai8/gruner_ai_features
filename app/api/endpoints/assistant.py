"""
AI Assistant endpoint: Strands-based agent with summarize / draft-response / translate tools.
Includes session/memory management (Redis with in-memory fallback) and production-ready setup.
Supports text and live-audio input (transcribe then run agent), plus feedback logging.
"""
import asyncio
import base64
import io
import json
import logging
import re
import threading
import time
import uuid
from typing import Any, List, Optional

import httpx
import psycopg2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from app.core.config import settings
from app.services.ai_service import transcribe_audio
from app.services.tts_service import text_to_speech, text_to_speech_svara

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Session / memory store (Redis preferred; in-memory fallback when Redis unavailable)
# ---------------------------------------------------------------------------

# --- Redis disabled (not running; saves ~5s startup timeout) ---
# try:
#     import redis
#     _redis = redis.Redis(
#         host=settings.REDIS_HOST,
#         port=settings.REDIS_PORT,
#         password=settings.REDIS_PASSWORD or None,
#         db=settings.REDIS_DB,
#         decode_responses=settings.REDIS_DECODE_RESPONSES,
#         socket_connect_timeout=5,
#         socket_timeout=5,
#     )
#     _redis.ping()
#     _REDIS_AVAILABLE = True
# except Exception as e:
#     logger.warning("Redis unavailable for assistant sessions: %s. Using in-memory fallback.", e)
#     _redis = None
#     _REDIS_AVAILABLE = False
_redis = None
_REDIS_AVAILABLE = False

_ASSISTANT_KEY_PREFIX = "assistant:session:"
_IN_MEMORY_SESSIONS: dict[str, List[dict[str, str]]] = {}
_IN_MEMORY_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Postgres feedback storage (simple connection per request)
# ---------------------------------------------------------------------------


def _get_feedback_connection():
    if not settings.POSTGRES_DB or not settings.POSTGRES_USER:
        raise RuntimeError("Postgres settings for assistant feedback are not configured.")
    return psycopg2.connect(
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
    )


def save_assistant_feedback(feedback_message: str, feedback: str) -> None:
    """
    Persist assistant feedback to Postgres.
    On failure, logs and returns without raising to the client.
    """
    feedback = (feedback or "").strip().lower()
    if feedback not in {"positive", "negative"}:
        logger.warning("Ignoring invalid feedback value: %s", feedback)
        return

    try:
        conn = _get_feedback_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback_assistant (feedback_id, feedback_message, feedback, created_at)
            VALUES (%s, %s, %s, NOW())
            """,
            (str(uuid.uuid4()), feedback_message, feedback),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.exception("Failed to save assistant feedback: %s", e)


def _session_key(session_id: str) -> str:
    return f"{_ASSISTANT_KEY_PREFIX}{session_id}"


def get_session_messages(session_id: str) -> List[dict[str, str]]:
    """Load message list for a session (role + content)."""
    if _REDIS_AVAILABLE:
        try:
            raw = _redis.get(_session_key(session_id))
            if raw:
                return json.loads(raw)
            return []
        except Exception as e:
            logger.warning("Redis get failed for session %s: %s", session_id, e)
            return []
    with _IN_MEMORY_LOCK:
        return list(_IN_MEMORY_SESSIONS.get(session_id, []))


def append_session_messages(
    session_id: str,
    new_messages: List[dict[str, str]],
) -> None:
    """Append messages to session and trim to max history. Set TTL when using Redis."""
    if _REDIS_AVAILABLE:
        try:
            key = _session_key(session_id)
            existing = get_session_messages(session_id)
            combined = (existing + new_messages)[-settings.ASSISTANT_MAX_HISTORY_MESSAGES:]
            _redis.setex(
                key,
                settings.ASSISTANT_SESSION_TTL_SECONDS,
                json.dumps(combined),
            )
            return
        except Exception as e:
            logger.warning("Redis set failed for session %s: %s", session_id, e)
    with _IN_MEMORY_LOCK:
        lst = _IN_MEMORY_SESSIONS.setdefault(session_id, [])
        lst.extend(new_messages)
        _IN_MEMORY_SESSIONS[session_id] = lst[-settings.ASSISTANT_MAX_HISTORY_MESSAGES:]


def build_context_from_history(messages: List[dict[str, str]], current_message: str) -> str:
    """Build a single prompt that includes recent conversation context for the agent."""
    if not messages:
        return current_message
    lines = ["Previous conversation:"]
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role.capitalize()}: {content}")
    lines.append("")
    lines.append(f"Current user message: {current_message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools (call existing feature endpoints)
# ---------------------------------------------------------------------------


@tool
def summarize_text_tool(text: str, model: Optional[str] = None) -> dict:
    """
    Summarize arbitrary text by calling the backend /summarize-text endpoint.

    Args:
        text: The text content to summarize.
        model: Optional model identifier to use for summarization.
    """
    try:
        response = httpx.post(
            f"{settings.API_BASE_URL}/api/features/summarize-text",
            json={"text": text, "model": model},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to reach summarize-text endpoint: {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Summarize-text endpoint returned error: {e.response.text}"
        ) from e


@tool
def draft_response_tool(message: str, tone: str = "auto") -> dict:
    """
    Draft a response by calling the backend /draft-response endpoint.

    Args:
        message: The original user message to improve.
        tone: Desired tone (professional, casual, friendly, formal, auto).
    """
    try:
        response = httpx.post(
            f"{settings.API_BASE_URL}/api/features/draft-response",
            json={"message": message, "tone": tone},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to reach draft-response endpoint: {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Draft-response endpoint returned error: {e.response.text}"
        ) from e


@tool
def translate_text_tool(
    text: str,
    target_language: str,
) -> dict:
    """
    Translate text by calling the backend /translate-text endpoint.

    Args:
        text: The text to translate.
        target_language: Target language code (e.g. 'en', 'de', 'fr').
        model: Optional model identifier to use for translation.
    """
    try:
        response = httpx.post(
            f"{settings.API_BASE_URL}/api/features/translate-text",
            json={
                "text": text,
                "target_language": target_language,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to reach translate-text endpoint: {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Translate-text endpoint returned error: {e.response.text}"
        ) from e


# ---------------------------------------------------------------------------
# Agent (Groq via OpenAI-compatible API)
# ---------------------------------------------------------------------------

groq_model = OpenAIModel(
    client_args={
        "api_key": settings.GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
    },
    model_id=settings.AI_MODEL,
    params={
        "temperature": 0.0,
        "max_tokens": 1000,
    },
)

assistant_agent = Agent(
    model=groq_model,
    tools=[summarize_text_tool, draft_response_tool, translate_text_tool],
    system_prompt=(
        "You are a helpful, friendly AI assistant for chat messages.\n"
        "\n"
        "You have three specialized tools that call backend APIs:\n"
        "- summarize_text_tool: summarize a block of text.\n"
        "- draft_response_tool: improve grammar and rewrite a single message in a chosen tone.\n"
        "- translate_text_tool: translate text into a target language.\n"
        "\n"
        "Behavior rules:\n"
        "1) Use summarize_text_tool when the user asks you to summarize text.\n"
        "2) Use draft_response_tool when the user asks you to improve, rewrite, fix grammar, or change the tone of a message.\n"
        "3) Use translate_text_tool when the user asks you to translate text into a target language.\n"
        "4) For general questions, greetings, conversations, or any other topics, respond helpfully and naturally WITHOUT calling any tools. You are allowed to answer any question to the best of your ability.\n"
        "5) For each user request, choose at most ONE tool when it truly matches the user's intent. If none of the tools are appropriate, answer directly.\n"
        "6) Do NOT describe the tools, their names, or that you are calling tools. The user should only see the final answer.\n"
        "7) For summarization requests, return a clear summary (1–2 short paragraphs or 3–6 bullet points).\n"
        "8) For drafting/rewriting requests, return ONLY the improved or tone-adjusted message.\n"
        "9) For translation requests, return ONLY the translated text.\n"
        "10) If a tool fails or is unavailable, apologize briefly and answer using your own reasoning.\n"
        "11) When 'Previous conversation' context is provided, use it only for coherence; still respond to the 'Current user message' as the main request.\n"
    ),
)

# Dedicated thread pool for blocking agent calls (production-ready)
_agent_executor: Optional[Any] = None
_executor_lock = threading.Lock()


def _get_executor():
    global _agent_executor
    with _executor_lock:
        if _agent_executor is None:
            import concurrent.futures
            _agent_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=settings.ASSISTANT_EXECUTOR_WORKERS,
                thread_name_prefix="assistant_agent",
            )
        return _agent_executor


# ---------------------------------------------------------------------------
# Request / response schemas and endpoint
# ---------------------------------------------------------------------------


class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=settings.ASSISTANT_MAX_MESSAGE_LENGTH)
    session_id: Optional[str] = Field(None, description="Optional session id for conversation continuity.")


class AssistantResponse(BaseModel):
    reply: str
    session_id: str


class AssistantFeedbackRequest(BaseModel):
    message: str = Field(..., min_length=1)
    feedback: str = Field(..., description="One of: positive, negative")


class AssistantSpeakRequest(BaseModel):
    """Request body for TTS: speak the given text (e.g. after GPT-Edge reply)."""
    text: str = Field(..., min_length=1, max_length=settings.ASSISTANT_MAX_MESSAGE_LENGTH)
    voice: Optional[str] = Field(None, description="Optional Edge TTS voice id")


@router.post("/assistant/speak")
async def assistant_speak(request: AssistantSpeakRequest) -> JSONResponse:
    """
    Convert text to speech using edge-tts. Used by GPT-Edge to read aloud the assistant reply.
    Returns JSON with audio_base64 (MP3) and audio_content_type.
    """
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    try:
        audio_bytes = await text_to_speech(text, voice=request.voice)
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="TTS produced no audio.")
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        return JSONResponse(
            content={
                "audio_base64": audio_base64,
                "audio_content_type": "audio/mpeg",
            }
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("TTS failed: %s", e)
        raise HTTPException(status_code=500, detail="Text-to-speech failed.") from e


@router.get("/config")
async def get_assistant_config() -> JSONResponse:
    """Return client-safe config (e.g. Nova voice backend URL) for the chat assistant UI."""
    return JSONResponse(content={
        "nova_voice_url": settings.NOVA_VOICE_URL or "",
    })


@router.post("/assistant/stream", response_class=StreamingResponse)
async def stream_assistant(request: AssistantRequest) -> StreamingResponse:
    """
    Stream the Strands-based AI assistant with the same session memory as POST /assistant.

    - If `session_id` is provided, previous turns are used as context.
    - If omitted, a new session is created; session_id is sent in the first and final SSE events.
    - After the stream completes, user message and full reply are persisted to the session store.
    """
    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")
    if len(message) > settings.ASSISTANT_MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message exceeds maximum length of {settings.ASSISTANT_MAX_MESSAGE_LENGTH} characters.",
        )

    session_id = (request.session_id or "").strip() or str(uuid.uuid4())
    history = get_session_messages(session_id)
    prompt = build_context_from_history(history, message)

    async def event_generator():
        full_reply_parts: List[str] = []
        try:
            # Send session_id immediately so the client can store it for follow-ups
            yield f"event: status\ndata: {json.dumps({'status': 'agent_started', 'session_id': session_id})}\n\n"

            async for chunk in assistant_agent.stream_async(prompt):
                if "data" in chunk and "delta" in chunk and isinstance(chunk["data"], str):
                    token = chunk["data"]
                    full_reply_parts.append(token)
                    yield f"data: {json.dumps({'content': token})}\n\n"
                elif "result" in chunk:
                    pass

                await asyncio.sleep(0.005)

            full_reply = "".join(full_reply_parts).strip()
            append_session_messages(
                session_id,
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": full_reply or "(no output)"},
                ],
            )
            yield f"event: end\ndata: {json.dumps({'status': 'finished', 'session_id': session_id})}\n\n"

        except Exception as e:
            logger.exception("Assistant stream failed")
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/assistant/feedback")
async def assistant_feedback(request: AssistantFeedbackRequest) -> JSONResponse:
    """
    Capture explicit user feedback (thumbs-up / thumbs-down) on an assistant reply.

    - feedback: 'positive' or 'negative'
    - message: the assistant's reply text the user reacted to
    """
    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    feedback = (request.feedback or "").strip().lower()
    if feedback not in {"positive", "negative"}:
        raise HTTPException(
            status_code=400,
            detail="feedback must be 'positive' or 'negative'.",
        )

    loop = asyncio.get_event_loop()
    # Run DB insert in a thread to avoid blocking the event loop
    await loop.run_in_executor(None, lambda: save_assistant_feedback(message, feedback))

    return JSONResponse(content={"status": "ok"})


@router.post("/assistant/audio")
async def assistant_audio(
    file: UploadFile = File(..., description="Audio recording (e.g. webm, wav, mp3)"),
    session_id: Optional[str] = Form(None),
) -> JSONResponse:
    """
    Accept live audio from the user: transcribe with Groq Whisper, then run the assistant
    with the same session memory as text endpoints. Returns reply, session_id, and transcription.
    """
    if not file.filename and not getattr(file, "content_type", ""):
        raise HTTPException(status_code=400, detail="Audio file is required.")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {e}") from e

    if not content or len(content) < 100:
        raise HTTPException(status_code=400, detail="Audio data too short or empty.")

    filename = file.filename or "audio.webm"
    buf = io.BytesIO(content)
    buf.name = filename

    loop = asyncio.get_event_loop()
    try:
        transcription = await loop.run_in_executor(
            None,
            lambda: transcribe_audio((filename, buf)),
        )
    except Exception as e:
        logger.exception("Transcription failed for assistant audio")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e

    if not transcription or (isinstance(transcription, str) and transcription.strip().lower().startswith("error")):
        raise HTTPException(
            status_code=400,
            detail=transcription or "Transcription returned no text.",
        )

    message = transcription.strip()
    if len(message) > settings.ASSISTANT_MAX_MESSAGE_LENGTH:
        message = message[: settings.ASSISTANT_MAX_MESSAGE_LENGTH]

    sid = (session_id or "").strip() or str(uuid.uuid4())
    history = get_session_messages(sid)
    prompt = build_context_from_history(history, message)

    executor = _get_executor()
    try:
        reply = await loop.run_in_executor(
            executor,
            lambda: assistant_agent(prompt),
        )
        reply_text = str(reply).strip()
    except Exception as e:
        logger.exception("Assistant agent run failed (audio)")
        raise HTTPException(status_code=500, detail=f"Assistant failed: {e}") from e

    append_session_messages(
        sid,
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply_text or "(no output)"},
        ],
    )

    return JSONResponse(
        content={
            "reply": reply_text,
            "session_id": sid,
            "transcription": message,
        },
    )


# ---------------------------------------------------------------------------
# Edge TTS (Voice-to-Voice): speak → transcribe → Strands agent → TTS → play reply
# ---------------------------------------------------------------------------

@router.post("/assistant/v2v")
async def assistant_v2v(
    file: UploadFile = File(..., description="Audio recording (e.g. webm, wav, mp3)"),
    session_id: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
) -> JSONResponse:
    """
    Voice-to-voice: user speaks → transcribe (Groq Whisper) → Strands agent → TTS (edge-tts) → return reply + audio.
    Same session memory as text/audio endpoints. Returns reply text, transcription, session_id, and audio_base64 (MP3).
    """
    if not file.filename and not getattr(file, "content_type", ""):
        raise HTTPException(status_code=400, detail="Audio file is required.")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {e}") from e

    if not content or len(content) < 100:
        raise HTTPException(status_code=400, detail="Audio data too short or empty.")

    filename = file.filename or "audio.webm"
    buf = io.BytesIO(content)
    buf.name = filename

    loop = asyncio.get_event_loop()
    try:
        transcription = await loop.run_in_executor(
            None,
            lambda: transcribe_audio((filename, buf)),
        )
    except Exception as e:
        logger.exception("Edge TTS transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e

    if not transcription or (isinstance(transcription, str) and transcription.strip().lower().startswith("error")):
        raise HTTPException(
            status_code=400,
            detail=transcription or "Transcription returned no text.",
        )

    message = transcription.strip()
    if len(message) > settings.ASSISTANT_MAX_MESSAGE_LENGTH:
        message = message[: settings.ASSISTANT_MAX_MESSAGE_LENGTH]

    sid = (session_id or "").strip() or str(uuid.uuid4())
    history = get_session_messages(sid)
    prompt = build_context_from_history(history, message)
    # Edge TTS Language Mapping based on voice ID prefix
    _EDGE_LANG_MAP = {
        "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "az": "Azerbaijani", 
        "bg": "Bulgarian", "bn": "Bengali", "bs": "Bosnian", "ca": "Catalan", 
        "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek", 
        "en": "English", "es": "Spanish", "et": "Estonian", "fa": "Persian", 
        "fi": "Finnish", "fil": "Filipino", "fr": "French", "ga": "Irish", 
        "gl": "Galician", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", 
        "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian", "is": "Icelandic", 
        "it": "Italian", "ja": "Japanese", "jv": "Javanese", "ka": "Georgian", 
        "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean", "lo": "Lao", 
        "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam", 
        "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese", 
        "my": "Burmese", "nb": "Norwegian Bokmål", "ne": "Nepali", "nl": "Dutch", 
        "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", 
        "ru": "Russian", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", 
        "so": "Somali", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", 
        "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu", 
        "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", 
        "uz": "Uzbek", "vi": "Vietnamese", "zu": "Zulu", "zh": "Chinese"
    }
    
    if voice:
        lang_code = voice.split("-")[0] if "-" in voice else voice
        lang_name = _EDGE_LANG_MAP.get(lang_code, "")
        if lang_name and lang_code != "en":
            prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT: The user has selected {lang_name} as their preferred language. "
                f"You MUST respond entirely in {lang_name} language using {lang_name} script. "
                f"Do NOT respond in English."
            )

    executor = _get_executor()
    try:
        reply = await loop.run_in_executor(
            executor,
            lambda: assistant_agent(prompt),
        )
        reply_text = str(reply).strip()
    except Exception as e:
        logger.exception("Edge TTS assistant agent failed")
        raise HTTPException(status_code=500, detail=f"Assistant failed: {e}") from e

    append_session_messages(
        sid,
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply_text or "(no output)"},
        ],
    )

    # TTS: convert reply to speech (MP3)
    audio_base64 = ""
    if reply_text:
        try:
            audio_bytes = await text_to_speech(reply_text, voice=voice)
            if audio_bytes:
                audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        except Exception as e:
            logger.warning("Edge TTS TTS failed (reply still returned as text): %s", e)

    return JSONResponse(
        content={
            "reply": reply_text,
            "session_id": sid,
            "transcription": message,
            "audio_base64": audio_base64,
            "audio_content_type": "audio/mpeg",
        },
    )


@router.post("/assistant/v2v-svara")
async def assistant_v2v_svara(
    file: UploadFile = File(..., description="Audio recording (e.g. webm, wav, mp3)"),
    session_id: Optional[str] = Form(None),
    voice_id: Optional[str] = Form(None),
    emotion: Optional[str] = Form("neutral"),
) -> JSONResponse:
    """
    Voice-to-voice: user speaks → transcribe (Groq Whisper) → Strands agent → TTS (svara-tts)
    → return reply + audio.

    Same session memory as text/audio endpoints. Returns reply text, transcription,
    session_id, and audio_base64 (WAV bytes base64-encoded).
    """
    t_request_start = time.perf_counter()
    logger.info(
        "[v2v-svara] Request received | voice_id=%s session_id=%s",
        voice_id, session_id,
    )

    if not file.filename and not getattr(file, "content_type", ""):
        raise HTTPException(status_code=400, detail="Audio file is required.")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {e}") from e

    if not content or len(content) < 100:
        raise HTTPException(status_code=400, detail="Audio data too short or empty.")

    logger.info(
        "[v2v-svara] Audio received | filename=%s size_bytes=%d",
        file.filename or "audio.webm", len(content),
    )

    filename = file.filename or "audio.webm"
    buf = io.BytesIO(content)
    buf.name = filename

    # Derive BCP-47 language code from voice_id (e.g. "mr_female" → "mr").
    # Fall back to None so Whisper auto-detects when voice_id is unknown/unsupplied.
    _SVARA_LANG_CODES = {
        "en", "hi", "bn", "ta", "te", "mr", "gu", "kn",
        "ml", "pa", "or", "as", "ne", "sa", "ur", "sd",
    }
    stt_language: str | None = None
    if voice_id:
        lang_prefix = voice_id.rsplit("_", 1)[0] if "_" in voice_id else voice_id
        stt_language = lang_prefix if lang_prefix in _SVARA_LANG_CODES else None

    logger.info(
        "[v2v-svara] STT starting | model=whisper-large-v3 language=%s",
        stt_language or "auto-detect",
    )
    t_stt_start = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        transcription = await loop.run_in_executor(
            None,
            lambda: transcribe_audio((filename, buf), language=stt_language),
        )
    except Exception as e:
        logger.exception("[v2v-svara] STT failed after %.2fs", time.perf_counter() - t_stt_start)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e
    t_stt_elapsed = time.perf_counter() - t_stt_start

    if not transcription or (isinstance(transcription, str) and transcription.strip().lower().startswith("error")):
        raise HTTPException(
            status_code=400,
            detail=transcription or "Transcription returned no text.",
        )

    message = transcription.strip()
    if len(message) > settings.ASSISTANT_MAX_MESSAGE_LENGTH:
        message = message[: settings.ASSISTANT_MAX_MESSAGE_LENGTH]

    logger.info(
        "[v2v-svara] STT complete | elapsed=%.2fs chars=%d text=%r",
        t_stt_elapsed, len(message), message[:80],
    )

    sid = (session_id or "").strip() or str(uuid.uuid4())
    history = get_session_messages(sid)
    prompt = build_context_from_history(history, message)

    # Map voice_id prefix to language name so the LLM responds in the correct language
    _VOICE_LANG_MAP = {
        "en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
        "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
        "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
        "ne": "Nepali", "sa": "Sanskrit", "mai": "Maithili", "bho": "Bhojpuri",
        "ur": "Urdu", "sd": "Sindhi", "mni": "Manipuri",
    }
    if voice_id:
        lang_code = voice_id.rsplit("_", 1)[0] if "_" in voice_id else voice_id
        lang_name = _VOICE_LANG_MAP.get(lang_code, "")
        if lang_name and lang_code != "en":
            prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT: The user has selected {lang_name} as their preferred language. "
                f"You MUST respond entirely in {lang_name} language using {lang_name} script. "
                f"Do NOT respond in English."
            )

    if emotion == "auto":
        prompt = (
            f"{prompt}\n\n"
            'IMPORTANT: The user has set the emotion to "Auto". You MUST determine the appropriate '
            'emotion for your reply based on the conversation context. '
            'Your final response MUST be a valid JSON object with EXACTLY two keys: "reply" (your literal response text) '
            'and "emotion" (one of: "neutral", "happy", "sad", "angry", "excited", "surprised"). '
            'Do NOT include any other text or markdown outside of the JSON block.'
        )

    logger.info("[v2v-svara] LLM starting | session_id=%s emotion_mode=%s", sid, emotion)
    t_llm_start = time.perf_counter()
    executor = _get_executor()
    try:
        reply = await loop.run_in_executor(
            executor,
            lambda: assistant_agent(prompt),
        )
        reply_text = str(reply).strip()
    except Exception as e:
        logger.exception("[v2v-svara] LLM failed after %.2fs", time.perf_counter() - t_llm_start)
        raise HTTPException(status_code=500, detail=f"Assistant failed: {e}") from e
    t_llm_elapsed = time.perf_counter() - t_llm_start
    
    detected_emotion = emotion
    if emotion == "auto":
        try:
            match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", reply_text, re.DOTALL)
            json_str = match.group(1) if match else reply_text
            data = json.loads(json_str)
            reply_text = data.get("reply", reply_text)
            detected_emotion = data.get("emotion", "neutral").lower()
            valid_emotions = {"neutral", "happy", "sad", "angry", "excited", "surprised"}
            if detected_emotion not in valid_emotions:
                detected_emotion = "neutral"
        except Exception as e:
            logger.warning("[v2v-svara] Could not parse JSON for auto-emotion. Fallback to neutral. Error: %s", e)
            detected_emotion = "neutral"

    logger.info(
        "[v2v-svara] LLM complete | elapsed=%.2fs detected_emotion=%s reply_chars=%d reply=%r",
        t_llm_elapsed, detected_emotion, len(reply_text), reply_text[:80],
    )

    append_session_messages(
        sid,
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply_text or "(no output)"},
        ],
    )

    audio_base64 = ""
    audio_content_type = "audio/mpeg"
    audio_error: Optional[str] = None
    t_tts_elapsed = 0.0
    if reply_text:
        logger.info(
            "[v2v-svara] TTS starting | voice_id=%s reply_chars=%d",
            voice_id, len(reply_text),
        )
        t_tts_start = time.perf_counter()
        try:
            audio_bytes, audio_content_type = await text_to_speech_svara(reply_text, voice_id=voice_id, emotion=detected_emotion)
            t_tts_elapsed = time.perf_counter() - t_tts_start
            if audio_bytes:
                audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
                logger.info(
                    "[v2v-svara] TTS complete | elapsed=%.2fs audio_bytes=%d",
                    t_tts_elapsed, len(audio_bytes),
                )
            else:
                logger.warning("[v2v-svara] TTS returned empty audio after %.2fs", t_tts_elapsed)
        except Exception as e:
            t_tts_elapsed = time.perf_counter() - t_tts_start
            logger.warning(
                "[v2v-svara] TTS failed after %.2fs (reply still returned as text): %s",
                t_tts_elapsed, e,
            )
            audio_error = str(e)

    t_total_elapsed = time.perf_counter() - t_request_start
    logger.info(
        "[v2v-svara] Request complete | total=%.2fs stt=%.2fs llm=%.2fs tts=%.2fs"
        " | voice_id=%s session_id=%s",
        t_total_elapsed, t_stt_elapsed, t_llm_elapsed, t_tts_elapsed,
        voice_id, sid,
    )

    return JSONResponse(
        content={
            "reply": reply_text,
            "session_id": sid,
            "transcription": message,
            "audio_base64": audio_base64,
            "audio_content_type": audio_content_type,
            "audio_error": audio_error or "",
        },
    )
