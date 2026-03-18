"""
Nova 2 Sonic Socket.IO handlers — merged into main app.
Uses app.backend for Bedrock S2S session and tools.
"""

import base64
import json
import logging
from typing import Dict

import socketio

from app.backend.bedrock_session import BedrockS2SSession
from app.backend.tools import ToolRegistry

logger = logging.getLogger("nova")

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    max_http_buffer_size=10 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25,
)

sessions: Dict[str, BedrockS2SSession] = {}
tool_registry = ToolRegistry()
_audio_window_started: Dict[str, float] = {}
_audio_window_bytes: Dict[str, int] = {}
_MAX_AUDIO_BYTES_PER_10S = 2_000_000


@sio.event
async def connect(sid, environ):
    logger.info("Nova client connected: %s", sid)
    await sio.emit("connected", {"sid": sid}, to=sid)


@sio.event
async def disconnect(sid):
    logger.info("Nova client disconnected: %s", sid)
    session = sessions.pop(sid, None)
    if session:
        await session.close()


@sio.event
async def start_session(sid, data: dict):
    """
    Start a new Bedrock Nova Sonic streaming session.
    data = { region, voice, systemPrompt, temperature, topP, maxTokens, tools, sampleRate }
    """
    logger.info("[%s] start_session: %s", sid, json.dumps({k: v for k, v in (data or {}).items() if k != "systemPrompt"}))

    old = sessions.pop(sid, None)
    if old:
        await old.close()

    region = (data or {}).get("region", "us-east-1")
    voice = (data or {}).get("voice", "tiffany")
    system_prompt = (data or {}).get("systemPrompt", "You are a helpful voice assistant in the chat.")
    temperature = float((data or {}).get("temperature", 0.7))
    top_p = float((data or {}).get("topP", 0.9))
    max_tokens = int((data or {}).get("maxTokens", 1024))
    sample_rate = int((data or {}).get("sampleRate", 24000))
    enabled_tools = (data or {}).get("tools", [])

    tool_specs = tool_registry.get_specs(enabled_tools)

    session = BedrockS2SSession(
        sid=sid,
        sio=sio,
        region=region,
        voice=voice,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        sample_rate=sample_rate,
        tool_specs=tool_specs,
        tool_registry=tool_registry,
    )
    sessions[sid] = session

    try:
        await session.start()
        await sio.emit("session_started", {"success": True}, to=sid)
        logger.info("[%s] Nova session started successfully", sid)
    except Exception as e:
        logger.exception("[%s] Failed to start Nova session: %s", sid, e)
        sessions.pop(sid, None)
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def audio_input(sid, data):
    """Receive raw PCM audio chunk from browser (base64 encoded)."""
    session = sessions.get(sid)
    if not session:
        return
    try:
        if isinstance(data, (bytes, bytearray)):
            audio_bytes = bytes(data)
        else:
            audio_bytes = base64.b64decode(data)
        now = asyncio.get_event_loop().time()
        started = _audio_window_started.get(sid, 0.0)
        if not started or (now - started) >= 10.0:
            _audio_window_started[sid] = now
            _audio_window_bytes[sid] = 0
        _audio_window_bytes[sid] = _audio_window_bytes.get(sid, 0) + len(audio_bytes)
        if _audio_window_bytes[sid] > _MAX_AUDIO_BYTES_PER_10S:
            logger.warning("[%s] audio_input rate limit exceeded bytes10s=%d", sid, _audio_window_bytes[sid])
            return
        await session.send_audio(audio_bytes)
    except Exception as e:
        logger.warning("[%s] audio_input error: %s", sid, e)


@sio.event
async def text_input(sid, data: dict):
    """Send text message (cross-modal text input)."""
    session = sessions.get(sid)
    if not session:
        return
    text = (data or {}).get("text", "")
    if text:
        await session.send_text(text)


@sio.event
async def barge_in(sid, data=None):
    """User interrupted — send content end + reset."""
    session = sessions.get(sid)
    if session:
        await session.handle_barge_in()


@sio.event
async def end_session(sid, data=None):
    """Gracefully end the session."""
    session = sessions.pop(sid, None)
    if session:
        await session.close()
        await sio.emit("session_ended", {}, to=sid)


def create_socket_app(other_asgi_app):
    """Wrap the FastAPI app with Socket.IO so Nova voice runs on the same server."""
    return socketio.ASGIApp(sio, other_asgi_app=other_asgi_app)
