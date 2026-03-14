"""
Nova 2 Sonic Voicebot - FastAPI + Socket.IO Server
Python equivalent of the Express/TypeScript implementation.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

import boto3  # used indirectly via bedrock_session credential injection
import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from bedrock_session import BedrockS2SSession
from tools import ToolRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("server")

# ─── FastAPI + Socket.IO setup ────────────────────────────────────────────────

app = FastAPI(title="Nova Sonic Voicebot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    max_http_buffer_size=10 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25,
)

# Wrap FastAPI app with Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Active sessions: sid -> BedrockS2SSession
sessions: Dict[str, BedrockS2SSession] = {}

tool_registry = ToolRegistry()

# ─── Static files & routes ────────────────────────────────────────────────────

PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "..", "public")

@app.get("/")
async def root():
    return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}

# Mount static assets after dynamic routes
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")


# ─── Socket.IO event handlers ─────────────────────────────────────────────────

@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    await sio.emit("connected", {"sid": sid}, to=sid)


@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    session = sessions.pop(sid, None)
    if session:
        await session.close()


@sio.event
async def start_session(sid, data: dict):
    """
    Start a new Bedrock Nova Sonic streaming session.
    data = { region, voice, systemPrompt, temperature, topP, maxTokens,
             tools, responseRate, sampleRate }
    """
    logger.info(f"[{sid}] start_session: {json.dumps({k: v for k, v in data.items() if k != 'systemPrompt'})}")

    # Clean up any existing session
    old = sessions.pop(sid, None)
    if old:
        await old.close()

    region = data.get("region", "us-east-1")
    voice = data.get("voice", "tiffany")
    system_prompt = data.get("systemPrompt", "You are a helpful voice assistant.")
    temperature = float(data.get("temperature", 0.7))
    top_p = float(data.get("topP", 0.9))
    max_tokens = int(data.get("maxTokens", 1024))
    sample_rate = int(data.get("sampleRate", 24000))
    enabled_tools = data.get("tools", [])

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
        logger.info(f"[{sid}] Session started successfully")
    except Exception as e:
        logger.error(f"[{sid}] Failed to start session: {e}")
        sessions.pop(sid, None)
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def audio_input(sid, data):
    """Receive raw PCM audio chunk from browser (base64 encoded)."""
    session = sessions.get(sid)
    if not session:
        return
    try:
        # data may be bytes or base64 string
        if isinstance(data, (bytes, bytearray)):
            audio_bytes = bytes(data)
        else:
            audio_bytes = base64.b64decode(data)
        await session.send_audio(audio_bytes)
    except Exception as e:
        logger.warning(f"[{sid}] audio_input error: {e}")


@sio.event
async def text_input(sid, data: dict):
    """Send text message (cross-modal text input)."""
    session = sessions.get(sid)
    if not session:
        return
    text = data.get("text", "")
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


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 3000))
    logger.info(f"Starting Nova Sonic Voicebot on {host}:{port}")
    uvicorn.run(
        "server:socket_app",
        host=host,
        port=port,
        log_level="info",
        reload=os.environ.get("DEV", "").lower() == "true",
    )
