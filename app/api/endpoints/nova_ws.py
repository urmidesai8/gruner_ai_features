"""
Native FastAPI WebSocket endpoint for Amazon Nova Sonic (bidirectional voice).

This intentionally co-exists with the Socket.IO implementation.
The WebSocket endpoint is designed for production use with:
- binary audio frames (16-bit PCM) from clients
- JSON control messages (start_session, text_input, barge_in, end_session)
- structured JSON events back to clients (transcript, audio_output, message_stop, error)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid
from typing import Optional
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.backend.bedrock_session import BedrockS2SSession
from app.backend.tools import ToolRegistry

# Use Uvicorn's logger handlers so logs reliably show up in terminal.
logger = logging.getLogger("uvicorn.error").getChild("nova_ws")

router = APIRouter()

_REGION_RE = re.compile(r"^[a-z]{2}-[a-z0-9-]+-\d+$")
_START_SESSION_GRACE_SECONDS = 20


class _WsEmitter:
    """Adapter so BedrockS2SSession can emit events to a WebSocket."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._send_lock = asyncio.Lock()

    async def emit(self, event: str, data: dict, to: Optional[str] = None) -> None:  # noqa: ARG002
        # Standardized envelope: { event, data }
        try:
            # Starlette/FastAPI WebSocket send is not safe concurrently across tasks.
            # BedrockS2SSession emits from background tasks; serialize all sends.
            async with self._send_lock:
                await self.websocket.send_json({"event": event, "data": data})
        except Exception:
            # Client may have disconnected; avoid cascading failures from background tasks.
            logger.warning("Nova WS emit failed event=%s", event, exc_info=True)


def _safe_region(raw: object) -> str:
    """
    Normalize and validate region to avoid invalid Bedrock endpoint hostnames like:
    bedrock-runtime..amazonaws.com
    """
    region = (str(raw) if raw is not None else "").strip()
    if not region:
        return "us-east-1"
    if not _REGION_RE.match(region):
        raise ValueError(f"Invalid AWS region: {region!r}")
    return region


@router.websocket("/ws/nova")
async def nova_sonic_ws(websocket: WebSocket) -> None:
    """
    WebSocket protocol:

    Client -> Server (text JSON):
      - { "type": "start_session", "region", "voice", "systemPrompt", "temperature", "topP", "maxTokens", "sampleRate", "tools": [] }
      - { "type": "text_input", "text": "..." }
      - { "type": "barge_in" }
      - { "type": "end_utterance" }  (force end-of-utterance for VAD-driven turn taking)
      - { "type": "end_session" }
      - { "type": "audio_base64", "data": "<base64>" }  (fallback)

    Client -> Server (binary):
      - raw bytes of 16-bit little-endian PCM (mono, 16kHz recommended)

    Server -> Client (JSON):
      - { "event": "session_started", "data": {"success": true, "session_id": "..."} }
      - { "event": "transcript", "data": {"text": "...", "role": "assistant"|"user"} }
      - { "event": "audio_output", "data": {"audio": "<base64-lpcm>"} }
      - { "event": "message_stop", "data": {"stopReason": "end_turn"} }
      - { "event": "error", "data": {"message": "..."} }
      - { "event": "session_ended", "data": {} }
    """

    session_id = str(uuid.uuid4())
    client = websocket.client.host if websocket.client else "unknown"
    await websocket.accept()
    logger.info("Nova WS connected session_id=%s client=%s", session_id, client)

    emitter = _WsEmitter(websocket)
    tool_registry = ToolRegistry()
    session: Optional[BedrockS2SSession] = None
    session_started = False
    started_at = time.time()
    first_audio_at: float | None = None
    audio_frames = 0
    audio_bytes = 0

    async def _close_session(reason: str) -> None:
        nonlocal session, session_started
        if session is not None:
            try:
                await session.close()
            except Exception:
                logger.exception("Nova WS close failed session_id=%s reason=%s", session_id, reason)
            session = None
        session_started = False

    try:
        # Require clients to start a session promptly (prevents idle open sockets).
        start_deadline = asyncio.get_event_loop().time() + _START_SESSION_GRACE_SECONDS
        while True:
            # If no session is started within the grace window, close.
            if not session_started and asyncio.get_event_loop().time() > start_deadline:
                await emitter.emit("error", {"message": "start_session not received in time; closing connection."})
                logger.info("Nova WS closing idle connection session_id=%s", session_id)
                break

            # Use a short timeout so we can enforce deadlines and react promptly.
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            mtype = msg.get("type")

            if mtype == "websocket.disconnect":
                raise WebSocketDisconnect()

            if mtype == "websocket.receive" and msg.get("bytes") is not None:
                if session is None or not session_started:
                    # Ignore audio until session is started
                    continue
                try:
                    b = msg["bytes"]
                    if first_audio_at is None:
                        first_audio_at = time.time()
                        logger.info(
                            "Nova WS first audio session_id=%s after=%.3fs bytes=%d",
                            session_id,
                            first_audio_at - started_at,
                            len(b),
                        )
                    audio_frames += 1
                    audio_bytes += len(b)
                    if audio_frames % 100 == 0:
                        logger.info(
                            "Nova WS audio stats session_id=%s frames=%d bytes=%d",
                            session_id,
                            audio_frames,
                            audio_bytes,
                        )
                    await session.send_audio(b)
                except Exception as e:
                    logger.warning("Nova WS audio_input error session_id=%s err=%s", session_id, e)
                continue

            if mtype != "websocket.receive":
                continue

            text = msg.get("text")
            if not text:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await emitter.emit("error", {"message": "Invalid JSON message."})
                continue

            ptype = (payload.get("type") or "").strip()

            if ptype == "start_session":
                await _close_session("restart")
                try:
                    region = _safe_region(payload.get("region"))
                except ValueError as ve:
                    await emitter.emit("error", {"message": str(ve)})
                    continue
                voice = payload.get("voice", "tiffany")
                system_prompt = payload.get("systemPrompt", "You are a helpful voice assistant in the chat.")
                temperature = float(payload.get("temperature", 0.7))
                top_p = float(payload.get("topP", 0.9))
                max_tokens = int(payload.get("maxTokens", 1024))
                sample_rate = int(payload.get("sampleRate", 24000))
                enabled_tools = payload.get("tools", [])
                logger.info(
                    "Nova WS start_session session_id=%s region=%s voice=%s sample_rate=%s tools=%d",
                    session_id,
                    region,
                    voice,
                    sample_rate,
                    len(enabled_tools) if isinstance(enabled_tools, list) else 0,
                )

                tool_specs = tool_registry.get_specs(enabled_tools)
                session = BedrockS2SSession(
                    sid=session_id,
                    sio=emitter,  # adapter with .emit()
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
                try:
                    # Hard upper bound so UI doesn't "hang" for minutes.
                    t0 = time.time()
                    await asyncio.wait_for(session.start(), timeout=45.0)
                    logger.info(
                        "Nova WS bedrock session.start ok session_id=%s took=%.3fs",
                        session_id,
                        time.time() - t0,
                    )
                    session_started = True
                    await emitter.emit("session_started", {"success": True, "session_id": session_id})
                    logger.info("Nova WS session_started session_id=%s region=%s voice=%s", session_id, region, voice)
                except Exception as e:
                    logger.exception("Nova WS start_session failed session_id=%s err=%s", session_id, e)
                    await _close_session("start_failed")
                    await emitter.emit("error", {"message": str(e)})

            elif ptype == "text_input":
                if session is None or not session_started:
                    await emitter.emit("error", {"message": "Session not started. Send start_session first."})
                    continue
                txt = payload.get("text", "")
                if txt:
                    logger.info("Nova WS text_input session_id=%s len=%d", session_id, len(txt))
                    await session.send_text(txt)

            elif ptype == "barge_in":
                if session is not None and session_started:
                    logger.info("Nova WS barge_in session_id=%s", session_id)
                    await session.handle_barge_in()

            elif ptype == "end_utterance":
                # For client-side VAD: end current audio content so Nova can finalize ASR/response,
                # then immediately start a fresh audio content block for the next utterance.
                if session is not None and session_started:
                    logger.info("Nova WS end_utterance session_id=%s", session_id)
                    await session.handle_barge_in()

            elif ptype == "audio_base64":
                if session is None or not session_started:
                    continue
                b64 = payload.get("data", "")
                if not b64:
                    continue
                try:
                    await session.send_audio(base64.b64decode(b64))
                except Exception as e:
                    logger.warning("Nova WS audio_base64 error session_id=%s err=%s", session_id, e)

            elif ptype == "end_session":
                await _close_session("client_end")
                await emitter.emit("session_ended", {})
                logger.info("Nova WS session_ended session_id=%s", session_id)

            else:
                await emitter.emit("error", {"message": f"Unknown message type: {ptype or '(missing)'}"})

    except WebSocketDisconnect:
        logger.info("Nova WS disconnected session_id=%s", session_id)
    except Exception:
        logger.exception("Nova WS crashed session_id=%s", session_id)
    finally:
        await _close_session("finally")
        try:
            await websocket.close()
        except Exception:
            pass

