"""
Text-to-speech service for Edge TTS and Svara TTS (via AWS SageMaker).
Uses edge-tts (Microsoft Edge TTS, no API key required) for Edge TTS.
Uses boto3 + SageMaker runtime for Svara TTS.
"""
from __future__ import annotations

import asyncio
import base64
import re
import struct
import logging
import json
from typing import Optional, Tuple

import boto3

logger = logging.getLogger(__name__)

# Lazy import to avoid requiring edge_tts when not using Edge TTS
_edge_tts = None

def _get_edge_tts():
    global _edge_tts
    if _edge_tts is None:
        try:
            import edge_tts as et
            _edge_tts = et
        except ImportError:
            raise RuntimeError(
                "edge-tts is required for Edge TTS. Install with: pip install edge-tts"
            )
    return _edge_tts


async def text_to_speech(
    text: str,
    voice: Optional[str] = None,
    rate: str = "+0%",
    volume: str = "+0%",
) -> bytes:
    """
    Convert text to speech using edge-tts. Returns MP3 audio bytes.

    Args:
        text: Text to speak.
        voice: Voice id (e.g. "en-US-JennyNeural"). If None, uses config default.
        rate: Speaking rate, e.g. "+0%", "-10%".
        volume: Volume, e.g. "+0%".

    Returns:
        MP3 audio bytes.

    Raises:
        RuntimeError: If edge-tts is not installed or TTS fails.
    """
    from app.core.config import settings
    text = (text or "").strip()
    if not text:
        return b""

    voice_id = (voice or getattr(settings, "TTS_VOICE", None)) or "en-US-JennyNeural"
    et = _get_edge_tts()
    communicate = et.Communicate(text, voice_id, rate=rate, volume=volume)
    chunks = []
    try:
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                chunks.append(chunk.get("data", b""))
    except Exception as e:
        logger.exception("edge-tts stream failed")
        raise RuntimeError(f"TTS failed: {e}") from e
    return b"".join(chunks)


def _split_text_for_svara(text: str, max_chars: int) -> list[str]:
    """
    Svara's API typically expects relatively short inputs.
    Split on sentence boundaries first, then fall back to safe hard splits.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Rough sentence splitting while keeping punctuation.
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current.strip())
            current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            # Hard split long segments by whitespace to avoid breaking words.
            words = part.split()
            buf = ""
            for w in words:
                candidate = (buf + " " + w).strip()
                if len(candidate) > max_chars:
                    if buf:
                        chunks.append(buf.strip())
                    buf = w
                else:
                    buf = candidate
            if buf:
                chunks.append(buf.strip())
            continue

        candidate = (current + " " + part).strip() if current else part
        if len(candidate) > max_chars:
            flush()
            current = part
        else:
            current = candidate

    flush()
    return [c for c in chunks if c]


def _pcm_s16le_to_wav(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Wrap little-endian signed 16-bit PCM in a standard WAV header."""
    if not pcm_bytes:
        return b""
    subchunk2_size = len(pcm_bytes)
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    chunk_size = 36 + subchunk2_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM audio format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        subchunk2_size,
    )
    return header + pcm_bytes


def _invoke_sagemaker_tts(
    endpoint_name: str,
    text: str,
    voice: str,
    region: str,
    aws_access_key: str,
    aws_secret_key: str,
) -> bytes:
    """
    Synchronous helper that invokes the SageMaker endpoint using boto3.
    Returns decoded WAV audio bytes.
    """
    session_kwargs = {"region_name": region}
    if aws_access_key and aws_secret_key:
        session_kwargs["aws_access_key_id"] = aws_access_key
        session_kwargs["aws_secret_access_key"] = aws_secret_key

    boto_session = boto3.Session(**session_kwargs)
    client = boto_session.client("sagemaker-runtime")

    payload = {"inputs": text, "voice_id": voice}

    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read())

    # The endpoint may return a list; unwrap it.
    if isinstance(result, list):
        result = result[0]

    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"SageMaker Svara TTS returned error: {result['error']}")

    audio_b64 = result.get("audio_base64", "")
    if not audio_b64:
        raise RuntimeError("SageMaker Svara TTS response missing 'audio_base64'.")

    return base64.b64decode(audio_b64)


async def text_to_speech_svara(
    text: str,
    voice_id: Optional[str] = None,
) -> Tuple[bytes, str]:
    """
    Convert text to speech using the kenpath/svara-tts-v1 model hosted on
    AWS SageMaker.

    Uses synchronous boto3 run inside an asyncio executor so it does not
    block the FastAPI event loop.

    Returns:
        (audio_bytes, audio_content_type)
    """
    from app.core.config import settings

    text = (text or "").strip()
    if not text:
        return b"", "audio/wav"

    endpoint_name = (getattr(settings, "SAGEMAKER_TTS_ENDPOINT_NAME", "") or "").strip()
    if not endpoint_name:
        raise RuntimeError(
            "SageMaker endpoint name is not configured. "
            "Set SAGEMAKER_TTS_ENDPOINT_NAME in your environment."
        )

    voice = (voice_id or getattr(settings, "SVARA_VOICE_ID", "") or "").strip() or "en_male"
    max_chars = int(getattr(settings, "SVARA_MAX_TEXT_CHARS", 5000) or 5000)
    region = getattr(settings, "AWS_REGION", "ap-south-1")
    aws_access_key = getattr(settings, "AWS_ACCESS_KEY_ID", "").strip()
    aws_secret_key = getattr(settings, "AWS_SECRET_ACCESS_KEY", "").strip()

    chunks = _split_text_for_svara(text, max_chars=max_chars)
    if not chunks:
        return b"", "audio/wav"

    all_audio: list[bytes] = []
    loop = asyncio.get_event_loop()

    for chunk in chunks:
        try:
            chunk_audio = await loop.run_in_executor(
                None,
                lambda c=chunk: _invoke_sagemaker_tts(
                    endpoint_name=endpoint_name,
                    text=c,
                    voice=voice,
                    region=region,
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                ),
            )
            if chunk_audio:
                all_audio.append(chunk_audio)
        except Exception as e:
            logger.exception("SageMaker invoke_endpoint failed for chunk '%s...'", chunk[:30])
            raise RuntimeError(f"Failed to reach SageMaker Svara TTS endpoint: {e}") from e

    audio_bytes = b"".join(all_audio)
    return audio_bytes, "audio/wav"
