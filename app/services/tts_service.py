"""
Text-to-speech service for Edge TTS (voice-to-voice) assistant replies.
Uses edge-tts (Microsoft Edge TTS, no API key required).
"""
import logging
from typing import Optional

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
