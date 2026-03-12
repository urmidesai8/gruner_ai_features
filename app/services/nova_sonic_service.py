import asyncio
import base64
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Tuple

import numpy as np
from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import Config
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

MODEL_ID = "amazon.nova-sonic-v1:0"

# Audio configuration for Nova Sonic
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SAMPLES = 1024  # number of samples per chunk we send to Nova Sonic


def _load_dotenv(dotenv_path: str | None = None) -> None:
    """
    Minimal .env loader so this service can run without pydantic.
    Only supports simple KEY=VALUE lines.
    """
    path = Path(dotenv_path or ".env")
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _ensure_aws_env_credentials() -> None:
    """
    Map project-style keys to standard AWS env vars if needed.
    """
    if os.getenv("AWS_ACCESS_KEY_ID") is None and os.getenv("AWS_ACCESS_KEY"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY", "")
    if os.getenv("AWS_SECRET_ACCESS_KEY") is None and os.getenv("AWS_SECRET_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_KEY", "")


def _resolve_region() -> str:
    """
    Resolve the region for Nova Sonic.
    Prefer NOVA_REGION, then standard AWS region env vars, then ap-south-1.
    """
    return (
        os.getenv("NOVA_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "ap-south-1"
    )


async def run_nova_sonic_on_pcm(
    pcm_samples: np.ndarray,
    input_rate: int = INPUT_RATE,
    system_prompt: str | None = None,
) -> Tuple[str, bytes]:
    """
    Run a single voice-to-voice interaction with Amazon Nova Sonic using
    pre-recorded PCM int16 audio samples.

    Args:
        pcm_samples: 1D numpy array of int16 samples (mono).
        input_rate: Sample rate of the audio (default 16 kHz).
        system_prompt: Optional system prompt to steer the assistant's behavior.

    Returns:
        (transcript_text, audio_bytes) where:
            - transcript_text is the assistant's text transcript (may be empty).
            - audio_bytes is raw int16 PCM at 24 kHz mono suitable for playback.
    """
    if pcm_samples.dtype != np.int16:
        pcm_samples = pcm_samples.astype("int16")

    _load_dotenv()
    _ensure_aws_env_credentials()

    region = _resolve_region()

    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        raise RuntimeError(
            "Missing AWS credentials. Ensure .env has AWS_ACCESS_KEY / AWS_SECRET_KEY "
            "or standard AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
        )

    config = Config(
        endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
    )
    client = BedrockRuntimeClient(config=config)

    prompt_name = str(uuid.uuid4())
    system_content_name = str(uuid.uuid4())
    audio_content_name = str(uuid.uuid4())

    stream = await client.invoke_model_with_bidirectional_stream(
        InvokeModelWithBidirectionalStreamOperationInput(model_id=MODEL_ID)
    )

    async def send_event(event_obj: dict) -> None:
        payload = json.dumps(event_obj)
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=payload.encode("utf-8"))
        )
        await stream.input_stream.send(chunk)

    await send_event(
        {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7,
                    }
                }
            }
        }
    )

    await send_event(
        {
            "event": {
                "promptStart": {
                    "promptName": prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": OUTPUT_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                }
            }
        }
    )

    await send_event(
        {
            "event": {
                "contentStart": {
                    "promptName": prompt_name,
                    "contentName": system_content_name,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "SYSTEM",
                    "textInputConfiguration": {
                        "mediaType": "text/plain",
                    },
                }
            }
        }
    )

    system_prompt_value = system_prompt or (
        "You are a friendly assistant. The user and you will engage in a spoken dialog "
        "exchanging the transcripts of a natural real-time conversation. Keep your responses short, "
        "generally two or three sentences for chatty scenarios."
    )

    await send_event(
        {
            "event": {
                "textInput": {
                    "promptName": prompt_name,
                    "contentName": system_content_name,
                    "content": system_prompt_value,
                }
            }
        }
    )

    await send_event(
        {
            "event": {
                "contentEnd": {
                    "promptName": prompt_name,
                    "contentName": system_content_name,
                }
            }
        }
    )

    await send_event(
        {
            "event": {
                "contentStart": {
                    "promptName": prompt_name,
                    "contentName": audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": input_rate,
                        "sampleSizeBits": 16,
                        "channelCount": CHANNELS,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        }
    )

    audio_bytes = pcm_samples.tobytes()
    bytes_per_sample = 2  # int16
    chunk_size_bytes = CHUNK_SAMPLES * CHANNELS * bytes_per_sample
    for i in range(0, len(audio_bytes), chunk_size_bytes):
        chunk = audio_bytes[i : i + chunk_size_bytes]
        if not chunk:
            continue
        blob = base64.b64encode(chunk).decode("utf-8")
        await send_event(
            {
                "event": {
                    "audioInput": {
                        "promptName": prompt_name,
                        "contentName": audio_content_name,
                        "content": blob,
                    }
                }
            }
        )

    await send_event(
        {
            "event": {
                "contentEnd": {
                    "promptName": prompt_name,
                    "contentName": audio_content_name,
                }
            }
        }
    )

    await send_event({"event": {"promptEnd": {"promptName": prompt_name}}})

    playback_buffer = bytearray()
    transcript_parts: list[str] = []

    try:
        while True:
            try:
                output = await asyncio.wait_for(stream.await_output(), timeout=30)
            except asyncio.TimeoutError:
                break

            if output is None:
                break

            result = await output[1].receive()

            if (
                result is None
                or not getattr(result, "value", None)
                or not getattr(result.value, "bytes_", None)
            ):
                continue

            response_data = result.value.bytes_.decode("utf-8")
            try:
                json_data = json.loads(response_data)
            except json.JSONDecodeError:
                continue

            event = json_data.get("event", {})

            if "textOutput" in event:
                text = event["textOutput"].get("content", "")
                if text:
                    transcript_parts.append(text)
            elif "audioOutput" in event:
                audio_content = event["audioOutput"].get("content")
                if audio_content:
                    playback_buffer.extend(base64.b64decode(audio_content))
            elif "completionEnd" in event or "sessionEnd" in event:
                break
    finally:
        try:
            await stream.input_stream.close()
        except Exception:
            pass

    transcript = "".join(transcript_parts).strip()
    return transcript, bytes(playback_buffer)


def run_nova_sonic_on_pcm_sync(
    pcm_samples: np.ndarray,
    input_rate: int = INPUT_RATE,
    system_prompt: str | None = None,
) -> Tuple[str, bytes]:
    """
    Synchronous wrapper around run_nova_sonic_on_pcm for use in sync contexts.
    """
    return asyncio.run(run_nova_sonic_on_pcm(pcm_samples, input_rate=input_rate, system_prompt=system_prompt))

