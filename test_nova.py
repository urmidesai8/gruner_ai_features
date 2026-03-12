import asyncio
import base64
import json
import os
import sys
import uuid
from pathlib import Path

import boto3
import numpy as np
import sounddevice as sd
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

"""
Standalone voice-to-voice test script using Amazon Nova Sonic over the
experimental Bedrock Python SDK (bidirectional streaming API).

Requires:
- AWS credentials in environment (or .env) with access to Bedrock Runtime
- Region that supports `amazon.nova-sonic-v1:0`
- `sounddevice`, `numpy`, and `aws-sdk-bedrock-runtime` installed

Run from the project root:
    python test.py
"""

MODEL_ID = "amazon.nova-sonic-v1:0"

# Audio configuration for mic input (16 kHz mono) and model output (24 kHz mono)
CHUNK_SAMPLES = 1024  # number of samples per chunk we send to Nova Sonic
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000


def _load_dotenv(dotenv_path: str | None = None) -> None:
    """
    Minimal .env loader so this script can run standalone without pydantic.
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


def check_nova_sonic_access() -> None:
    """
    Quick sanity check: is Nova Sonic visible in this account/region?

    Uses the control-plane Bedrock client via boto3.
    """
    _load_dotenv()

    # Prefer explicit NOVA_REGION but allow your default AWS region to be ap-south-1.
    # Note: as of Nova v1 docs, Sonic is in us-east-1, eu-north-1, ap-northeast-1.
    region = (
        os.getenv("NOVA_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "ap-south-1"
    )

    print(f"Checking Nova Sonic access in region: {region}")
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        resp = bedrock.list_foundation_models()
    except Exception as e:
        print(f"Error calling list_foundation_models: {e}", file=sys.stderr)
        return

    models = [m.get("modelId") for m in resp.get("modelSummaries", [])]
    if MODEL_ID in models:
        print("✅ Nova Sonic is visible in this region and account.")
    else:
        print("❌ Nova Sonic is NOT listed in this region.")
        print(
            "If your main region is ap-south-1, set NOVA_REGION to a Sonic region "
            "(for example us-east-1) in your .env and try again."
        )


async def run_nova_sonic() -> None:
    """
    One-shot voice-to-voice interaction with Nova Sonic using the
    experimental aws-sdk-bedrock-runtime client.
    """
    # Load credentials from local .env so this script is fully standalone.
    # .env can contain either the project-style keys or standard AWS keys.
    _load_dotenv()

    # Map project-style keys to standard AWS env vars if needed.
    if os.getenv("AWS_ACCESS_KEY_ID") is None and os.getenv("AWS_ACCESS_KEY"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY", "")
    if os.getenv("AWS_SECRET_ACCESS_KEY") is None and os.getenv("AWS_SECRET_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_KEY", "")

    # Your default region is ap-south-1, but Nova Sonic v1 is only
    # available in a few regions. Prefer NOVA_REGION override.
    region = (
        os.getenv("NOVA_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "ap-south-1"
    )

    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print(
            "Missing AWS credentials. Ensure .env has AWS_ACCESS_KEY / AWS_SECRET_KEY "
            "or standard AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize experimental Bedrock runtime client for bidirectional streaming.
    config = Config(
        endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
    )
    client = BedrockRuntimeClient(config=config)

    # Record a short clip from the microphone (blocking)
    duration_sec = 10
    print(f"Speak to the Nova Sonic assistant... (recording ~{duration_sec} seconds)")
    recording = sd.rec(
        int(duration_sec * INPUT_RATE),
        samplerate=INPUT_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    print("Finished recording microphone audio, preparing to send to Nova Sonic...")

    # Unique IDs for this prompt/content
    prompt_name = str(uuid.uuid4())
    system_content_name = str(uuid.uuid4())
    audio_content_name = str(uuid.uuid4())

    # Open bidirectional stream
    print(f"Opening Nova Sonic bidirectional stream in region {region} with model {MODEL_ID}...")
    stream = await client.invoke_model_with_bidirectional_stream(
        InvokeModelWithBidirectionalStreamOperationInput(model_id=MODEL_ID)
    )

    async def send_event(event_obj: dict) -> None:
        payload = json.dumps(event_obj)
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=payload.encode("utf-8"))
        )
        await stream.input_stream.send(chunk)

    # 1) sessionStart
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
    print("Sent sessionStart event.")

    # 2) promptStart with audio output config
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
    print("Sent promptStart event with text+audio output configuration.")

    # 3) System TEXT prompt (matches the official simple example)
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
    print("Sent system contentStart event.")

    system_prompt = (
        "You are a friendly assistant. The user and you will engage in a spoken dialog "
        "exchanging the transcripts of a natural real-time conversation. Keep your responses short, "
        "generally two or three sentences for chatty scenarios."
    )
    print("Sent system textInput event.")

    await send_event(
        {
            "event": {
                "textInput": {
                    "promptName": prompt_name,
                    "contentName": system_content_name,
                    "content": system_prompt,
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
    print("Sent system contentEnd event.")

    # 4) contentStart for AUDIO (user mic input)
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
                        "sampleRateHertz": INPUT_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        }
    )
    print("Sent audio contentStart event (user microphone input).")

    # 5) Send recorded audio in chunks as audioInput
    audio_bytes = recording.tobytes()
    bytes_per_sample = 2  # int16
    chunk_size_bytes = CHUNK_SAMPLES * CHANNELS * bytes_per_sample
    total_chunks = 0
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
        total_chunks += 1

    print(f"Finished sending audioInput chunks to Nova Sonic (chunks sent: {total_chunks}).")

    # 6) contentEnd for audio, then promptEnd
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
    print("Sent audio contentEnd event.")

    await send_event({"event": {"promptEnd": {"promptName": prompt_name}}})
    print("Sent promptEnd event. Waiting for AI response (audio will play through speakers)...")

    playback_buffer = bytearray()
    received_anything = False

    try:
        while True:
            try:
                # Avoid hanging forever if nothing comes back
                output = await asyncio.wait_for(stream.await_output(), timeout=30)
            except asyncio.TimeoutError:
                if not received_anything:
                    print(
                        "No response received from Nova Sonic within 30 seconds. "
                        "Check that the model is enabled in your account and the region is correct."
                    )
                break

            # output is typically (metadata, stream_chunk)
            # but can be None when the stream is closed.
            if output is None:
                print("Nova Sonic closed the stream without further events.")
                break

            result = await output[1].receive()

            # Some SDK versions may return None when the stream is complete.
            if result is None or not getattr(result, "value", None) or not getattr(result.value, "bytes_", None):
                continue

            if not received_anything:
                print("First response event received from Nova Sonic.")
                received_anything = True
            response_data = result.value.bytes_.decode("utf-8")
            try:
                json_data = json.loads(response_data)
            except json.JSONDecodeError:
                continue

            event = json_data.get("event", {})

            if "textOutput" in event:
                text = event["textOutput"].get("content", "")
                if text:
                    print("AI (transcript):", text)

            elif "audioOutput" in event:
                audio_content = event["audioOutput"].get("content")
                if audio_content:
                    playback_buffer.extend(base64.b64decode(audio_content))

            elif "completionEnd" in event or "sessionEnd" in event:
                break

    except Exception as e:
        print(f"Error while receiving Nova Sonic response: {e}", file=sys.stderr)

    # Close the stream input
    try:
        await stream.input_stream.close()
    except Exception:
        pass

    # Play buffered audio
    if playback_buffer:
        output_samples = np.frombuffer(playback_buffer, dtype=np.int16)
        sd.play(output_samples, samplerate=OUTPUT_RATE)
        sd.wait()


def main() -> None:
    try:
        asyncio.run(run_nova_sonic())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()