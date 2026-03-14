"""
BedrockS2SSession — bidirectional streaming session with Amazon Nova 2 Sonic.

Requires:
  pip install aws-sdk-bedrock-runtime boto3

Key fix: boto3 resolves credentials from ALL standard sources
(~/.aws/credentials profiles, env vars, IAM roles, SSO, etc.)
and we inject them as env vars so the experimental SDK can use them.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import boto3
import socketio

# ── AWS experimental SDK ───────────────────────────────────────────────────────
from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme

# ── Credentials resolver (try both SDK versions) ──────────────────────────────
try:
    from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver
except ImportError:
    try:
        from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver
    except ImportError:
        EnvironmentCredentialsResolver = None

logger = logging.getLogger("bedrock_session")

MODEL_ID = "amazon.nova-2-sonic-v1:0"
SESSION_START_TIMEOUT = 30  # seconds


def _inject_credentials_from_boto3(region: str):
    """
    Resolve AWS credentials via boto3 (handles profiles, env, IAM roles, SSO…)
    and write them into the environment so EnvironmentCredentialsResolver finds them.
    Returns the resolved region.
    """
    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError(
            "No AWS credentials found. Configure via ~/.aws/credentials, "
            "AWS_PROFILE, AWS_ACCESS_KEY_ID, or an IAM role."
        )
    frozen = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    elif "AWS_SESSION_TOKEN" in os.environ:
        del os.environ["AWS_SESSION_TOKEN"]

    # Resolve region: arg > env > profile default > fallback
    resolved = (
        region
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_REGION")
        or session.region_name
        or "us-east-1"
    )
    os.environ["AWS_DEFAULT_REGION"] = resolved
    logger.info(f"Credentials resolved via boto3 for region={resolved} "
                f"(key=...{frozen.access_key[-4:]})")
    return resolved


class BedrockS2SSession:
    def __init__(
        self,
        sid: str,
        sio: socketio.AsyncServer,
        region: str,
        voice: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        sample_rate: int,
        tool_specs: List[dict],
        tool_registry,
    ):
        self.sid = sid
        self.sio = sio
        self.region = region
        self.voice = voice
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.sample_rate = sample_rate
        self.tool_specs = tool_specs
        self.tool_registry = tool_registry

        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

        self._closed = False
        self._stream = None
        self._bedrock_client = None
        self._recv_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._send_queue: asyncio.Queue = asyncio.Queue()

        # Tool state
        self._tool_name: str = ""
        self._tool_use_id: str = ""
        self._tool_content: dict = {}

        # Role / display state
        self._current_role: str = ""
        self._display_assistant_text: bool = True

    # ── Client factory ────────────────────────────────────────────────────────

    def _make_client(self) -> BedrockRuntimeClient:
        if EnvironmentCredentialsResolver is None:
            raise RuntimeError(
                "aws-sdk-bedrock-runtime not installed correctly. "
                "Run: pip install aws-sdk-bedrock-runtime"
            )

        # Inject credentials from boto3 so EnvironmentCredentialsResolver finds them
        resolved_region = _inject_credentials_from_boto3(self.region)
        self.region = resolved_region

        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
        )
        return BedrockRuntimeClient(config)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        loop = asyncio.get_event_loop()

        # Build client in executor (may do I/O for credential resolution)
        self._bedrock_client = await loop.run_in_executor(None, self._make_client)

        logger.info(f"[{self.sid}] Opening Bedrock bidirectional stream...")

        # Open stream with a hard timeout so we never hang indefinitely
        try:
            self._stream = await asyncio.wait_for(
                self._bedrock_client.invoke_model_with_bidirectional_stream(
                    InvokeModelWithBidirectionalStreamOperationInput(model_id=MODEL_ID)
                ),
                timeout=SESSION_START_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Timed out opening Bedrock stream after {SESSION_START_TIMEOUT}s. "
                "Check your AWS region and Bedrock Nova Sonic model access."
            )

        # Background tasks
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

        # Handshake
        await self._send(self._evt_session_start())
        await self._send(self._evt_prompt_start())
        await self._send(self._evt_system_prompt_start())
        await self._send(self._evt_system_prompt_text())
        await self._send(self._evt_system_prompt_end())
        await self._send(self._evt_audio_content_start())

        logger.info(f"[{self.sid}] Session opened (region={self.region}, voice={self.voice})")

    async def close(self):
        if self._closed:
            return
        self._closed = True
        logger.info(f"[{self.sid}] Closing session")
        try:
            if self._stream and self._send_queue:
                await self._send(self._evt_audio_content_end())
                await self._send(self._evt_prompt_end())
                await self._send(self._evt_session_end())
                await asyncio.sleep(0.4)
        except Exception:
            pass
        for task in [self._send_task, self._recv_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # ── Send loop ─────────────────────────────────────────────────────────────

    async def _send(self, json_str: str):
        await self._send_queue.put(json_str)

    async def _send_loop(self):
        try:
            while not self._closed:
                try:
                    json_str = await asyncio.wait_for(self._send_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                chunk = InvokeModelWithBidirectionalStreamInputChunk(
                    value=BidirectionalInputPayloadPart(bytes_=json_str.encode("utf-8"))
                )
                try:
                    await self._stream.input_stream.send(chunk)
                except Exception as e:
                    if not self._closed:
                        logger.error(f"[{self.sid}] Send error: {e}")
                    break
        except asyncio.CancelledError:
            pass

    # ── Receive loop ──────────────────────────────────────────────────────────

    async def _recv_loop(self):
        try:
            while not self._closed:
                try:
                    output = await self._stream.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        raw = result.value.bytes_.decode("utf-8")
                        try:
                            await self._handle_event(json.loads(raw))
                        except json.JSONDecodeError:
                            logger.warning(f"[{self.sid}] Non-JSON chunk: {raw[:80]}")
                except StopAsyncIteration:
                    logger.info(f"[{self.sid}] Stream closed by server")
                    await self.sio.emit("session_ended", {}, to=self.sid)
                    break
                except Exception as e:
                    if not self._closed:
                        err = str(e)
                        logger.error(f"[{self.sid}] Recv error: {err}")
                        await self.sio.emit("error", {"message": err}, to=self.sid)
                    break
        except asyncio.CancelledError:
            pass

    # ── Event dispatcher ──────────────────────────────────────────────────────

    async def _handle_event(self, data: dict):
        event = data.get("event", {})
        if not event:
            return

        if "contentStart" in event:
            cs = event["contentStart"]
            self._current_role = cs.get("role", "")
            extra = cs.get("additionalModelFields", "")
            if extra:
                try:
                    fields = json.loads(extra)
                    self._display_assistant_text = (
                        fields.get("generationStage") == "SPECULATIVE"
                    )
                except Exception:
                    self._display_assistant_text = True
            else:
                self._display_assistant_text = True
            # Notify frontend a new content block is starting
            await self.sio.emit("content_start", {"role": self._current_role}, to=self.sid)

        elif "textOutput" in event:
            text = event["textOutput"].get("content", "")
            role = event["textOutput"].get("role", self._current_role).upper()
            if '{ "interrupted" : true }' in text:
                await self.sio.emit("barge_in_detected", {}, to=self.sid)
                return
            if role == "ASSISTANT" and self._display_assistant_text:
                await self.sio.emit("transcript", {"text": text, "role": "assistant"}, to=self.sid)
            elif role == "USER":
                await self.sio.emit("transcript", {"text": text, "role": "user"}, to=self.sid)

        elif "audioOutput" in event:
            audio = event["audioOutput"].get("content", "")
            await self.sio.emit("audio_output", {"audio": audio}, to=self.sid)

        elif "toolUse" in event:
            tu = event["toolUse"]
            self._tool_name = tu.get("toolName", "")
            self._tool_use_id = tu.get("toolUseId", "")
            self._tool_content = tu
            logger.info(f"[{self.sid}] Tool invoked: {self._tool_name}")
            await self.sio.emit("tool_call", {
                "name": self._tool_name,
                "input": tu.get("content", {}),
            }, to=self.sid)

        elif "contentEnd" in event:
            ce = event["contentEnd"]
            await self.sio.emit("content_end", {"type": ce.get("type", "")}, to=self.sid)
            if ce.get("type") == "TOOL":
                await self._execute_tool()

        elif "completionEnd" in event:
            # Full response turn is complete — tell frontend to finalise bubble
            await self.sio.emit("message_stop", {"stopReason": "end_turn"}, to=self.sid)

    # ── Public send API ───────────────────────────────────────────────────────

    async def send_audio(self, audio_bytes: bytes):
        if self._closed:
            return
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        await self._send(json.dumps({
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": b64,
                }
            }
        }))

    async def send_text(self, text: str):
        if self._closed:
            return
        cname = str(uuid.uuid4())
        for evt in [
            json.dumps({"event": {"contentStart": {
                "promptName": self.prompt_name, "contentName": cname,
                "type": "TEXT", "interactive": True, "role": "USER",
                "textInputConfiguration": {"mediaType": "text/plain"},
            }}}),
            json.dumps({"event": {"textInput": {
                "promptName": self.prompt_name, "contentName": cname,
                "content": text,
            }}}),
            json.dumps({"event": {"contentEnd": {
                "promptName": self.prompt_name, "contentName": cname,
            }}}),
        ]:
            await self._send(evt)

    async def handle_barge_in(self):
        if self._closed:
            return
        logger.info(f"[{self.sid}] Barge-in received")
        await self._send(json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                }
            }
        }))
        self.audio_content_name = str(uuid.uuid4())
        await self._send(self._evt_audio_content_start())

    # ── Tool execution ────────────────────────────────────────────────────────

    async def _execute_tool(self):
        tool_name, tool_use_id, tool_content = (
            self._tool_name, self._tool_use_id, self._tool_content
        )
        self._tool_name = self._tool_use_id = ""
        self._tool_content = {}
        if not tool_name:
            return

        raw = tool_content.get("content", {})
        if isinstance(raw, str):
            try:
                tool_input = json.loads(raw)
            except json.JSONDecodeError:
                tool_input = {}
        elif isinstance(raw, dict):
            tool_input = raw
        else:
            tool_input = {}

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: self.tool_registry.execute(tool_name, tool_input)
            )
        except Exception as e:
            result = {"error": str(e)}

        logger.info(f"[{self.sid}] Tool [{tool_name}]: {str(result)[:160]}")
        await self.sio.emit("tool_result", {"name": tool_name, "result": result}, to=self.sid)

        rname = str(uuid.uuid4())
        for evt in [
            json.dumps({"event": {"contentStart": {
                "promptName": self.prompt_name, "contentName": rname,
                "type": "TOOL", "role": "TOOL",
                "toolResultInputConfiguration": {
                    "toolUseId": tool_use_id, "type": "TEXT",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                },
            }}}),
            json.dumps({"event": {"toolResult": {
                "promptName": self.prompt_name, "contentName": rname,
                "content": json.dumps(result),
            }}}),
            json.dumps({"event": {"contentEnd": {
                "promptName": self.prompt_name, "contentName": rname,
            }}}),
        ]:
            await self._send(evt)

    # ── Event builders ────────────────────────────────────────────────────────

    def _evt_session_start(self) -> str:
        return json.dumps({"event": {"sessionStart": {"inferenceConfiguration": {
            "maxTokens": self.max_tokens,
            "topP": self.top_p,
            "temperature": self.temperature,
        }}}})

    def _evt_prompt_start(self) -> str:
        cfg: Dict[str, Any] = {
            "promptName": self.prompt_name,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": self.sample_rate,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": self.voice,
                "encoding": "base64",
                "audioType": "SPEECH",
            },
        }
        if self.tool_specs:
            cfg["toolConfiguration"] = {
                "tools": self.tool_specs,
                "toolChoice": {"auto": {}},
            }
        return json.dumps({"event": {"promptStart": cfg}})

    def _evt_system_prompt_start(self) -> str:
        return json.dumps({"event": {"contentStart": {
            "promptName": self.prompt_name,
            "contentName": self.content_name,
            "type": "TEXT",
            "interactive": False,
            "role": "SYSTEM",
            "textInputConfiguration": {"mediaType": "text/plain"},
        }}})

    def _evt_system_prompt_text(self) -> str:
        return json.dumps({"event": {"textInput": {
            "promptName": self.prompt_name,
            "contentName": self.content_name,
            "content": self.system_prompt,
        }}})

    def _evt_system_prompt_end(self) -> str:
        return json.dumps({"event": {"contentEnd": {
            "promptName": self.prompt_name,
            "contentName": self.content_name,
        }}})

    def _evt_audio_content_start(self) -> str:
        return json.dumps({"event": {"contentStart": {
            "promptName": self.prompt_name,
            "contentName": self.audio_content_name,
            "type": "AUDIO",
            "interactive": True,
            "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 16000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "audioType": "SPEECH",
                "encoding": "base64",
            },
        }}})

    def _evt_audio_content_end(self) -> str:
        return json.dumps({"event": {"contentEnd": {
            "promptName": self.prompt_name,
            "contentName": self.audio_content_name,
        }}})

    def _evt_prompt_end(self) -> str:
        return json.dumps({"event": {"promptEnd": {"promptName": self.prompt_name}}})

    def _evt_session_end(self) -> str:
        return json.dumps({"event": {"sessionEnd": {}}})
