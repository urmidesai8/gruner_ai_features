try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    Groq = None  # type: ignore[assignment]
from ..core.config import settings

groq_client = Groq(api_key=settings.GROQ_API_KEY) if Groq else None

# -----------------------------------------------------------------------------
# SeamlessM4T (Hugging Face): lazy-loaded for meeting transcription.
# Model is downloaded once and then loaded from cache (~/.cache/huggingface/hub).
# -----------------------------------------------------------------------------
_seamless_processor = None
_seamless_model = None
SEAMLESS_MODEL_ID = "facebook/hf-seamless-m4t-medium"


def _get_seamless_model_and_processor():
    """Load SeamlessM4T model+processor once; reuse on subsequent calls."""
    global _seamless_processor, _seamless_model
    if _seamless_processor is not None and _seamless_model is not None:
        return _seamless_processor, _seamless_model

    from transformers import AutoProcessor, SeamlessM4TModel  # type: ignore
    import torch  # type: ignore

    processor = AutoProcessor.from_pretrained(SEAMLESS_MODEL_ID)
    model = SeamlessM4TModel.from_pretrained(SEAMLESS_MODEL_ID, torch_dtype="auto")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    _seamless_processor = processor
    _seamless_model = model
    return _seamless_processor, _seamless_model


def transcribe_audio_seamless_m4t(audio_path: str, tgt_lang: str = "eng") -> str:
    """
    ASR transcription using SeamlessM4T (speech-to-text).

    Returns an error string (starting with "Error:") on failure.
    Uses librosa for loading (avoids torchaudio/torchcodec/FFmpeg on Windows).
    """
    import os
    if not os.path.isfile(audio_path):
        return f"Error: file not found: {audio_path}"

    try:
        import torch  # type: ignore
        import librosa  # type: ignore
    except ImportError as e:
        return f"Error: install dependencies (e.g. pip install torch librosa). {e}"

    try:
        processor, model = _get_seamless_model_and_processor()

        # Use librosa to avoid torchaudio/torchcodec/FFmpeg dependency on Windows
        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        inputs = processor(audios=audio, sampling_rate=sample_rate, return_tensors="pt")
        # Match model device and dtype to avoid "Input type (float) and bias type (Half)" error
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        out = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                v = v.to(device=device)
                if v.is_floating_point():
                    v = v.to(dtype=dtype)
            out[k] = v
        inputs = out

        with torch.no_grad():
            generated_tokens = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)

        # generated_tokens is token IDs for text output
        text = processor.batch_decode(generated_tokens, skip_special_tokens=True)
        out = (text[0] if text else "").strip()
        return out or "Error: empty transcription from SeamlessM4T."
    except Exception as e:
        return f"Error transcription failed: {str(e)}"

# -----------------------------------------------------------------------------
# VibeVoice-ASR (Hugging Face): lazy-loaded for meeting transcription.
# Model is downloaded once and then loaded from cache (~/.cache/huggingface/hub).
# -----------------------------------------------------------------------------
_vibevoice_model = None
_vibevoice_processor = None
VIBEVOICE_MODEL_ID = "microsoft/VibeVoice-ASR"


def _get_vibevoice_model_and_processor():
    """Load VibeVoice-ASR model and processor once; reuse from cache on subsequent calls.
    Raises ImportError or RuntimeError if the model is not available in this transformers install.
    """
    global _vibevoice_model, _vibevoice_processor
    if _vibevoice_model is not None and _vibevoice_processor is not None:
        return _vibevoice_model, _vibevoice_processor
    # VibeVoiceForASRTraining is not in the standard transformers package; it may require
    # a dev install or a separate package. We raise so the caller can fall back to Whisper.
    from transformers import VibeVoiceForASRTraining, AutoProcessor  # type: ignore
    import torch  # type: ignore

    # from_pretrained uses cache by default; next runs load from cache
    model = VibeVoiceForASRTraining.from_pretrained(
        VIBEVOICE_MODEL_ID,
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(VIBEVOICE_MODEL_ID)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    _vibevoice_model = model
    _vibevoice_processor = processor
    return _vibevoice_model, _vibevoice_processor


def transcribe_audio_vibevoice(audio_path: str) -> str:
    """
    Transcribe an audio file using microsoft/VibeVoice-ASR (Hugging Face) when available.
    Uses the model from cache after the first download.
    Returns an error string (starting with "Error:") if VibeVoice is not installed or fails;
    the endpoint can then fall back to Groq Whisper.
    """
    import os
    try:
        import librosa  # type: ignore
        import torch  # type: ignore
    except ImportError as e:
        return f"Error: install dependencies (e.g. pip install torch librosa). {e}"

    if not os.path.isfile(audio_path):
        return f"Error: file not found: {audio_path}"

    try:
        model, processor = _get_vibevoice_model_and_processor()
    except (ImportError, RuntimeError) as e:
        # VibeVoiceForASRTraining not in transformers; caller should use Whisper fallback
        return f"Error: VibeVoice not available: {e}"

    try:
        # Load audio; VibeVoice typically expects 16 kHz
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        inputs = processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs)

        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = (transcription[0] if transcription else "").strip()
        return text or "Error: empty transcription from VibeVoice-ASR."
    except Exception as e:
        return f"Error transcription failed: {str(e)}"


# -----------------------------------------------------------------------------
# Whisper Large V3 (Hugging Face): lazy-loaded for meeting transcription.
# Model is downloaded once and then loaded from cache (~/.cache/huggingface/hub).
# -----------------------------------------------------------------------------
_whisper_processor = None
_whisper_model = None
WHISPER_MODEL_ID = "openai/whisper-large-v3"


def _get_whisper_model_and_processor():
    """Load Whisper Large V3 model and processor once; reuse on subsequent calls."""
    global _whisper_processor, _whisper_model
    if _whisper_processor is not None and _whisper_model is not None:
        return _whisper_processor, _whisper_model

    from transformers import AutoProcessor, WhisperForConditionalGeneration  # type: ignore
    import torch  # type: ignore

    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype="auto",
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    _whisper_processor = processor
    _whisper_model = model
    return _whisper_processor, _whisper_model


def transcribe_audio_whisper_local(audio_path: str, language: str = "en") -> tuple[str, list | None]:
    """
    Transcribe an audio file using local Whisper Large V3 (Hugging Face).

    Returns an error string (starting with "Error:") on failure.
    Uses librosa for loading (avoids torchaudio/torchcodec/FFmpeg on Windows).
    """
    import os
    if not os.path.isfile(audio_path):
        return f"Error: file not found: {audio_path}"

    try:
        import torch  # type: ignore
        import librosa  # type: ignore
    except ImportError as e:
        return f"Error: install dependencies (e.g. pip install torch librosa). {e}"

    try:
        processor, model = _get_whisper_model_and_processor()

        # Use librosa to avoid torchaudio/torchcodec/FFmpeg dependency on Windows
        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features
        # Match model dtype (float16 on GPU, float32 on CPU) to avoid "Input type (float) and bias type (Half)" error
        input_features = input_features.to(device=model.device, dtype=model.dtype)

        gen_kwargs = {"task": "transcribe"}
        if language and language != "auto":
            gen_kwargs["language"] = language
        with torch.no_grad():
            generated_ids = model.generate(input_features, **gen_kwargs)

        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        text = (transcription[0] if transcription else "").strip()
        # Return tuple to match expected signature (text, segments)
        # Segments are not easily available with raw model.generate without pipeline, so returning None for now.
        return (text or "Error: empty transcription from Whisper Large V3."), None
    except Exception as e:
        return f"Error transcription failed: {str(e)}", None


def call_groq_ai(prompt: str, model_name: str = None) -> str:
    if not groq_client:
        return "Error: missing dependency 'groq'. Please install it."
    if not groq_client.api_key:
        return "Error: GROQ_API_KEY not set."
    
    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name or settings.AI_MODEL,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def transcribe_audio(file_buffer) -> str:
    """
    Transcribe audio file using Groq Whisper.
    file_buffer: file-like object with .name attribute (needed by Groq client)
    """
    if not groq_client:
        return "Error: missing dependency 'groq'. Please install it."
    if not groq_client.api_key:
        return "Error: GROQ_API_KEY not set."

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=file_buffer,
            model="whisper-large-v3",
            response_format="json",
            language="en",
            temperature=0.0
        )
        return transcription.text
    except Exception as e:
        return f"Error transcription failed: {str(e)}"


def transcribe_audio_with_timestamps(file_buffer) -> tuple[str, list]:
    """
    Transcribe audio using Groq Whisper and return text + segments with timestamps.
    """
    if not groq_client:
        return "Error: missing dependency 'groq'.", []
    if not groq_client.api_key:
        return "Error: GROQ_API_KEY not set.", []

    try:
        # We use verbose_json to get segments/timestamps
        transcription = groq_client.audio.transcriptions.create(
            file=file_buffer,
            model="whisper-large-v3",
            response_format="verbose_json",
            language="en",
            temperature=0.0
        )
        # transcription is an object with .text and .segments (list of dicts/objects)
        return transcription.text, transcription.segments
    except Exception as e:
        return f"Error transcription failed: {str(e)}", []


def format_meeting_transcription(transcript: str, model_name: str | None = None, segments: list | None = None) -> str:
    """
    Format a raw transcription into a meeting-style transcript with speakers.

    This uses a Groq chat model to infer speaker turns from plain text and
    return a readable transcript using real speaker names/titles where possible.
    """
    if not transcript.strip():
        return "Error: empty transcription text."

    prompt = f"""
You are an expert meeting transcription formatter.

You are given a raw transcription of a multi-speaker meeting with little or no
speaker labeling. Your job is to rewrite it into a clean, readable transcript
with inferred speaker turns AND human-friendly speaker names.

FORMATTING RULES (you MUST follow these exactly):
- Put EVERY speaker turn on its OWN line. Start each new speaker on a new line.
- Use exactly this format per line: "SpeakerName: what they said"
- Add a BLANK LINE between different speakers so the transcript is easy to read.
- When the transcript clearly indicates a person's name, title, or role
  (e.g. "Chairman", "Councillor Boyce", "Ms Lewis"), use that as the speaker
  label. Only use "Speaker 1", "Speaker 2" if you cannot infer any name or role.
- Group consecutive sentences by the SAME speaker on one line; when the speaker
  changes, start a NEW line with the new speaker's name and a blank line above it.
- Preserve the original meaning and all important detail. Do not invent content.
- Output ONLY the transcript: no intro, no explanation, no markdown.

Example output format:

Chairman: Thank you. Good evening, Councillor Ms Lewis.

Councillor Lewis: Good evening, I'm sorry I had an emergency. We're just about to start the officer's report.

Chairman: Understood. Let's proceed.

Raw transcription:
\"\"\"{transcript}\"\"\"
"""

    return call_groq_ai(prompt, model_name=model_name)
