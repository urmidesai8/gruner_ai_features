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
    """
    import os
    if not os.path.isfile(audio_path):
        return f"Error: file not found: {audio_path}"

    try:
        import torch  # type: ignore
        import torchaudio  # type: ignore
    except ImportError as e:
        return f"Error: install dependencies (e.g. pip install torchaudio). {e}"

    try:
        processor, model = _get_seamless_model_and_processor()

        waveform, sample_rate = torchaudio.load(audio_path)
        # convert to mono
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # resample to 16kHz (commonly expected)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        audio = waveform.squeeze(0).cpu().numpy()
        inputs = processor(audios=audio, sampling_rate=sample_rate, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}

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


def _seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as [HH:MM:SS]."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"






# -----------------------------------------------------------------------------
# Local Whisper (Transformers): lazy-loaded per model, cache reused.
# Supported: openai/whisper-large-v3, openai/whisper-small.
# -----------------------------------------------------------------------------
_whisper_pipelines = {}  # model_id -> pipeline


def _whisper_model_id(asr_model: str) -> str:
    """Map UI/request model name to Hugging Face model id."""
    if asr_model in ("whisper-large-v3", "openai/whisper-large-v3"):
        return "openai/whisper-large-v3"
    if asr_model in ("whisper-small", "openai/whisper-small"):
        return "openai/whisper-small"
    return asr_model


def _get_whisper_pipeline(model_id: str):
    """Load Whisper model and processor once per model_id; reuse from cache."""
    global _whisper_pipelines
    if model_id in _whisper_pipelines:
        return _whisper_pipelines[model_id]
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline  # type: ignore
    import torch  # type: ignore

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
    )
    _whisper_pipelines[model_id] = pipe
    return pipe


def transcribe_audio_whisper_local(audio_path: str, asr_model: str = "whisper-large-v3"):
    """
    Transcribe audio using local Whisper (Transformers). Load model once per model, then from cache.
    asr_model: e.g. whisper-large-v3, openai/whisper-small (UI/request value).
    Returns (text, segments) where segments is a list of {"start": float, "end": float, "text": str}.
    On error returns ("Error: ...", []).
    """
    import os
    if not os.path.isfile(audio_path):
        return f"Error: file not found: {audio_path}", []

    model_id = _whisper_model_id(asr_model)
    try:
        pipe = _get_whisper_pipeline(model_id)
        out = pipe(audio_path, return_timestamps=True)
        text = (out.get("text") or "").strip()
        segments = []
        for chunk in out.get("chunks") or []:
            ts = chunk.get("timestamp")
            seg_text = (chunk.get("text") or "").strip()
            if ts and seg_text:
                start, end = ts[0], ts[1]
                segments.append({"start": float(start), "end": float(end), "text": seg_text})
        if not segments and text:
            segments.append({"start": 0.0, "end": 0.0, "text": text})
        if not text and segments:
            text = " ".join(s["text"] for s in segments)
        return text, segments
    except Exception as e:
        return f"Error transcription failed: {str(e)}", []


def transcribe_audio_with_timestamps(file_buffer):
    """
    Transcribe audio with segment-level timestamps (Groq Whisper verbose_json).
    Returns (text, segments) where segments is a list of {"start": float, "end": float, "text": str}.
    On error returns ("Error: ...", []).
    """
    if not groq_client:
        return "Error: missing dependency 'groq'. Please install it.", []
    if not groq_client.api_key:
        return "Error: GROQ_API_KEY not set.", []

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=file_buffer,
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="en",
            temperature=0.0,
        )
        text = getattr(transcription, "text", None) or ""
        segments = []
        # Groq verbose_json may expose .segments (segment granularity) or .words (word granularity)
        if hasattr(transcription, "segments") and transcription.segments:
            for seg in transcription.segments:
                start = getattr(seg, "start", 0.0)
                end = getattr(seg, "end", 0.0)
                seg_text = getattr(seg, "text", "").strip()
                if seg_text:
                    segments.append({"start": float(start), "end": float(end), "text": seg_text})
        elif hasattr(transcription, "words") and transcription.words:
            # Coalesce words into segments by grouping consecutive words
            for w in transcription.words:
                start = getattr(w, "start", 0.0)
                end = getattr(w, "end", 0.0)
                word = getattr(w, "word", "").strip()
                if word:
                    segments.append({"start": float(start), "end": float(end), "text": word})
        if not segments and text:
            segments.append({"start": 0.0, "end": 0.0, "text": text})
        return text, segments
    except Exception as e:
        return f"Error transcription failed: {str(e)}", []


def format_meeting_transcription(transcript: str, model_name: str | None = None, segments: list | None = None) -> str:
    """
    Format a raw transcription into a meeting-style transcript with speakers.
    If segments (list of {"start", "end", "text"}) are provided, output includes
    timestamps in the form [HH:MM:SS] before each speaker line.
    """
    if not transcript.strip():
        return "Error: empty transcription text."

    if segments:
        # Build timestamped input for the LLM: [HH:MM:SS] text per segment
        ts_lines = []
        for s in segments:
            start = s.get("start", 0.0)
            text = s.get("text", "").strip()
            if text:
                ts_lines.append(f"{_seconds_to_timestamp(start)} {text}")
        timestamped_input = "\n".join(ts_lines)
        prompt = f"""
You are an expert meeting transcription formatter.

You are given a timestamped raw transcription of a multi-speaker meeting. Each line starts with [HH:MM:SS].
Your job is to infer speaker turns and produce a clean transcript WITH TIMESTAMPS and speaker names.

FORMATTING RULES (you MUST follow these exactly):
- Keep the exact [HH:MM:SS] timestamp at the START of each line (use the timestamp from the segment that starts that speaker turn).
- Use exactly this format per line: "[HH:MM:SS] SpeakerName: what they said"
- Add a BLANK LINE between different speakers.
- When the transcript clearly indicates a person's name, title, or role, use that as the speaker label. Otherwise use "Speaker 1", "Speaker 2".
- Group consecutive segments by the SAME speaker on one line; when the speaker changes, start a NEW line with the new speaker and a blank line above it. Use the timestamp of the first segment in that turn.
- Preserve the original meaning and all important detail. Do not invent content.
- Output ONLY the transcript: no intro, no explanation, no markdown.

Example output format:

[00:00:00] Chairman: Thank you. Good evening, Councillor Ms Lewis.

[00:00:08] Councillor Lewis: Good evening, I'm sorry I had an emergency. We're just about to start the officer's report.

[00:00:18] Chairman: Understood. Let's proceed.

Timestamped raw transcription:
\"\"\"{timestamped_input}\"\"\"
"""
    else:
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
