"""
Audio upload, transcription, and text summarization endpoints.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ...services.ai_service import (
    transcribe_audio,
    transcribe_audio_with_timestamps,
    transcribe_audio_whisper_local,
    transcribe_audio_vibevoice,
    transcribe_audio_seamless_m4t,
    format_meeting_transcription,
)
from ...services.summarizer import generate_text_summary
from ...services.meeting_transcription_service import store_meeting_transcription


UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class TextSummaryRequest(BaseModel):
    text: str
    model: Optional[str] = None


class AudioFileRequest(BaseModel):
    filename: str


class MeetingAudioFileRequest(BaseModel):
    filename: str
    model: Optional[str] = None
    asr_model: Optional[str] = "whisper-large-v3"
    participant_ids: Optional[List[str]] = None
    meeting_agenda: Optional[str] = None
    store_in_qdrant: Optional[bool] = True


router = APIRouter()


@router.post("/upload-audio")
async def upload_audio_file(file: UploadFile = File(...)) -> JSONResponse:
    """Upload an audio file to static/uploads and return URL and filename."""
    try:
        file_ext = os.path.splitext(file.filename or "")[1] or ".webm"
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return JSONResponse(content={
            "url": f"/static/uploads/{unique_filename}",
            "filename": unique_filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e


@router.post("/transcribe-file")
async def transcribe_saved_file(request: AudioFileRequest) -> JSONResponse:
    """Transcribe a previously uploaded audio file by filename."""
    try:
        file_path = UPLOAD_DIR / request.filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        with open(file_path, "rb") as audio_file:
            transcription_text = transcribe_audio((request.filename, audio_file))
        return JSONResponse(content={"transcription": transcription_text})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}") from e


@router.post("/transcribe-meeting-file")
async def transcribe_meeting_file(request: MeetingAudioFileRequest) -> JSONResponse:
    """Transcribe an uploaded meeting recording and format by speakers."""
    try:
        file_path = UPLOAD_DIR / request.filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        asr_model = (request.asr_model or "whisper-large-v3").strip()
        segments = []
        if asr_model in ("facebook/seamless-m4t-medium", "facebook/hf-seamless-m4t-medium"):
            raw_transcription = await run_in_threadpool(transcribe_audio_seamless_m4t, str(file_path))
            if raw_transcription.startswith("Error"):
                with open(file_path, "rb") as audio_file:
                    raw_transcription = transcribe_audio((request.filename, audio_file))
        elif asr_model == "microsoft/VibeVoice-ASR":
            raw_transcription = await run_in_threadpool(transcribe_audio_vibevoice, str(file_path))
            if raw_transcription.startswith("Error"):
                with open(file_path, "rb") as audio_file:
                    raw_transcription = transcribe_audio((request.filename, audio_file))
        elif asr_model in ("openai/whisper-small",):
            raw_transcription, segments = await run_in_threadpool(
                transcribe_audio_whisper_local, str(file_path), asr_model
            )
        else:
            with open(file_path, "rb") as audio_file:
                raw_transcription, segments = transcribe_audio_with_timestamps((request.filename, audio_file))
        if isinstance(raw_transcription, str) and raw_transcription.startswith("Error"):
            raise HTTPException(status_code=500, detail=raw_transcription)
        formatted_transcription = format_meeting_transcription(
            raw_transcription,
            model_name=request.model,
            segments=segments if segments else None,
        )
        if isinstance(formatted_transcription, str) and formatted_transcription.startswith("Error"):
            return JSONResponse(content={
                "transcription": raw_transcription,
                "formatted_transcription": None,
                "segments": segments if segments else None,
                "notice": "Meeting formatting failed; returning raw transcription only.",
            })
        response_data = {
            "transcription": raw_transcription,
            "formatted_transcription": formatted_transcription,
            "segments": segments if segments else None,
        }
        if request.store_in_qdrant and request.participant_ids:
            try:
                storage_result = store_meeting_transcription(
                    transcription=raw_transcription,
                    participant_ids=request.participant_ids,
                    meeting_agenda=request.meeting_agenda,
                )
                response_data["meeting_id"] = storage_result["meeting_id"]
                response_data["stored_in_qdrant"] = True
                response_data["meeting_time"] = storage_result["meeting_time"]
            except Exception as e:
                print(f"Warning: Failed to store transcription in Qdrant: {e}")
                response_data["stored_in_qdrant"] = False
                response_data["storage_error"] = str(e)
        else:
            response_data["stored_in_qdrant"] = False
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meeting transcription failed: {str(e)}") from e


@router.post("/transcribe")
async def transcribe_voice_note(file: UploadFile = File(...)) -> JSONResponse:
    """Transcribe an uploaded audio file (multipart)."""
    try:
        transcription_text = transcribe_audio((file.filename, file.file))
        if transcription_text.startswith("Error"):
            raise HTTPException(status_code=500, detail=transcription_text)
        return JSONResponse(content={"transcription": transcription_text})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}") from e


@router.post("/summarize-text")
async def summarize_text(request: TextSummaryRequest) -> JSONResponse:
    """Summarize raw text (e.g. from transcription)."""
    try:
        result = generate_text_summary(request.text, model=request.model)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}") from e
