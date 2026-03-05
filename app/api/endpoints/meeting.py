"""
Meeting recording summary, task extraction, Q&A, agenda and intelligence endpoints.
"""

from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...services.summarizer import generate_text_summary
from ...services.meeting_task_service import extract_meeting_tasks
from ...services.meeting_transcription_service import ask_meeting_question
from ...services.meeting_agenda_service import analyze_agenda_vs_discussion
from ...services.meeting_intelligence_service import generate_meeting_intelligence


class MeetingRecordingSummaryRequest(BaseModel):
    transcription: str
    model: Optional[str] = None


class MeetingTasksRequest(BaseModel):
    text: str
    model: Optional[str] = None


class MeetingAskRequest(BaseModel):
    query: str
    participant_id: str
    limit: Optional[int] = 3
    score_threshold: Optional[float] = 0.3
    model: Optional[str] = None


class AgendaItem(BaseModel):
    title: str
    planned_minutes: int


class AgendaIntelligenceRequest(BaseModel):
    transcript: str
    agenda_items: List[AgendaItem]
    actual_meeting_minutes: int
    model: Optional[str] = None


class MeetingIntelligenceRequest(BaseModel):
    transcript: str
    segments: Optional[List[Dict]] = None
    model: Optional[str] = None


router = APIRouter()


@router.post("/meeting-recording/summary")
async def meeting_recording_summary(request: MeetingRecordingSummaryRequest) -> JSONResponse:
    """Post-process a full meeting transcription into a structured summary."""
    if not request.transcription.strip():
        raise HTTPException(status_code=400, detail="Transcription text is required.")
    try:
        summary = generate_text_summary(
            request.transcription,
            model=request.model,
        )
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meeting recording summary failed: {str(e)}") from e


@router.post("/meeting-tasks")
async def meeting_tasks(request: MeetingTasksRequest) -> JSONResponse:
    """Extract tasks from a meeting transcription using spaCy NER + rule-based detection."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Transcription text is required.")
    try:
        spacy_model = request.model or "en_core_web_sm"
        result = extract_meeting_tasks(request.text, model_name=spacy_model)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meeting task extraction failed: {str(e)}") from e


@router.post("/meeting_ask")
async def meeting_ask(request: MeetingAskRequest) -> JSONResponse:
    """Answer questions based on meeting transcriptions."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required.")
    if not request.participant_id or not request.participant_id.strip():
        raise HTTPException(status_code=400, detail="Participant ID is required.")
    try:
        result = ask_meeting_question(
            query=request.query,
            participant_id=request.participant_id,
            limit=request.limit or 3,
            score_threshold=request.score_threshold or 0.3,
            model=request.model,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meeting Q&A failed: {str(e)}") from e


@router.post("/meeting/agenda-intelligence")
async def meeting_agenda_intelligence(request: AgendaIntelligenceRequest) -> JSONResponse:
    """Agenda vs Discussion Intelligence: compare planned agenda vs actual discussion."""
    if not request.transcript or not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcription text is required.")
    if not request.agenda_items:
        raise HTTPException(status_code=400, detail="At least one agenda item is required.")
    if request.actual_meeting_minutes <= 0:
        raise HTTPException(status_code=400, detail="Actual meeting minutes must be greater than 0.")
    try:
        agenda_dicts = [{"title": item.title, "planned_minutes": item.planned_minutes} for item in request.agenda_items]
        result = analyze_agenda_vs_discussion(
            transcript=request.transcript,
            agenda_items=agenda_dicts,
            actual_meeting_minutes=request.actual_meeting_minutes,
            model=request.model,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agenda intelligence failed: {str(e)}") from e


@router.post("/meeting/intelligence-insights")
async def meeting_intelligence_insights(request: MeetingIntelligenceRequest) -> JSONResponse:
    """Meeting Intelligence & Insights: higher-level analytics."""
    if not request.transcript or not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcription text is required.")
    try:
        result = generate_meeting_intelligence(
            transcript=request.transcript,
            segments=request.segments,
            model=request.model,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meeting intelligence failed: {str(e)}") from e
