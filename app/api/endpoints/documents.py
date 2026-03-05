"""
Document upload, text extraction, and question-answering endpoints.
"""

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...services.ai_service import call_groq_ai
from ...services.summarizer import generate_text_summary
from ...services.document_extraction_service import extract_document_text_and_tables
from ...services.document_extraction_qdrant_service import (
    store_document_extraction,
    get_document_extraction,
    get_document_extraction_by_original_name_and_user,
)


UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_DOC_EXTENSIONS = {".pdf", ".docx"}


class DocumentTextExtractionRequest(BaseModel):
    doc_id: str
    model: Optional[str] = None
    uploaded_user_id: Optional[str] = None
    doc_upload_time: Optional[str] = None
    original_name: Optional[str] = None


class DocumentQARequest(BaseModel):
    doc_id: str
    uploaded_user_id: str
    question: str
    model: Optional[str] = None


router = APIRouter()


def _run_document_extraction_pipeline(
    file_path: Path,
    doc_id: str,
    uploaded_user_id: Optional[str],
    doc_upload_time: Optional[str],
    model: Optional[str],
    original_name: Optional[str] = None,
) -> dict:
    """Run full document extraction: Docling -> LLM format -> summarize -> store in Qdrant."""
    if original_name and uploaded_user_id:
        existing = get_document_extraction_by_original_name_and_user(
            original_name, uploaded_user_id
        )
        if existing:
            formatted_document_text = (existing.get("formatted_document_text") or "").strip()
            document_summary = (existing.get("document_summary") or "").strip() or "Summary generated."
            return {
                "extraction_id": str(uuid.uuid4()),
                "formatted_document_text": formatted_document_text,
                "document_summary": document_summary,
            }
    clean_text, tables = extract_document_text_and_tables(file_path)
    format_prompt = f"""You are an expert document reconstruction assistant.
You are given:
- CLEAN_TEXT: linear text extracted from a document
- TABLES_JSON: structured table data (list of tables, with headers and rows)
Reconstruct a clean, well-formatted Markdown version of the document.
Requirements:
- Preserve sections and headings (use Markdown headings like #, ##, etc.).
- Preserve paragraph structure and lists (bullet/numbered).
- Recreate tables as Markdown tables using headers and rows from TABLES_JSON.
- Where the original document likely contained images or figures, infer and insert short placeholders like "[Image: description]" if mentioned in the text.
- Do NOT output JSON or explanations. Return ONLY the formatted document text.

CLEAN_TEXT:
\"\"\"{clean_text}\"\"\"

TABLES_JSON:
\"\"\"{json.dumps(tables, ensure_ascii=False)}\"\"\"
"""
    formatted_document_text = call_groq_ai(format_prompt, model_name=model)
    if not formatted_document_text or formatted_document_text.startswith("Error"):
        formatted_document_text = clean_text
    summary_result = generate_text_summary(formatted_document_text, model=model)
    document_summary = summary_result.get("summary", "Summary generated.")
    extraction_id = str(uuid.uuid4())
    try:
        store_document_extraction(
            doc_id=doc_id,
            document_summary=document_summary,
            uploaded_user_id=uploaded_user_id,
            doc_upload_time=doc_upload_time,
            formatted_document_text=formatted_document_text,
            original_name=original_name,
        )
    except Exception as e:
        print(f"Warning: Failed to store document extraction in Qdrant: {e}")
    return {
        "extraction_id": extraction_id,
        "formatted_document_text": formatted_document_text,
        "document_summary": document_summary,
    }


@router.post("/upload-document")
async def upload_document_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    uploaded_user_id: Optional[str] = Form(None),
) -> JSONResponse:
    """Accept a document file (pdf or docx), save to static/uploads, return immediately. Extraction runs in background."""
    ext = (os.path.splitext(file.filename or "")[1] or "").lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only .pdf and .docx are allowed. Got: {ext or 'no extension'}",
        )
    try:
        doc_upload_time = datetime.now(timezone.utc).isoformat()
        original_name = file.filename or ""
        unique_filename = f"{uuid.uuid4()}{ext}"
        file_path = UPLOAD_DIR / unique_filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        background_tasks.add_task(
            _run_document_extraction_pipeline,
            file_path,
            unique_filename,
            uploaded_user_id,
            doc_upload_time,
            None,
            original_name,
        )
        return JSONResponse(content={
            "doc_id": unique_filename,
            "url": f"/static/uploads/{unique_filename}",
            "filename": unique_filename,
            "original_name": original_name or unique_filename,
            "uploaded_user_id": uploaded_user_id,
            "doc_upload_time": doc_upload_time,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e


@router.post("/document-text-extraction")
async def document_text_extraction(request: DocumentTextExtractionRequest) -> JSONResponse:
    """Extract and format text from an uploaded document, then summarize and store in Qdrant."""
    if not request.doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required.")
    file_path = UPLOAD_DIR / request.doc_id
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        result = _run_document_extraction_pipeline(
            file_path=file_path,
            doc_id=request.doc_id,
            uploaded_user_id=request.uploaded_user_id,
            doc_upload_time=request.doc_upload_time,
            model=request.model,
            original_name=request.original_name,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document extraction failed: {str(e)}") from e


@router.post("/document-qa")
async def document_qa(request: DocumentQARequest) -> JSONResponse:
    """Answer a question based on a specific document's extracted text in Qdrant."""
    if not request.doc_id or not request.uploaded_user_id:
        raise HTTPException(status_code=400, detail="doc_id and uploaded_user_id are required.")
    payload = get_document_extraction(
        doc_id=request.doc_id,
        uploaded_user_id=request.uploaded_user_id,
    )
    if not payload:
        raise HTTPException(status_code=404, detail="No extraction found for this document and user.")
    context_text = (
        (payload.get("formatted_document_text") or "").strip()
        or (payload.get("document_summary") or "").strip()
    )
    if not context_text:
        raise HTTPException(status_code=400, detail="Document extraction text is empty.")
    qa_prompt = f"""You are a helpful assistant answering questions based ONLY on the document below.

DOCUMENT:
\"\"\"{context_text}\"\"\"

Question: {request.question}

If the answer is clearly present, answer concisely.
If the answer is not present, say you cannot answer based on this document.
"""
    answer = call_groq_ai(qa_prompt, model_name=request.model)
    return JSONResponse(content={
        "doc_id": request.doc_id,
        "uploaded_user_id": request.uploaded_user_id,
        "question": request.question,
        "answer": answer,
    })
