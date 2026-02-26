"""
Store document extraction metadata in Qdrant.

Creates and uses the "document_extraction" collection. Payload fields:
- doc_id, uploaded_user_id, doc_upload_time, document_summary, formatted_document_text
"""
import uuid
from typing import Optional

from qdrant_client.http import models as qmodels

from app.services.memory_service import (
    _get_qdrant_client,
    embed_text,
    VECTOR_SIZE,
    DISTANCE,
)


DOCUMENT_EXTRACTION_COLLECTION = "document_extraction"


def ensure_document_extraction_collection() -> None:
    """Ensure the document_extraction collection exists in Qdrant."""
    try:
        client = _get_qdrant_client()
        if not client.collection_exists(DOCUMENT_EXTRACTION_COLLECTION):
            client.create_collection(
                collection_name=DOCUMENT_EXTRACTION_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_SIZE,
                    distance=DISTANCE,
                ),
            )
            print(f"Created Qdrant collection: {DOCUMENT_EXTRACTION_COLLECTION}")
    except Exception as e:
        print(f"Warning: Could not ensure document_extraction collection: {e}")


def store_document_extraction(
    doc_id: str,
    document_summary: str,
    uploaded_user_id: Optional[str] = None,
    doc_upload_time: Optional[str] = None,
    formatted_document_text: Optional[str] = None,
) -> str:
    """
    Store document extraction details in Qdrant.

    Args:
        doc_id: Document ID from /upload-document (filename in static/uploads).
        document_summary: Summary from /document-text-extraction.
        uploaded_user_id: Optional; from /upload-document.
        doc_upload_time: Optional; from /upload-document (e.g. ISO UTC).
        formatted_document_text: Optional; formatted full text from /document-text-extraction.

    Returns:
        The point ID (extraction record id) used in Qdrant.
    """
    ensure_document_extraction_collection()
    client = _get_qdrant_client()

    payload = {
        "doc_id": doc_id,
        "uploaded_user_id": uploaded_user_id or "",
        "doc_upload_time": doc_upload_time or "",
        "document_summary": document_summary or "",
        "formatted_document_text": formatted_document_text or "",
    }

    vector = embed_text(document_summary or "")
    point_id = str(uuid.uuid4())

    client.upsert(
        collection_name=DOCUMENT_EXTRACTION_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        ],
    )
    return point_id
