"""
Store document extraction metadata in Qdrant.

Creates and uses the "document_extraction" collection. Payload fields:
- doc_id, uploaded_user_id, doc_upload_time, document_summary, formatted_document_text
"""
import uuid
from typing import Optional, List

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
        created = False
        if not client.collection_exists(DOCUMENT_EXTRACTION_COLLECTION):
            client.create_collection(
                collection_name=DOCUMENT_EXTRACTION_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_SIZE,
                    distance=DISTANCE,
                ),
            )
            print(f"Created Qdrant collection: {DOCUMENT_EXTRACTION_COLLECTION}")
            created = True

        # Ensure payload indexes for filtering by doc_id and uploaded_user_id
        for field_name in ("doc_id", "uploaded_user_id"):
            try:
                client.create_payload_index(
                    collection_name=DOCUMENT_EXTRACTION_COLLECTION,
                    field_name=field_name,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
                if created:
                    print(f"Created payload index for '{field_name}' in {DOCUMENT_EXTRACTION_COLLECTION}")
            except Exception as idx_err:
                # Index may already exist; ignore that case
                if "already exists" not in str(idx_err).lower():
                    print(f"Note: Could not create index for {field_name}: {idx_err}")
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


def get_document_extraction(
    doc_id: str,
    uploaded_user_id: str,
) -> Optional[dict]:
    """
    Fetch a single document_extraction payload for a specific (doc_id, uploaded_user_id).
    Returns the payload dict or None if not found.
    """
    ensure_document_extraction_collection()
    client = _get_qdrant_client()

    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchValue(value=doc_id),
            ),
            qmodels.FieldCondition(
                key="uploaded_user_id",
                match=qmodels.MatchValue(value=uploaded_user_id),
            ),
        ]
    )

    points: List[qmodels.ScoredPoint]
    points, _ = client.scroll(
        collection_name=DOCUMENT_EXTRACTION_COLLECTION,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    return points[0].payload or None
