import json
from typing import List, Dict, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.services.summarizer import generate_chat_summary, groq_client


# ---- Qdrant client & configuration ----

VECTOR_SIZE = 384  # dimension of embedding vectors (simple hash-based embedding for POC)
DISTANCE = qmodels.Distance.COSINE


def _get_qdrant_client() -> QdrantClient:
    """
    Create a Qdrant client using environment variables.

    NOTE: Do NOT hard-code credentials in code. Set:
    - QDRANT_URL
    - QDRANT_API_KEY
    in your environment or .env file.
    """
    if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set in environment for AI Memory feature."
        )

    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def ensure_collections_exist() -> None:
    """
    Ensure that the two main collections exist:
    - individual_chats
    - group_chats
    """
    client = _get_qdrant_client()

    for name in [
        settings.QDRANT_INDIVIDUAL_COLLECTION,
        settings.QDRANT_GROUP_COLLECTION,
    ]:
        if not client.collection_exists(name):
            client.recreate_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_SIZE,
                    distance=DISTANCE,
                ),
            )


# ---- Simple embedding function (POC) ----

def embed_text(text: str) -> List[float]:
    """
    Very simple deterministic embedding based on hashing.

    This is a POC implementation so you can wire Qdrant end-to-end without
    introducing a heavy embedding model dependency.

    For production, replace this with a real embedding model, e.g.:
    - OpenAI embeddings
    - sentence-transformers
    - HuggingFace embeddings
    """
    import hashlib
    import math

    if not text:
        return [0.0] * VECTOR_SIZE

    # Create a hash and spread it over VECTOR_SIZE positions
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat hash bytes to fill VECTOR_SIZE
    vals: List[float] = []
    while len(vals) < VECTOR_SIZE:
        for b in h:
            vals.append(float(b))
            if len(vals) >= VECTOR_SIZE:
                break
    # Normalize vector
    norm = math.sqrt(sum(v * v for v in vals))
    if norm == 0:
        return [0.0] * VECTOR_SIZE
    return [v / norm for v in vals]


# ---- Memory extraction from chat messages ----

def _build_memories_from_messages(
    messages: List[dict],
    model: Optional[str] = None,
) -> List[Dict]:
    """
    Use the existing chat summarizer to derive important memories from messages.

    We treat:
    - bullet_points -> type 'summary_point'
    - key_decisions -> type 'decision'
    - action_items  -> type 'action_item'

    Each becomes a separate memory item with summary_text and metadata.
    """
    if not messages:
        return []

    summary = generate_chat_summary(
        messages=messages,
        username=None,
        total_messages=200,  # limit for memory extraction
        model=model,
    )

    memories: List[Dict] = []
    # Basic time range approximation: from first to last message
    first_ts = messages[0].get("timestamp")
    last_ts = messages[-1].get("timestamp")

    def _add_memory(memory_type: str, text: str) -> None:
        if not text:
            return
        memories.append(
            {
                "memory_type": memory_type,
                "summary_text": text,
                "time_range": {"from": first_ts, "to": last_ts},
                "tags": [],  # can be enriched later
                "confidence": 0.9,  # heuristic for now
            }
        )

    for bp in summary.get("bullet_points", []):
        _add_memory("summary_point", bp)

    for dec in summary.get("key_decisions", []):
        _add_memory("decision", dec)

    for ai in summary.get("action_items", []):
        _add_memory("action_item", ai)

    return memories


# ---- Upsert helpers for individual & group chats ----

def upsert_individual_chat_memories(
    user1_id: str,
    user1_name: str,
    user2_id: str,
    user2_name: str,
    messages: List[dict],
    model: Optional[str] = None,
) -> int:
    """
    Build and upsert memory vectors for a 1:1 chat into the individual_chats collection.

    Returns number of memories upserted.
    """
    ensure_collections_exist()
    client = _get_qdrant_client()

    if not messages:
        return 0

    # Derive a stable chat_id based on user IDs
    sorted_ids = sorted([user1_id, user2_id])
    chat_id = f"individual:{sorted_ids[0]}:{sorted_ids[1]}"

    memories = _build_memories_from_messages(messages, model=model)
    if not memories:
        return 0

    points: List[qmodels.PointStruct] = []
    for mem in memories:
        vector = embed_text(mem["summary_text"])
        payload = {
            "chat_id": chat_id,
            "chat_type": "individual",
            "sender_user_id": user1_id,
            "sender_name": user1_name,
            "receiver_user_id": user2_id,
            "receiver_name": user2_name,
            "participants": [
                {"user_id": user1_id, "name": user1_name},
                {"user_id": user2_id, "name": user2_name},
            ],
            "memory_type": mem["memory_type"],
            "summary_text": mem["summary_text"],
            "time_range": mem.get("time_range"),
            "tags": mem.get("tags", []),
            "confidence": mem.get("confidence", 0.0),
        }

        points.append(
            qmodels.PointStruct(
                id=None,
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(
        collection_name=settings.QDRANT_INDIVIDUAL_COLLECTION,
        points=points,
    )
    return len(points)


def upsert_group_chat_memories(
    group_id: str,
    group_name: str,
    participants: List[Dict[str, str]],
    messages: List[dict],
    model: Optional[str] = None,
) -> int:
    """
    Build and upsert memory vectors for a group chat into the group_chats collection.

    participants: list of dicts with keys {\"user_id\", \"name\"}
    Returns number of memories upserted.
    """
    ensure_collections_exist()
    client = _get_qdrant_client()

    if not messages:
        return 0

    chat_id = f"group:{group_id}"

    memories = _build_memories_from_messages(messages, model=model)
    if not memories:
        return 0

    points: List[qmodels.PointStruct] = []

    for mem in memories:
        vector = embed_text(mem["summary_text"])
        payload = {
            "group_id": group_id,
            "group_name": group_name,
            "chat_id": chat_id,
            "chat_type": "group",
            "participants": participants,
            "memory_type": mem["memory_type"],
            "summary_text": mem["summary_text"],
            "time_range": mem.get("time_range"),
            "tags": mem.get("tags", []),
            "confidence": mem.get("confidence", 0.0),
        }

        points.append(
            qmodels.PointStruct(
                id=None,
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(
        collection_name=settings.QDRANT_GROUP_COLLECTION,
        points=points,
    )
    return len(points)


# ---- Query helpers ----

def search_individual_memories(
    user_id: str,
    query: str,
    limit: int = 10,
) -> List[Dict]:
    """
    Search important memories in individual chats for a given user.
    """
    ensure_collections_exist()
    client = _get_qdrant_client()

    query_vector = embed_text(query)

    # Search memories where the user is one of the participants
    filter_ = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="participants.user_id",
                match=qmodels.MatchAny(any=[user_id]),
            )
        ]
    )

    results = client.search(
        collection_name=settings.QDRANT_INDIVIDUAL_COLLECTION,
        query_vector=query_vector,
        limit=limit,
        query_filter=filter_,
    )

    return [hit.payload for hit in results]


def search_group_memories(
    group_id: str,
    query: str,
    limit: int = 10,
) -> List[Dict]:
    """
    Search important memories in group chats for a given group_id.
    """
    ensure_collections_exist()
    client = _get_qdrant_client()

    query_vector = embed_text(query)

    filter_ = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="group_id",
                match=qmodels.MatchValue(value=group_id),
            )
        ]
    )

    results = client.search(
        collection_name=settings.QDRANT_GROUP_COLLECTION,
        query_vector=query_vector,
        limit=limit,
        query_filter=filter_,
    )

    return [hit.payload for hit in results]

