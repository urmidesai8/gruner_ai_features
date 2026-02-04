import json
import uuid
from typing import List, Dict, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.services.summarizer import generate_chat_summary, generate_text_summary, generate_memory_extraction, groq_client


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

        client.create_payload_index(
            collection_name=name,
            field_name="participant_ids",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=name,
            field_name="chat_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        if name == settings.QDRANT_GROUP_COLLECTION:
            client.create_payload_index(
                collection_name=name,
                field_name="group_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )


# ---- Embedding function (FastEmbed) ----

from fastembed import TextEmbedding

_embedding_model = None

def get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        # Uses "BAAI/bge-small-en-v1.5" by default, which is efficient
        _embedding_model = TextEmbedding()
    return _embedding_model

def embed_text(text: str) -> List[float]:
    """
    Generate semantic embeddings using FastEmbed.
    """
    if not text:
        return [0.0] * VECTOR_SIZE

    model = get_embedding_model()
    # model.embed returns a generator of numpy arrays, we take the first one and convert to list
    embedding = list(model.embed([text]))[0]
    return embedding.tolist()


# ---- Memory extraction from chat messages ----

def _build_memories_from_messages(
    messages: List[dict],
    chat_type: str,
    model: Optional[str] = None,
) -> List[Dict]:
    """
    Use the existing chat summarizer to derive important memories from messages.
    
    Refactored to create a SINGLE comprehensive memory item for the entire conversation.
    This ensures we only have one vector point per chat.
    """
    if not messages:
        return []

    # Get generalized summary (still good for 'Overview' section)
    summary_data = generate_chat_summary(
        messages=messages,
        username=None,
        total_messages=200, 
        model=model,
    )
    
    # Get specialized extraction
    extraction_data = generate_memory_extraction(messages, chat_type, model=model)

    # Basic time range approximation
    first_ts = messages[0].get("timestamp")
    last_ts = messages[-1].get("timestamp")
    
    lines = []
    
    # Only include Overview if we didn't get specialized data, OR if the client specifically wants it
    # User requested: "The LLM should wisely classify... If any such categories not present that don't include it."
    # We will prioritize the specific categories. Use summary ONLY if extraction failed or is empty.
    
    has_specialized_content = False

    if chat_type == "individual":
        # Individual: Decisions, Tasks, Facts
        decisions = extraction_data.get("decisions", [])
        if decisions:
            lines.append("[Decisions]")
            for d in decisions:
                lines.append(f"- {d}")
            has_specialized_content = True
        
        tasks = extraction_data.get("tasks", [])
        if tasks:
            lines.append("\n[Tasks & Deadlines]")
            for t in tasks:
                lines.append(f"- {t}")
            has_specialized_content = True

        facts = extraction_data.get("facts", [])
        if facts:
            lines.append("\n[Important Facts]")
            for f in facts:
                lines.append(f"- {f}")
            has_specialized_content = True

    elif chat_type == "group":
        # Group: Decisions, FAQs
        decisions = extraction_data.get("decisions", [])
        if decisions:
            lines.append("[Decisions]")
            for d in decisions:
                lines.append(f"- {d}")
            has_specialized_content = True
        
        faqs = extraction_data.get("faqs", [])
        if faqs:
            lines.append("\n[FAQs]")
            for faq in faqs:
                q = faq.get("question", "Q")
                a = faq.get("answer", "A")
                lines.append(f"Q: {q}")
                lines.append(f"A: {a}")
                lines.append("") # spacer
            has_specialized_content = True

    if not has_specialized_content:
        # Fallback to generic overview if nothing else was found
        summary_text = summary_data.get("summary", "")
        if summary_text:
            lines.append(f"[Overview]\n{summary_text}")
            
    full_text = "\n".join(lines).strip()
    
    if not full_text:
        return []

    return [
        {
            "memory_type": "conversation_summary",
            "summary_text": full_text,
            "time_range": {"from": first_ts, "to": last_ts},
            "tags": ["summary", "comprehensive", chat_type],
            "confidence": 1.0,
        }
    ]


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

    # Map old names to current names in history using user_id
    current_names = {user1_id: user1_name, user2_id: user2_name}
    mapped_messages = []
    for msg in messages:
        m = msg.copy()
        uid = m.get("user_id")
        if uid in current_names:
            m["sender"] = current_names[uid]
        mapped_messages.append(m)

    memories = _build_memories_from_messages(mapped_messages, chat_type="individual", model=model)
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
            "participant_ids": [user1_id, user2_id],
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

        # Deterministic ID based on chat_id AND names to ensure new vectors for name changes
        # We use UUID v5 with DNS namespace + unique string
        id_seed = f"{chat_id}:{user1_name}:{user2_name}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_seed))

        points.append(
            qmodels.PointStruct(
                id=point_id,
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

    participants: list of dicts with keys {"user_id", "name"}
    Returns number of memories upserted.
    """
    ensure_collections_exist()
    client = _get_qdrant_client()

    if not messages:
        return 0

    chat_id = f"group:{group_id}"

    # Map old names to current names using user_id
    name_map = {p["user_id"]: p["name"] for p in participants}
    mapped_messages = []
    for msg in messages:
        m = msg.copy()
        uid = m.get("user_id")
        if uid in name_map:
            m["sender"] = name_map[uid]
        mapped_messages.append(m)

    memories = _build_memories_from_messages(mapped_messages, chat_type="group", model=model)
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
            "participant_ids": [p["user_id"] for p in participants],
            "participants": participants,
            "memory_type": mem["memory_type"],
            "summary_text": mem["summary_text"],
            "time_range": mem.get("time_range"),
            "tags": mem.get("tags", []),
            "confidence": mem.get("confidence", 0.0),
        }

        # Deterministic ID based on chat_id AND names
        id_seed = f"{chat_id}:{group_name}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_seed))

        points.append(
            qmodels.PointStruct(
                id=point_id,
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
                key="participant_ids",
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

