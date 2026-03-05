"""
AI memory refresh and search endpoints (individual and group).
"""

from typing import List, Dict, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...models.schemas import chat_history
from ...services.memory_service import (
    upsert_individual_chat_memories,
    upsert_group_chat_memories,
    search_individual_memories,
    search_group_memories,
)


class IndividualMemoryRefreshRequest(BaseModel):
    user1_id: str
    user1_name: str
    user2_id: str
    user2_name: str
    model: Optional[str] = None


class GroupMemoryRefreshRequest(BaseModel):
    group_id: str
    group_name: str
    participants: List[Dict[str, str]]
    model: Optional[str] = None


class IndividualMemorySearchRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 10


class GroupMemorySearchRequest(BaseModel):
    group_id: str
    query: str
    limit: int = 10


router = APIRouter()


@router.post("/ai-memory/refresh-individual")
async def refresh_individual_memory(request: IndividualMemoryRefreshRequest) -> JSONResponse:
    """Refresh AI memories for a 1:1 chat and upsert into Qdrant."""
    messages = chat_history.get_ai_enabled_messages()
    count = upsert_individual_chat_memories(
        user1_id=request.user1_id,
        user1_name=request.user1_name,
        user2_id=request.user2_id,
        user2_name=request.user2_name,
        messages=messages,
        model=request.model,
    )
    return JSONResponse(content={
        "status": "ok",
        "memories_upserted": count,
        "collection": "individual_chats",
    })


@router.post("/ai-memory/refresh-group")
async def refresh_group_memory(request: GroupMemoryRefreshRequest) -> JSONResponse:
    """Refresh AI memories for a group chat and upsert into Qdrant."""
    messages = chat_history.get_ai_enabled_messages()
    count = upsert_group_chat_memories(
        group_id=request.group_id,
        group_name=request.group_name,
        participants=request.participants,
        messages=messages,
        model=request.model,
    )
    return JSONResponse(content={
        "status": "ok",
        "memories_upserted": count,
        "collection": "group_chats",
    })


@router.post("/ai-memory/search-individual")
async def search_individual_memory(request: IndividualMemorySearchRequest) -> JSONResponse:
    """Semantic search over individual chat memories for a given user."""
    results = search_individual_memories(
        user_id=request.user_id,
        query=request.query,
        limit=request.limit,
    )
    return JSONResponse(content={"results": results})


@router.post("/ai-memory/search-group")
async def search_group_memory(request: GroupMemorySearchRequest) -> JSONResponse:
    """Semantic search over group chat memories for a given group_id."""
    results = search_group_memories(
        group_id=request.group_id,
        query=request.query,
        limit=request.limit,
    )
    return JSONResponse(content={"results": results})
