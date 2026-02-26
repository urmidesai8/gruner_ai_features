from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.cache_service import cache_service

router = APIRouter()

class CacheQuery(BaseModel):
    query: str

class CacheSet(BaseModel):
    query: str
    response: str

@router.post("/cache/exact")
async def check_exact_cache(payload: CacheQuery):
    """
    Layer 1: Check Exact Match Cache (Memcached).
    """
    result = cache_service.get_exact_match(payload.query)
    if result:
        return {"hit": True, "source": "memcached", "response": result}
    return {"hit": False, "source": "memcached", "response": None}

@router.post("/cache/semantic")
async def check_semantic_cache(payload: CacheQuery):
    """
    Layer 2: Check Semantic Match Cache (FAISS).
    """
    result, score = cache_service.get_semantic_match(payload.query)
    if result:
        # User defined threshold in service is 0.90, but we return score for visibility
        return {"hit": True, "source": "faiss", "score": score, "response": result}
    return {"hit": False, "source": "faiss", "score": score, "response": None}

@router.post("/cache/learn")
async def learn_response(payload: CacheSet):
    """
    Manually teach the cache a new Query -> Response pair.
    Updates both Memcached and FAISS.
    """
    cache_service.set_exact_match(payload.query, payload.response)
    cache_service.set_semantic_match(payload.query, payload.response)
    return {"status": "learned", "message": "Added to both Exact and Semantic caches."}

@router.delete("/cache")
async def clear_cache():
    """
    Clear all cached data (Memcached + FAISS).
    Useful for resetting 'bad' learning.
    """
    # 1. Clear FAISS
    cache_service.index.reset()
    cache_service.faiss_responses = []
    
    # 2. Clear Memcached (if available)
    if cache_service.mc:
        cache_service.mc.flush_all()
        
    return {"status": "cleared", "message": "All cache layers have been reset."}
