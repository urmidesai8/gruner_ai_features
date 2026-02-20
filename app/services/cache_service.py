from pymemcache.client import base
import faiss
import numpy as np
import hashlib
import json
import os
from sentence_transformers import SentenceTransformer

# Configuration
MEMCACHED_HOST = "localhost"
MEMCACHED_PORT = 11211
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

class CacheService:
    def __init__(self):
        # 1. Connect to Memcached (Layer 1)
        try:
            self.mc = base.Client((MEMCACHED_HOST, MEMCACHED_PORT))
            self.mc.set("ping", "pong") # Test connection
            print("Connected to Memcached successfully.")
            self.memcached_available = True
        except Exception as e:
            print(f"Warning: Could not connect to Memcached ({e}). Exact-match caching disabled.")
            self.memcached_available = False
            self.mc = None

        # 2. Key-Value Storage (Fallback for FAISS map if Memcached unavailable)
        self.local_kv_store = {}

        # 3. Initialize FAISS (Layer 2)
        # Using IndexFlatIP (Inner Product) for Cosine Similarity on normalized vectors
        self.index = faiss.IndexFlatIP(INDEX_DIMENSION)
        
        # We need a way to map FAISS ID -> Response Text. 
        # FAISS stores vectors and returns an integer ID (0, 1, 2...).
        # We'll use a simple list where index matches FAISS ID.
        self.faiss_responses = []

        # 4. Load Embedding Model
        print("Loading embedding model for Semantic Cache...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_ID)
        print("Cache Service initialized.")

    def _get_exact_key(self, query: str) -> str:
        """Hash the query to create a safe Memcached key."""
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def get_exact_match(self, query: str) -> str | None:
        """
        Layer 1: Check Exact Match Cache (Memcached).
        """
        if not self.memcached_available:
            return None

        key = self._get_exact_key(query)
        try:
            cached_response = self.mc.get(key)
            if cached_response:
                return cached_response.decode("utf-8")
        except Exception as e:
            print(f"Memcached get error: {e}")
        return None

    def set_exact_match(self, query: str, response: str, ttl: int = 3600):
        """
        Store response in Layer 1 Cache.
        """
        if self.memcached_available:
            key = self._get_exact_key(query)
            try:
                self.mc.set(key, response.encode("utf-8"), expire=ttl)
            except Exception as e:
                print(f"Memcached set error: {e}")

    def get_semantic_match(self, query: str, threshold: float = 0.70) -> tuple[str | None, float]:
        """
        Layer 2: Check Semantic Cache (FAISS).
        Returns (cached_response, score)
        """
        if self.index.ntotal == 0:
            return None, 0.0

        # 1. Embed Query
        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector) # Normalize for Cosine Similarity equivalent

        # 2. Search FAISS
        k = 1 # Top 1 match
        D, I = self.index.search(query_vector, k)
        
        score = float(D[0][0])
        idx = int(I[0][0])

        if score >= threshold:
            # Hit! Retrieve text
            if 0 <= idx < len(self.faiss_responses):
                return self.faiss_responses[idx], score
        
        return None, score

    def get_semantic_matches(self, query: str, top_k: int = 3, threshold: float = 0.60) -> list[dict]:
        """
        Layer 2: Check Semantic Cache (FAISS) for multiple matches.
        Returns list of {"text": str, "score": float}
        """
        if self.index.ntotal == 0:
            return []

        # 1. Embed Query
        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector)

        # 2. Search FAISS
        # Ensure we don't request more than we have
        k = min(top_k, self.index.ntotal)
        if k == 0: return []
        
        D, I = self.index.search(query_vector, k)
        
        results = []
        for i in range(k):
            score = float(D[0][i])
            idx = int(I[0][i])
            
            if score >= threshold:
                if 0 <= idx < len(self.faiss_responses):
                    results.append({
                        "text": self.faiss_responses[idx],
                        "score": round(score, 2)
                    })
        
        return results

    def set_semantic_match(self, query: str, response: str):
        """
        Store response in Layer 2 Cache.
        """
        # 1. Embed Query
        vector = self.model.encode([query])
        faiss.normalize_L2(vector)
        
        # 2. Add to FAISS
        self.index.add(vector)
        
        # 3. Store Response mapping
        # In a real app, this list grows indefinitely. Use Redis/DB.
        self.faiss_responses.append(response) 

    def cache_response(self, query: str, response: str):
        """
        Update ALL cache layers with a new Query -> Response pair.
        """
        self.set_exact_match(query, response)
        self.set_semantic_match(query, response)

# Global Instance
cache_service = CacheService()
