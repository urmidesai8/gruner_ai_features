"""
Chat Search Service for semantic search over conversation history.

This service performs semantic search on chat messages stored in memory
to find relevant conversations based on user queries.
"""
import math
import re
from typing import List, Dict, Optional
from app.models.schemas import chat_history
from app.services.memory_service import embed_text


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text (excluding stop words)."""
    # Common stop words
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'why',
        'how', 'i', 'you', 'we', 'they', 'this', 'these', 'those', 'do',
        'does', 'did', 'have', 'has', 'had', 'can', 'could', 'should', 'would'
    }
    
    # Normalize and split
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out stop words and short words
    keywords = {w for w in words if len(w) > 2 and w not in stop_words}
    return keywords


def _calculate_keyword_score(query: str, message: str) -> float:
    """
    Calculate keyword matching score between query and message.
    Returns a score between 0.0 and 1.0 based on keyword overlap.
    """
    query_keywords = _extract_keywords(query)
    message_keywords = _extract_keywords(message)
    
    if not query_keywords:
        return 0.0
    
    # Calculate overlap
    overlap = query_keywords.intersection(message_keywords)
    keyword_score = len(overlap) / len(query_keywords)
    
    # Boost for exact phrase matches
    query_lower = query.lower()
    message_lower = message.lower()
    
    # Check for exact phrase matches (case-insensitive)
    if query_lower in message_lower:
        keyword_score = min(1.0, keyword_score + 0.5)  # Strong boost for exact match
    
    # Check for important keyword combinations
    important_phrases = []
    for phrase in query_keywords:
        if len(phrase) > 4:  # Longer words are more important
            important_phrases.append(phrase)
    
    if important_phrases:
        found_important = sum(1 for phrase in important_phrases if phrase in message_lower)
        if found_important > 0:
            keyword_score = min(1.0, keyword_score + (found_important / len(important_phrases)) * 0.3)
    
    return min(1.0, keyword_score)


def search_chat_messages(
    query: str,
    limit: int = 10,
    min_score: float = 0.3,
    username: Optional[str] = None,
) -> List[Dict]:
    """
    Perform semantic search on chat messages.
    
    Args:
        query: Search query text
        limit: Maximum number of results to return
        min_score: Minimum similarity score threshold (0.0-1.0)
        username: Optional username to filter messages by sender
    
    Returns:
        List of message dictionaries with similarity scores, sorted by relevance
    """
    if not query or not query.strip():
        return []
    
    # Get all messages from chat history
    if username:
        # Filter messages by username if provided
        all_messages = [
            msg for msg in chat_history.get_all_messages()
            if msg.get("sender") == username
        ]
    else:
        all_messages = chat_history.get_all_messages()
    
    if not all_messages:
        return []
    
    # Create embedding for the query
    query_embedding = embed_text(query)
    
    # Calculate similarity for each message
    scored_messages = []
    for msg in all_messages:
        message_text = msg.get("message", "")
        
        # Skip empty messages or audio placeholders
        if not message_text or message_text == "[AUDIO]":
            continue
        
        # Calculate semantic similarity (using embeddings)
        message_embedding = embed_text(message_text)
        semantic_similarity = _cosine_similarity(query_embedding, message_embedding)
        
        # Calculate keyword matching score
        keyword_score = _calculate_keyword_score(query, message_text)
        
        # Combine scores: keyword matching is more important for exact matches
        # Weight: 60% keyword matching, 40% semantic similarity
        # But if keyword score is very high (>0.8), prioritize it more
        if keyword_score >= 0.8:
            # Strong keyword match - prioritize it heavily
            combined_score = 0.8 * keyword_score + 0.2 * semantic_similarity
        else:
            # Balanced approach
            combined_score = 0.6 * keyword_score + 0.4 * semantic_similarity
        
        # Boost for exact phrase matches
        query_lower = query.lower().strip()
        message_lower = message_text.lower()
        if query_lower in message_lower:
            combined_score = min(1.0, combined_score + 0.2)
        
        if combined_score >= min_score:
            # Create result with similarity score
            result = {
                **msg,
                "similarity_score": round(combined_score, 4),
                "keyword_score": round(keyword_score, 4),
                "semantic_score": round(semantic_similarity, 4),
                "relevance": "high" if combined_score >= 0.7 else "medium" if combined_score >= 0.5 else "low"
            }
            scored_messages.append(result)
    
    # Sort by combined similarity score (descending)
    scored_messages.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Deduplicate by message_id to avoid returning the same message multiple times
    seen_message_ids = set()
    seen_content_keys = set()
    unique_messages = []
    for msg in scored_messages:
        msg_id = msg.get("message_id")
        if msg_id:
            # Deduplicate by message_id (preferred method)
            if msg_id not in seen_message_ids:
                seen_message_ids.add(msg_id)
                unique_messages.append(msg)
        else:
            # If no message_id, deduplicate by message content + sender + timestamp
            content_key = (msg.get("message", ""), msg.get("sender", ""), msg.get("timestamp", ""))
            if content_key not in seen_content_keys:
                seen_content_keys.add(content_key)
                unique_messages.append(msg)
    
    # Return top unique results
    return unique_messages[:limit]


def search_chat_with_context(
    query: str,
    limit: int = 5,
    context_window: int = 2,
    min_score: float = 0.3,
    username: Optional[str] = None,
) -> List[Dict]:
    """
    Perform semantic search and return messages with surrounding context.
    
    Args:
        query: Search query text
        limit: Maximum number of result groups to return
        context_window: Number of messages before/after to include as context
        min_score: Minimum similarity score threshold
        username: Optional username to filter messages by sender
    
    Returns:
        List of message groups with context, sorted by relevance
    """
    # Get top matching messages
    top_messages = search_chat_messages(
        query=query,
        limit=limit * 3,  # Get more candidates to find unique contexts
        min_score=min_score,
        username=username,
    )
    
    if not top_messages:
        return []
    
    # Get all messages for context lookup
    all_messages = chat_history.get_all_messages()
    message_index_map = {msg["message_id"]: idx for idx, msg in enumerate(all_messages)}
    
    # Group messages with context
    result_groups = []
    processed_indices = set()
    
    for msg in top_messages:
        msg_id = msg["message_id"]
        if msg_id not in message_index_map:
            continue
        
        msg_idx = message_index_map[msg_id]
        
        # Skip if we've already included this message in another group
        if msg_idx in processed_indices:
            continue
        
        # Get context window
        start_idx = max(0, msg_idx - context_window)
        end_idx = min(len(all_messages), msg_idx + context_window + 1)
        
        # Mark indices as processed
        for i in range(start_idx, end_idx):
            processed_indices.add(i)
        
        # Build context group
        context_messages = all_messages[start_idx:end_idx]
        
        # Find the matching message in context
        matching_msg = next(
            (m for m in context_messages if m["message_id"] == msg_id),
            None
        )
        
        if matching_msg:
            result_groups.append({
                "matched_message": {
                    **matching_msg,
                    "similarity_score": msg["similarity_score"],
                    "keyword_score": msg.get("keyword_score", 0),
                    "semantic_score": msg.get("semantic_score", 0),
                    "relevance": msg["relevance"],
                },
                "context": [
                    {**m, "is_match": m["message_id"] == msg_id}
                    for m in context_messages
                ],
                "timestamp": matching_msg.get("timestamp"),
            })
    
    # Sort by similarity score and return top results
    result_groups.sort(
        key=lambda x: x["matched_message"]["similarity_score"],
        reverse=True
    )
    
    return result_groups[:limit]
