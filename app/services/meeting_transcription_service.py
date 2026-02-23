"""
Meeting Transcription Service for storing and querying meeting transcriptions in Qdrant.

This service handles:
1. Storing meeting transcriptions in Qdrant vector DB
2. Querying transcriptions for Q&A based on participant filtering
"""
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from qdrant_client.http import models as qmodels

from app.services.memory_service import _get_qdrant_client, embed_text, VECTOR_SIZE, DISTANCE


MEETING_TRANSCRIPTION_COLLECTION = "meeting_transcription"


def ensure_meeting_transcription_collection() -> None:
    """Ensure the meeting_transcription collection exists in Qdrant with required indexes."""
    try:
        client = _get_qdrant_client()
        collection_created = False
        
        if not client.collection_exists(MEETING_TRANSCRIPTION_COLLECTION):
            client.create_collection(
                collection_name=MEETING_TRANSCRIPTION_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_SIZE,
                    distance=DISTANCE,
                ),
            )
            print(f"Created Qdrant collection: {MEETING_TRANSCRIPTION_COLLECTION}")
            collection_created = True
        
        # Ensure payload index exists for participant_ids (required for filtering)
        # Check if index already exists by trying to create it (will fail silently if exists)
        try:
            client.create_payload_index(
                collection_name=MEETING_TRANSCRIPTION_COLLECTION,
                field_name="participant_ids",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            print(f"Created payload index for 'participant_ids' in {MEETING_TRANSCRIPTION_COLLECTION}")
        except Exception as idx_error:
            # Index might already exist, which is fine
            if "already exists" not in str(idx_error).lower() and collection_created:
                # Only log if it's a new collection and the error isn't about existing index
                print(f"Note: Could not create index for participant_ids: {idx_error}")
    except Exception as e:
        print(f"Warning: Could not ensure meeting_transcription collection: {e}")


def store_meeting_transcription(
    transcription: str,
    participant_ids: List[str],
    meeting_agenda: Optional[str] = None,
    meeting_id: Optional[str] = None,
) -> Dict:
    """
    Store a meeting transcription in Qdrant vector DB.
    
    Args:
        transcription: The meeting transcription text
        participant_ids: List of user IDs who participated in the meeting
        meeting_agenda: Optional meeting agenda/topic
        meeting_id: Optional meeting ID (generated if not provided)
    
    Returns:
        Dictionary with meeting_id and status
    """
    if not transcription or not transcription.strip():
        raise ValueError("Transcription text is required")
    
    if not participant_ids:
        raise ValueError("At least one participant ID is required")
    
    ensure_meeting_transcription_collection()
    client = _get_qdrant_client()
    
    # Generate meeting_id if not provided
    if not meeting_id:
        meeting_id = str(uuid.uuid4())
    
    # Get current date and time
    meeting_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create embedding for the transcription
    transcription_vector = embed_text(transcription)
    
    # Create payload
    payload = {
        "meeting_id": meeting_id,
        "participant_ids": participant_ids,
        "meeting_time": meeting_time,
        "meeting_agenda": meeting_agenda or "",
        "transcription": transcription,
    }
    
    # Generate a unique point ID (use meeting_id as UUID string)
    # Convert meeting_id UUID to a hash for use as point ID
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, meeting_id))
    
    # Store in Qdrant
    client.upsert(
        collection_name=MEETING_TRANSCRIPTION_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector=transcription_vector,
                payload=payload,
            )
        ],
    )
    
    return {
        "meeting_id": meeting_id,
        "status": "stored",
        "meeting_time": meeting_time,
    }


def search_meeting_transcriptions(
    query: str,
    participant_id: str,
    limit: int = 5,
    score_threshold: float = 0.3,
) -> List[Dict]:
    """
    Search meeting transcriptions for a specific participant.
    
    Args:
        query: Search query text
        participant_id: User ID to filter meetings by (must be in participant_ids)
        limit: Maximum number of results
        score_threshold: Minimum similarity score threshold
    
    Returns:
        List of matching meeting transcriptions with scores
    """
    if not query or not query.strip():
        return []
    
    if not participant_id:
        raise ValueError("Participant ID is required")
    
    ensure_meeting_transcription_collection()
    client = _get_qdrant_client()
    
    # Create query embedding
    query_vector = embed_text(query)
    
    # Filter: participant_id must be in participant_ids array
    filter_ = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="participant_ids",
                match=qmodels.MatchAny(any=[participant_id]),
            )
        ]
    )
    
    # Search - use the same pattern as memory_service.py
    # Note: score_threshold might not be supported in all versions, filter results manually if needed
    results = client.query_points(
        collection_name=MEETING_TRANSCRIPTION_COLLECTION,
        query=query_vector,
        limit=limit,
        query_filter=filter_,
    )
    
    # Format results and filter by score_threshold
    # query_points returns QueryResponse with .points attribute
    formatted_results = []
    
    # Handle both search() and query_points() return types
    if hasattr(results, 'points'):
        # query_points returns QueryResponse
        points = results.points
    elif hasattr(results, '__iter__'):
        # search() returns list of ScoredPoint
        points = results
    else:
        points = []
    
    for hit in points:
        # Get score - might be in different attributes depending on method
        score = getattr(hit, 'score', None)
        if score is None:
            score = getattr(hit, 'distance', 0)  # Some methods use distance instead
        
        # Filter by score threshold if provided
        if score < score_threshold:
            continue
        
        # Get payload
        payload = getattr(hit, 'payload', {})
        if not payload:
            continue
            
        formatted_results.append({
            "meeting_id": payload.get("meeting_id"),
            "meeting_time": payload.get("meeting_time"),
            "meeting_agenda": payload.get("meeting_agenda", ""),
            "transcription": payload.get("transcription", ""),
            "participant_ids": payload.get("participant_ids", []),
            "similarity_score": round(score, 4),
        })
    
    return formatted_results


def ask_meeting_question(
    query: str,
    participant_id: str,
    limit: int = 3,
    score_threshold: float = 0.3,
    model: Optional[str] = None,
) -> Dict:
    """
    Answer a question based on meeting transcriptions for a specific participant.
    
    Args:
        query: User's question
        participant_id: User ID asking the question (filters meetings they participated in)
        limit: Maximum number of relevant meetings to consider
        score_threshold: Minimum similarity score for relevant meetings
        model: Optional LLM model for generating answer
    
    Returns:
        Dictionary with answer, relevant meetings, and sources
    """
    from app.services.summarizer import groq_client
    
    if not query or not query.strip():
        raise ValueError("Query is required")
    
    # Search for relevant meeting transcriptions
    # Use lower threshold to catch more potentially relevant meetings
    # The LLM can filter out irrelevant ones
    relevant_meetings = search_meeting_transcriptions(
        query=query,
        participant_id=participant_id,
        limit=max(limit, 5),  # Get more meetings to have better context
        score_threshold=max(score_threshold, 0.1),  # Lower threshold to catch more results
    )
    
    if not relevant_meetings:
        return {
            "answer": "I couldn't find any relevant meetings for your question.",
            "relevant_meetings": [],
            "sources": [],
        }
    
    # Build context from relevant meetings
    # Use full transcriptions, but limit total context to avoid token limits
    context_parts = []
    total_length = 0
    max_context_length = 8000  # Increased from 1000 to allow more context
    
    for meeting in relevant_meetings:
        agenda = meeting.get("meeting_agenda", "")
        transcription = meeting.get("transcription", "")
        meeting_time = meeting.get("meeting_time", "")
        similarity_score = meeting.get("similarity_score", 0)
        
        # Include full transcription if space allows, otherwise truncate
        remaining_space = max_context_length - total_length
        if remaining_space <= 0:
            break
            
        if len(transcription) > remaining_space:
            transcription_snippet = transcription[:remaining_space] + "..."
        else:
            transcription_snippet = transcription
        
        context_parts.append(
            f"=== Meeting ({meeting_time}) ===\n"
            f"Agenda: {agenda}\n"
            f"Relevance Score: {similarity_score:.2f}\n"
            f"Full Transcription:\n{transcription_snippet}\n"
        )
        
        total_length += len(transcription_snippet)
    
    context = "\n\n".join(context_parts)
    
    # Generate answer using LLM with improved prompt
    prompt = f"""You are an expert at analyzing meeting transcriptions and answering questions accurately.

MEETING TRANSCRIPTIONS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Carefully read through the meeting transcriptions above
2. Look for information that directly or indirectly answers the user's question
3. Extract specific details, numbers, dates, names, and agreements mentioned
4. If the answer involves a number or measurement, provide the exact value mentioned
5. If the information is not explicitly stated but can be inferred, explain your reasoning
6. If the answer truly cannot be found, state: "The answer to this question cannot be found in the provided meeting transcriptions."

Provide a clear, detailed answer based on the transcriptions. Include relevant context and specific details when available.
"""
    
    try:
        if not groq_client.api_key:
            # Fallback: return relevant transcriptions without LLM answer
            return {
                "answer": f"Found {len(relevant_meetings)} relevant meeting(s). Please review the transcriptions below.",
                "relevant_meetings": relevant_meetings,
                "sources": [
                    {
                        "meeting_id": m.get("meeting_id"),
                        "meeting_time": m.get("meeting_time"),
                        "meeting_agenda": m.get("meeting_agenda"),
                        "similarity_score": m.get("similarity_score"),
                    }
                    for m in relevant_meetings
                ],
            }
        
        # Use Groq to generate answer with better model and more tokens
        api_params = {
            "model": model or "llama-3.3-70b-versatile",  # Use more capable model
            "messages": [
                {"role": "system", "content": "You are an expert at analyzing meeting transcriptions and extracting precise information to answer questions accurately."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,  # Lower temperature for more precise answers
            "max_tokens": 1000,  # Increased to allow more detailed answers
        }
        
        completion = groq_client.chat.completions.create(**api_params)
        answer = completion.choices[0].message.content.strip()
        
        return {
            "answer": answer,
            "relevant_meetings": relevant_meetings,
            "sources": [
                {
                    "meeting_id": m.get("meeting_id"),
                    "meeting_time": m.get("meeting_time"),
                    "meeting_agenda": m.get("meeting_agenda"),
                    "similarity_score": m.get("similarity_score"),
                }
                for m in relevant_meetings
            ],
        }
    except Exception as e:
        # Fallback on error
        return {
            "answer": f"Found {len(relevant_meetings)} relevant meeting(s), but encountered an error generating the answer: {str(e)}",
            "relevant_meetings": relevant_meetings,
            "sources": [
                {
                    "meeting_id": m.get("meeting_id"),
                    "meeting_time": m.get("meeting_time"),
                    "meeting_agenda": m.get("meeting_agenda"),
                    "similarity_score": m.get("similarity_score"),
                }
                for m in relevant_meetings
            ],
        }
