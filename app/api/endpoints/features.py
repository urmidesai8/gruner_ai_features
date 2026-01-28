from fastapi import APIRouter, HTTPException
from typing import List, Optional
from fastapi.responses import JSONResponse
import json
from ...models.schemas import FeatureRequest, chat_history
from ...services.ai_service import call_groq_ai
from ...services.summarizer import generate_chat_summary
from ...services.task_classifier import extract_tasks_from_messages

router = APIRouter()

@router.post("/prioritize")
async def prioritize_messages(messages: List[FeatureRequest]):
    """Classify priority for a list of messages"""
    if not messages:
        return {}
    
    prompt_items = [f"ID: {m.id} | Msg: {m.message}" for m in messages]
    prompt_text = "\n".join(prompt_items)
    
    prompt = f"""
    Analyze the priority of the following messages. 
    Return a JSON object where keys are IDs and values are one of: 'Low', 'Normal', 'High', 'Urgent'.
    
    Messages:
    {prompt_text}
    
    Return ONLY valid JSON.
    """
    
    try:
        response_text = call_groq_ai(prompt)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        results = json.loads(response_text)
    except Exception:
        results = {m.id: "Normal" for m in messages}
        
    return JSONResponse(content=results)

@router.post("/moderate")
async def moderate_messages(messages: List[FeatureRequest]):
    """Check moderation status"""
    if not messages:
        return {}
    
    prompt_items = [f"ID: {m.id} | Msg: {m.message}" for m in messages]
    prompt_text = "\n".join(prompt_items)
    
    prompt = f"""
    Check these messages for spam, scams, or abuse.
    Return a JSON object where keys are IDs and values are objects like {{ "safe": true }} or {{ "safe": false, "reason": "spam" }}.
    
    Messages:
    {prompt_text}
    
    Return ONLY valid JSON.
    """
    
    try:
        response_text = call_groq_ai(prompt)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        results = json.loads(response_text)
    except Exception:
        results = {m.id: {"safe": True} for m in messages}

    return JSONResponse(content=results)

@router.post("/smart-replies")
async def smart_replies(messages: List[FeatureRequest]):
    """Generate smart replies"""
    if not messages:
        return {"suggestions": []}
    
    last_msg = messages[-1]
    
    prompt = f"""
    Generate 3 short, context-aware reply suggestions for the following message:
    "{last_msg.message}"
    
    Return a JSON object: {{ "suggestions": ["Yes", "No", "maybe"] }}
    Return ONLY valid JSON.
    """
    
    try:
        response_text = call_groq_ai(prompt)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        result = json.loads(response_text)
    except Exception:
        result = {"suggestions": []}

    return JSONResponse(content=result)


@router.post("/chat-summarize")
async def summarize_chat(username: Optional[str] = None) -> JSONResponse:
    """
    Generate chat summary.

    Query parameters:
    - username: Optional username to get personalized "What did I miss?" summary.
    """
    try:
        if username:
            messages = chat_history.get_unread_messages(username)
            if not messages:
                messages = chat_history.get_all_messages()
        else:
            messages = chat_history.get_all_messages()

        summary = generate_chat_summary(messages, username)

        if username:
            chat_history.mark_as_read(username)

        return JSONResponse(content=summary)

    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}") from e


@router.get("/unread-messages")
async def get_messages(username: Optional[str] = None) -> JSONResponse:
    """Get all chat messages or unread messages for a user."""
    if username:
        return JSONResponse(
            content={
                "messages": chat_history.get_unread_messages(username),
                "unread_count": chat_history.get_unread_count(username),
            }
        )

    return JSONResponse(
        content={
            "messages": chat_history.get_all_messages(),
            "total_count": len(chat_history.messages),
        }
    )


@router.post("/tasks-classifier")
async def classify_tasks(username: Optional[str] = None) -> JSONResponse:
    """Identify tasks/todos in chat messages and return them in a structured format.

    - If `username` is provided, prefer that user's unread messages; if none, use all.
    - Otherwise, run on the entire chat history.
    """
    try:
        if username:
            messages = chat_history.get_unread_messages(username)
            if not messages:
                messages = chat_history.get_all_messages()
        else:
            messages = chat_history.get_all_messages()

        result = extract_tasks_from_messages(messages)
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Error classifying tasks: {str(e)}"
        ) from e
