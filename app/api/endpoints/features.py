from fastapi import APIRouter
from typing import List
from fastapi.responses import JSONResponse
import json
from ...models.schemas import FeatureRequest
from ...services.ai_service import call_groq_ai

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
    except:
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
    except:
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
    except:
        result = {"suggestions": []}

    return JSONResponse(content=result)
