import os
import asyncio
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from app.core.config import settings


router = APIRouter()





@tool
def summarize_text_tool(text: str, model: Optional[str] = None) -> dict:
    """
    Summarize arbitrary text by calling the backend /summarize-text endpoint.

    Args:
        text: The text content to summarize.
        model: Optional model identifier to use for summarization.
    """
    try:
        response = httpx.post(
            f"{settings.API_BASE_URL}/api/features/summarize-text",
            json={"text": text, "model": model},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to reach summarize-text endpoint: {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Summarize-text endpoint returned error: {e.response.text}"
        ) from e


@tool
def draft_response_tool(message: str, tone: str = "auto") -> dict:
    """
    Draft a response by calling the backend /draft-response endpoint.

    Args:
        message: The original user message to improve.
        tone: Desired tone (professional, casual, friendly, formal, auto).
    """
    try:
        response = httpx.post(
            f"{settings.API_BASE_URL}/api/features/draft-response",
            json={"message": message, "tone": tone},
            timeout=60.0,
        )
        response.raise_for_status()
        print(response.json())
        return response.json()
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to reach draft-response endpoint: {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Draft-response endpoint returned error: {e.response.text}"
        ) from e


@tool
def translate_text_tool(
    text: str,
    target_language: str,
    model: Optional[str] = None,
) -> dict:
    """
    Translate text by calling the backend /translate-text endpoint.

    Args:
        text: The text to translate.
        target_language: Target language code (e.g. 'en', 'de', 'fr').
        model: Optional model identifier to use for translation.
    """
    try:
        response = httpx.post(
            f"{settings.API_BASE_URL}/api/features/translate-text",
            json={
                "text": text,
                "target_language": target_language,
                "model": model,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to reach translate-text endpoint: {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Translate-text endpoint returned error: {e.response.text}"
        ) from e


groq_model = OpenAIModel(
    client_args={
        "api_key": settings.GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
    },
    model_id=settings.AI_MODEL,
    params={
        "temperature": 0.0,
        "max_tokens": 1000,
    },
)


assistant_agent = Agent(
    model=groq_model,
    tools=[summarize_text_tool, draft_response_tool, translate_text_tool],
    system_prompt=(
        "You are a focused writing and translation assistant for chat messages.\n"
        "\n"
        "Your capabilities come ONLY from three tools that call backend APIs:\n"
        "- summarize_text_tool: summarize a block of text.\n"
        "- draft_response_tool: improve grammar and rewrite a single message in a chosen tone.\n"
        "- translate_text_tool: translate text into a target language.\n"
        "\n"
        "Behavior rules (you MUST follow these):\n"
        "1) For each user request, choose at most ONE of these tools whose purpose best matches the request.\n"
        "2) Do NOT describe the tools, their names, or that you are calling tools. The user should only see the final answer.\n"
        "3) For summarization requests, return a clear summary of the given text (1–2 short paragraphs or 3–6 bullet points). Do not talk about what you did, just give the summary.\n"
        "4) For drafting/rewriting requests, return ONLY the improved or tone-adjusted message the user should send, not an explanation.\n"
        "5) For translation requests, return ONLY the translated text (you may optionally append a short note like '— translated to <language>').\n"
        "6) Do not combine summarize, draft, and translate in a single answer unless the user explicitly asks for multiple operations.\n"
        "7) If a tool fails or is unavailable, apologize briefly and answer using your own reasoning, still following the formatting rules above.\n"
    ),
)


class AssistantRequest(BaseModel):
    message: str


@router.post("/assistant")
async def run_assistant(request: AssistantRequest) -> JSONResponse:
    """
    Run the Strands-based AI assistant.

    The assistant can internally decide to call tools for:
    - Text summarization (/summarize-text)
    - Drafting responses (/draft-response)
    - Translation (/translate-text)
    """
    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    loop = asyncio.get_event_loop()
    try:
        # Run the (blocking) agent call in a thread so we do not block the event loop
        result = await loop.run_in_executor(None, assistant_agent, message)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Assistant failed: {str(e)}",
        ) from e

    return JSONResponse(content={"reply": str(result)})

