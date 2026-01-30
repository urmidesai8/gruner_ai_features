import json
from typing import List, Dict

from app.services.summarizer import groq_client  # reuse same LLM client



def translate_text(text: str, target_language: str = "en") -> dict:
    """Translate a single block of text into a target language using the LLM.

    Returns:
    {
      "translated_text": "Translated text...",
      "detected_language": "source language code"
    }
    """
    if not text:
        return {"translated_text": ""}

    if not groq_client.api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Please set it in your environment variables or .env file."
        )

    prompt = f"""You are a professional translation engine.

Target language: {target_language}

Translate the following text into the target language.

Return ONLY a JSON object with this structure:
{{
  "translated_text": "translated text in the target language only",
  "detected_language": "source language code (e.g. 'en', 'es', 'fr')"
}}

Guidelines:
- Preserve the original meaning and tone.
- Do NOT add explanations or notes.
- Do NOT wrap the JSON in markdown.

Text to translate:
{text}

Return the JSON now.
"""

    api_params = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise JSON-producing translation engine. "
                    "You MUST return only valid JSON with no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 8000,
    }

    try:
        # Try JSON mode if supported
        try:
            api_params["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        completion = groq_client.chat.completions.create(**api_params)
        response_text = completion.choices[0].message.content.strip()

        # Strip markdown fences if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()
        parsed = json.loads(response_text)

        return {
            "translated_text": parsed.get("translated_text", ""),
            "detected_language": parsed.get("detected_language", "unknown")
        }

    except Exception as e:
        print(f"Error translating text with LLM: {e}")
        return {"translated_text": "Error during translation."}


def translate_messages_batch(requests: List[Dict]) -> Dict:
    """Translate a batch of messages into a target language using the LLM.

    Expects each request dict to have:
    - id: unique message ID
    - text: original text
    - target_language: language code (e.g. 'en', 'es', 'fr', 'de', 'hi', 'zh', 'ja')

    Returns:
    {
      "translations": {
        "<id>": {
          "translated_text": "Texto traducido",
          "detected_language": "en"
        },
        ...
      }
    }
    """
    if not requests:
        return {"translations": {}}

    if not groq_client.api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Please set it in your environment variables or .env file."
        )

    # For now we assume all requests share the same target_language
    target_language = requests[0].get("target_language", "en")

    # Format items for the prompt
    items_text = "\n\n".join(
        f"ID: {item['id']}\nTEXT: {item['text']}" for item in requests
    )

    prompt = f"""You are a professional translation engine.

Target language: {target_language}

Translate each text below into the target language.

Return ONLY a JSON object with this structure:
{{
  "translations": {{
    "<id>": {{
      "translated_text": "translated text in the target language only",
      "detected_language": "source language code (e.g. 'en', 'es', 'fr')"
    }},
    ...
  }}
}}

Guidelines:
- Preserve the original meaning and tone.
- Do NOT add explanations or notes.
- Do NOT wrap the JSON in markdown code fences.

Texts:
{items_text}

Return the JSON now.
"""

    api_params = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise JSON-producing translation engine. "
                    "You MUST return only valid JSON with no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }

    try:
        # Try JSON mode if supported
        try:
            api_params["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        completion = groq_client.chat.completions.create(**api_params)
        response_text = completion.choices[0].message.content.strip()

        # Strip markdown fences if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()
        parsed = json.loads(response_text)

        translations = parsed.get("translations", {})
        if not isinstance(translations, dict):
            translations = {}

        return {"translations": translations}

    except Exception as e:  # pragma: no cover - defensive
        print(f"Error translating messages with LLM: {e}")
        # Fail soft: return empty structure so API still works
        return {"translations": {}}

