import json
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from app.services.summarizer import groq_client  # reuse same LLM client


HY_MT_MODEL_ID = "tencent/HY-MT1.5-1.8B"
T5_MODEL_ID = "google-t5/t5-base"
_hy_mt_tokenizer = None
_hy_mt_model = None
_t5_tokenizer = None
_t5_model = None

# Language code -> name for HY-MT prompt (model expects language names in the instruction)
_LANG_CODE_TO_NAME = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "hi": "Hindi", "pt": "Portuguese", "ru": "Russian",
    "ar": "Arabic", "it": "Italian", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "ms": "Malay", "uk": "Ukrainian",
}


def _get_hy_mt_model():
    """Lazy-load tokenizer and model for tencent/HY-MT1.5-1.8B (local transformers)."""
    global _hy_mt_tokenizer, _hy_mt_model
    if _hy_mt_model is not None:
        return _hy_mt_tokenizer, _hy_mt_model

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    _hy_mt_tokenizer = AutoTokenizer.from_pretrained(HY_MT_MODEL_ID, trust_remote_code=True)
    _hy_mt_model = AutoModelForCausalLM.from_pretrained(HY_MT_MODEL_ID, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _hy_mt_model = _hy_mt_model.to(device)
    _hy_mt_model.eval()
    return _hy_mt_tokenizer, _hy_mt_model


def _translate_text_hy_mt(text: str, target_language: str) -> dict:
    import torch

    if not text:
        return {"translated_text": "", "detected_language": "unknown"}

    tokenizer, model = _get_hy_mt_model()
    target_name = _LANG_CODE_TO_NAME.get(target_language.strip().lower(), target_language)
    prompt = (
        f"Translate the following segment into {target_name}, without additional explanation.\n\n{text}"
    )

    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=min(2048, getattr(model.config, "max_position_embeddings", 2048)),
    )
    # Only pass input_ids and attention_mask; model does not use token_type_ids
    gen_kwargs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }

    with torch.no_grad():
        out = model.generate(
            **gen_kwargs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (after the prompt)
    input_length = gen_kwargs["input_ids"].shape[1]
    translated = tokenizer.decode(out[0][input_length:], skip_special_tokens=True).strip()
    return {"translated_text": translated, "detected_language": "unknown"}


def _get_t5_model():
    """Lazy-load tokenizer and model for google-t5/t5-base (local transformers)."""
    global _t5_tokenizer, _t5_model
    if _t5_model is not None:
        return _t5_tokenizer, _t5_model

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    _t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_ID)
    _t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _t5_model = _t5_model.to(device)
    _t5_model.eval()
    return _t5_tokenizer, _t5_model


def _translate_text_t5(text: str, target_language: str) -> dict:
    """Translate using T5 (prefix: translate English to {target}: text)."""
    import torch

    if not text:
        return {"translated_text": "", "detected_language": "unknown"}

    tokenizer, model = _get_t5_model()
    target_name = _LANG_CODE_TO_NAME.get(target_language.strip().lower(), target_language)
    # T5 was trained with prefix "translate English to German: ..." (English -> other)
    prefix = f"translate English to {target_name}: "
    input_text = prefix + text

    device = next(model.parameters()).device
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    gen_kwargs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }

    with torch.no_grad():
        out = model.generate(
            **gen_kwargs,
            max_length=150,
            min_length=0,
            num_beams=2,
            length_penalty=2.0,
            early_stopping=True,
        )

    translated = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return {"translated_text": translated, "detected_language": "unknown"}


def translate_text(text: str, target_language: str = "en", model: str = None) -> dict:
    """Translate a single block of text into a target language using the LLM.

    Returns:
    {
      "translated_text": "Translated text...",
      "detected_language": "source language code"
    }
    """
    if not text:
        return {"translated_text": ""}

    if model == HY_MT_MODEL_ID:
        try:
            return _translate_text_hy_mt(text, target_language)
        except Exception as e:
            print(f"Error translating text with HY-MT: {e}")
            return {"translated_text": "Error during translation.", "detected_language": "unknown"}

    if model == T5_MODEL_ID:
        try:
            return _translate_text_t5(text, target_language)
        except Exception as e:
            print(f"Error translating text with T5: {e}")
            return {"translated_text": "Error during translation.", "detected_language": "unknown"}

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
        "model": model or "llama-3.3-70b-versatile",
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


def translate_messages_batch(requests: List[Dict], model: str = None) -> Dict:
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

    if model == HY_MT_MODEL_ID:
        target_language = requests[0].get("target_language", "en")
        out = {}
        for item in requests:
            msg_id = str(item.get("id"))
            try:
                out[msg_id] = _translate_text_hy_mt(item.get("text", ""), target_language)
            except Exception as e:
                print(f"HY-MT batch translation error: {e}")
                out[msg_id] = {"translated_text": "Error during translation.", "detected_language": "unknown"}
        return {"translations": out}

    if model == T5_MODEL_ID:
        target_language = requests[0].get("target_language", "en")
        out = {}
        for item in requests:
            msg_id = str(item.get("id"))
            try:
                out[msg_id] = _translate_text_t5(item.get("text", ""), target_language)
            except Exception as e:
                print(f"T5 batch translation error: {e}")
                out[msg_id] = {"translated_text": "Error during translation.", "detected_language": "unknown"}
        return {"translations": out}

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
        "model": model or "llama-3.1-8b-instant",
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

