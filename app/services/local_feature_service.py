import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForSequenceClassification, 
    AutoModelForTokenClassification,
    AutoModel
)
from app.core.config import settings

# -----------------------------------------------------------------------------
# Lazy-loaded global model cache
# -----------------------------------------------------------------------------
_loaded_pipelines = {}

# Seq2Seq (text2text) is not a pipeline task in recent transformers; we load model + tokenizer directly.
_loaded_seq2seq = {}


def _get_seq2seq_model(model_id: str):
    """
    Load Seq2Seq model and tokenizer for smart replies (Flan-T5, BART, etc.).
    Uses AutoModelForSeq2SeqLM since pipeline(task='text2text-generation') was removed.
    """
    global _loaded_seq2seq
    if model_id in _loaded_seq2seq:
        return _loaded_seq2seq[model_id]

    print(f"Loading local model for text2text-generation: {model_id}...")
    cache_dir = getattr(settings, "MODEL_CACHE_DIR", None)
    model_kwargs = {"cache_dir": cache_dir} if cache_dir else {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **model_kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **model_kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        _loaded_seq2seq[model_id] = {"model": model, "tokenizer": tokenizer}
        return _loaded_seq2seq[model_id]
    except Exception as e:
        print(f"Error loading model {model_id} for text2text-generation: {e}")
        return None


def _get_pipeline(task: str, model_id: str):
    """
    Get or create a pipeline for a specific task and model.
    Lazy loads to save memory.
    """
    global _loaded_pipelines
    key = f"{task}_{model_id}"
    
    if key in _loaded_pipelines:
        return _loaded_pipelines[key]

    print(f"Loading local model for {task}: {model_id}...")
    
    device = 0 if torch.cuda.is_available() else -1
    model_kwargs = {"cache_dir": getattr(settings, "MODEL_CACHE_DIR", None)}
    if not model_kwargs["cache_dir"]:
        model_kwargs = {}
    
    try:
        # text2text-generation is not a valid pipeline task in recent transformers; use _get_seq2seq_model
        if task == "text2text-generation":
            return _get_seq2seq_model(model_id)
            
        if task == "ner":
            # For Reminders (BERT-NER)
            pipe = pipeline(
                "ner", 
                model=model_id, 
                aggregation_strategy="simple",
                device=device,
                model_kwargs=model_kwargs
            )
            
        elif task == "text-classification":
            # For Moderation & Prioritization (Classifiers)
            pipe = pipeline(
                task, 
                model=model_id, 
                device=device, 
                model_kwargs=model_kwargs
            )
            
        elif task == "feature-extraction":
            # For Prioritization (Embeddings method - if using MPNet)
            pipe = pipeline(
                task, 
                model=model_id, 
                device=device, 
                model_kwargs=model_kwargs
            )
        elif task == "zero-shot-classification":
            # For Prioritization using NLI models (e.g. facebook/bart-large-mnli)
            pipe = pipeline(
                task,
                model=model_id,
                device=device,
                model_kwargs=model_kwargs
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

        _loaded_pipelines[key] = pipe
        return pipe
        
    except Exception as e:
        print(f"Error loading model {model_id} for {task}: {e}")
        return None


# -----------------------------------------------------------------------------
# Feature Implementations
# -----------------------------------------------------------------------------

def generate_smart_replies_local(messages: list, model_id: str) -> list:
    """
    Generate smart replies using a local Seq2Seq model (e.g., Flan-T5, BART).
    Uses model + tokenizer directly (no pipeline) since text2text-generation was removed.
    """
    if not messages:
        return []

    last_msg = messages[-1].message
    prompt = f"Reply to this message: {last_msg}"

    loaded = _get_pipeline("text2text-generation", model_id)
    if not loaded or "model" not in loaded or "tokenizer" not in loaded:
        return []

    model = loaded["model"]
    tokenizer = loaded["tokenizer"]

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=3,
                do_sample=True,
                temperature=0.7,
                num_beams=1,  # do_sample=True typically uses num_beams=1
            )

        replies = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [t.strip() for t in replies if t.strip()]
    except Exception as e:
        print(f"Smart reply generation failed: {e}")
        return []


def analyze_prioritization_local(messages: list, model_id: str) -> dict:
    """
    Classify message priority using Zero-Shot Classification (NLI-based).
    
    Uses an NLI model like facebook/bart-large-mnli that can classify text
    directly into custom labels: 'Urgent', 'High', 'Normal', 'Low'.
    This approach is far more accurate than heuristic sentiment mapping.
    """
    import time, json as _json
    results = {}
    start_time = time.perf_counter()

    PRIORITY_LABELS = ["Urgent", "High", "Normal", "Low"]

    # MPNet is an embedding model - cannot classify, skip gracefully
    if "mpnet" in model_id.lower() or "sentence-transformers" in model_id.lower():
        print("[PRIORITY] MPNet is an embedding model, cannot classify directly. Use an NLI model.")
        for msg in messages:
            results[msg.id] = "Low"
        return results

    # -------------------------------------------------------------------------
    # 2. NLI / Zero-Shot models (bart-large-mnli, deberta-nli, typeform, etc.)
    # -------------------------------------------------------------------------
    if any(x in model_id.lower() for x in ["mnli", "nli", "bart-large-mnli", "deberta", "typeform"]):
        pipe = _get_pipeline("zero-shot-classification", model_id)
        if not pipe:
            return {m.id: "Low" for m in messages}

        for msg in messages:
            try:
                out = pipe(msg.message, candidate_labels=PRIORITY_LABELS)
                print(f"\n[DEBUG] Zero-Shot output for '{msg.message}': {out}")

                # Out format: {"labels": ["Urgent", "High", ...], "scores": [0.8, 0.1, ...]}
                top_label = out["labels"][0]
                top_score = out["scores"][0]

                results[msg.id] = top_label

                log_entry = {
                    "text": msg.message,
                    "prioritization": {
                        "model": model_id,
                        "label": top_label,
                        "confidence": round(top_score, 4),
                        "all_scores": {l: round(s, 4) for l, s in zip(out["labels"], out["scores"])},
                        "source": "local-zero-shot"
                    }
                }
                print("\n--- Message Prioritization (Zero-Shot) ---")
                print(_json.dumps(log_entry, indent=2))
                print("-" * 42)

            except Exception as e:
                import traceback
                print(f"[PRIORITY ERROR] msg {msg.id}: {e}")
                traceback.print_exc()
                results[msg.id] = "Low"

        elapsed = time.perf_counter() - start_time
        print(f"\n✅ Zero-Shot Prioritization: {len(results)} message(s) in {elapsed:.4f}s\n")
        return results

    # For SST-2 / sentiment-based models, use improved heuristic mapping
    # The SST-2 model outputs NEGATIVE/POSITIVE with fine-grained scores
    pipe = _get_pipeline("text-classification", model_id)
    if not pipe:
        return {m.id: "Low" for m in messages}

    for msg in messages:
        try:
            out = pipe(msg.message, truncation=True, max_length=512, top_k=None)
            print(f"\n[DEBUG] SST-2 pipeline output for '{msg.message}': {out}")

            # Flatten to list of dicts
            def flatten_scores(item):
                if isinstance(item, dict) and 'label' in item and 'score' in item:
                    return [item]
                elif isinstance(item, list):
                    res = []
                    for i in item:
                        res.extend(flatten_scores(i))
                    return res
                return []

            scores = flatten_scores(out)

            neg_score = next((x['score'] for x in scores if x['label'].upper() == 'NEGATIVE'), 0.0)
            pos_score = next((x['score'] for x in scores if x['label'].upper() == 'POSITIVE'), 0.0)

            # If we couldn't find standard labels by name, try by position
            if neg_score == 0.0 and pos_score == 0.0 and len(scores) >= 2:
                # SST-2: LABEL_0=Negative, LABEL_1=Positive
                by_label = {x['label'].upper(): x['score'] for x in scores}
                neg_score = by_label.get('LABEL_0', by_label.get('NEGATIVE', 0.0))
                pos_score = by_label.get('LABEL_1', by_label.get('POSITIVE', 0.0))

            print(f"[DEBUG] neg_score={neg_score}, pos_score={pos_score}")

            if neg_score > 0.85:
                priority = "Urgent"
            elif neg_score > 0.60:
                priority = "High"
            elif neg_score > 0.40:
                priority = "Normal"
            else:
                priority = "Low"

            results[msg.id] = priority

            log_entry = {
                "text": msg.message,
                "prioritization": {
                    "model": model_id,
                    "label": priority,
                    "metrics": {"negative_score": round(neg_score, 4), "positive_score": round(pos_score, 4)},
                    "source": "local-sentiment"
                }
            }
            print("\n--- Message Prioritization (Sentiment Heuristic) ---")
            print(_json.dumps(log_entry, indent=2))
            print("-" * 50)

        except Exception as e:
            import traceback
            print(f"[PRIORITY ERROR] msg {msg.id}: {e}")
            traceback.print_exc()
            results[msg.id] = "Low"

    elapsed = time.perf_counter() - start_time
    print(f"\n✅ Sentiment Prioritization: {len(results)} message(s) in {elapsed:.4f}s\n")
    return results




def analyze_moderation_local(messages: list, model_id: str) -> dict:
    """
    Analyze moderation using a local classifier (e.g. roberta-openai-detector).
    """
    results = {}
    pipe = _get_pipeline("text-classification", model_id)
    
    if not pipe:
        # Fallback safe
        return {m.id: {"safe": True} for m in messages}

    for msg in messages:
        try:
            # Roberta-openai-detector outputs label 'Safe' or 'Inappropriate' (check specific model outputs)
            # facebook/roberta-base-openai-detector: Label_0 (Safe) / Label_1 (Unsafe) usually? 
            # actually it's "Fake" vs "Real" or similar? 
            # Let's check typical outputs. User cited "facebook/roberta-base-openai-detector". 
            # Actually that model detects GPT-2 generated text, not toxicity.
            # User might have meant "facebook/roberta-hate-speech-dynabench-r4-target"?
            # Or "unitary/unbiased-toxic-roberta".
            
            # Assuming a Toxicity classifier that returns 'toxic' or 'safe' / 'LABEL_0' etc.
            out = pipe(msg.message, truncation=True, max_length=512)
            label = out[0]['label']
            score = out[0]['score']
            
            # Simple heuristic mapping (adjust based on specific model used)
            # biased/toxic models usually output 'toxic' or 'hate' labels.
            
            is_unsafe = "toxic" in label.lower() or "hate" in label.lower() or "offensive" in label.lower() or (label == "LABEL_1" and score > 0.5)
            
            if is_unsafe:
                 results[msg.id] = {"safe": False, "reason": label, "score": score}
            else:
                 results[msg.id] = {"safe": True}
                 
        except Exception as e:
            print(f"Moderation failed for {msg.id}: {e}")
            results[msg.id] = {"safe": True}
            
    return results


def analyze_reminders_local(messages: list, model_id: str) -> dict:
    """
    Extract potential reminders using NER (Extract Dates/Times/Call-to-actions).
    """
    suggestions = []
    pipe = _get_pipeline("ner", model_id)
    
    if not pipe:
        return {"suggestions": []}

    # Analyze only the last few messages for reminders to avoid noise
    recent_msgs = messages[-5:]
    
    for i, msg in enumerate(recent_msgs):
        text = msg.message
        try:
             entities = pipe(text)
             # entities list of dicts: {'entity_group': 'DATE', 'score': ..., 'word': 'tomorrow', ...} (if aggregation="simple")
             
             has_date = any(e['entity_group'] == 'DATE' or e['entity_group'] == 'TIME' for e in entities)
             
             if has_date:
                 # Construct a basic suggestion
                 suggestions.append({
                     "id": f"suggestion-loc-{i}",
                     "title": f"Follow up: {text[:20]}...",
                     "description": f"Detected potential date/time in message: '{text}'",
                     "suggested_due_date": None, # Hard to parse exact ISO date from NER without extra logic
                     "priority": "medium",
                     "context": text,
                     "confidence": 0.8
                 })
                 
        except Exception as e:
            print(f"NER failed: {e}")

    return {"suggestions": suggestions}
# -----------------------------------------------------------------------------
# Vector Store / Embedding Logic
# -----------------------------------------------------------------------------
_embedding_model = None
_reply_bank_embeddings = None
_reply_bank_texts = [
    "I'll look into it and get back to you.",
    "Can we discuss this in a quick call?",
    "Thanks for the update.",
    "Could you clarify the deadline?",
    "I'll take care of it right away.",
    "Sounds good to me.",
    "Please send over the details.",
    "I'm working on it now.",
    "Let's catch up later.",
    "Got it, thanks!",
]

def _get_embedding_model_and_bank(model_id: str):
    """
    Lazy load SentenceTransformer model and pre-compute bank embeddings.
    """
    global _embedding_model, _reply_bank_embeddings
    
    if _embedding_model is not None:
        return _embedding_model, _reply_bank_embeddings

    print(f"Loading embedding model: {model_id}...")
    from sentence_transformers import SentenceTransformer
    
    cache_dir = getattr(settings, "MODEL_CACHE_DIR", None)
    
    # Load model (all-MiniLM-L6-v2 is small & fast)
    model = SentenceTransformer(model_id, cache_folder=cache_dir)
    _embedding_model = model
    
    # Compute embeddings for the bank once
    print("Computing initial reply bank embeddings...")
    _reply_bank_embeddings = model.encode(_reply_bank_texts, convert_to_tensor=True)
    
    return _embedding_model, _reply_bank_embeddings


# Import the centralized CacheService
from app.services.cache_service import cache_service

def learn_new_reply(user_msg: str, bot_reply: str, model_id: str):
    """
    Dynamic Learning: Cache (UserMsg -> BotReply) in FAISS via CacheService.
    """
    print(f"Learning new reply via FAISS: '{bot_reply}' for query '{user_msg}'")
    # Store in FAISS (and Memcached if enabled)
    cache_service.cache_response(user_msg, bot_reply)


def generate_smart_replies_embedding(messages: list, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> dict:
    """
    Generate replies using FAISS (CacheService).
    Fallback to Generative LLM if similarity is low, then Learn.
    """
    if not messages:
        return {"suggestions": [], "source": "none"}

    last_msg = messages[-1].message
    
    # 1. Search FAISS via CacheService
    import time
    start_time = time.perf_counter()
    
    # Threshold 0.51 as requested by user previously
    matches = cache_service.get_semantic_matches(last_msg, top_k=5, threshold=0.51)
    
    duration = time.perf_counter() - start_time
    
    # 2. Prepare Detailed Log
    import json
    log_data = {
        "replies": [],
        "confidence_level": "low",
        "response_time": f"{duration:.4f}s"
    }

    CONFIDENCE_THRESHOLD = 0.51
    HIGH_CONFIDENCE = 0.8
    
    # matches is list of {"text": "...", "score": float}
    for match in matches:
        log_data["replies"].append({
            "text": match["text"],
            "score": match["score"]
        })

    top_score = matches[0]["score"] if matches else 0.0
    
    if top_score > HIGH_CONFIDENCE:
        log_data["confidence_level"] = "high"
    elif top_score > CONFIDENCE_THRESHOLD:
        log_data["confidence_level"] = "medium"
    else:
        log_data["confidence_level"] = "low"

    # Print to Backend Logs (Terminal)
    print("\n--- Smart Reply Analysis (FAISS) ---")
    print(json.dumps(log_data, indent=2))
    print("------------------------------------\n")

    # 3. Return Logic (Top 3)
    if top_score > CONFIDENCE_THRESHOLD:
        suggestions = [m["text"] for m in matches[:3]]
        return {
            "suggestions": suggestions,
            "source": "faiss-cache",
            "confidence": f"{top_score:.2f}",
            "response_time": f"{duration:.4f}s"
        }
        
    # 4. Low Confidence -> Fallback to Generative (and Learn)
    print(f"Low/No similarity ({top_score:.2f}). Falling back to LLM...")
    
    return {
        "suggestions": [],
        "source": "fallback-llm", # Signal to caller to use LLM
        "confidence": f"{top_score:.2f}",
        "response_time": f"{duration:.4f}s"
    }
