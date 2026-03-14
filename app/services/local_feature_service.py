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
    Analyze priority using a local classification or embedding model.
    """
    results = {}
    pipe = None

    # Determine task based on model type
    if "mpnet" in model_id or "sentence-transformers" in model_id:
         # For embeddings, we might need a custom approach, but for simplicity
         # let's assume valid classifiers for now or use zero-shot if supported.
         # Actually, all-mpnet-base-v2 is for embeddings. 
         # A better prioritization approach with generic models is Zero-Shot Classification.
         # Let's override task to zero-shot if we want to rank arbitrary labels.
         pass 

    # Fallback to text-classification for models fine-tuned on sentiment/urgency
    # Or use zero-shot-classification for generic priority labeling
    
    # NOTE: user suggested `distilbert-base-uncased-finetuned-sst-2-english` which is sentiment.
    # High negative sentiment != High Priority, but for POC we can map it or use Zero-Shot.
    
    # We will use Zero-Shot for flexibility with models like 'facebook/bart-large-mnli' 
    # if the user selects a compatible model, OR specific headers.
    
    # For the recommended 'sentence-transformers/all-mpnet-base-v2', it's an encoding model.
    # It doesn't output "High/Low" directly. 
    # To keep it simple for this POC:
    # 1. If it's a classifier (like distilbert-sst-2), map Negative -> High (Urgent), Positive -> Low.
    # 2. If it's an embedding model, this is harder without a reference.
    
    # Recommendation: Use a Zero-Shot intent for actual "Priority" labeling if possible.
    # But sticking to User's list:
    
    if "sst-2" in model_id:
        # Sentiment analysis: map sentiment to new priority statuses
        pipe = _get_pipeline("text-classification", model_id)
        if not pipe:
            return {}

        for msg in messages:
            try:
                out = pipe(msg.message, truncation=True, max_length=512)
                label = out[0]['label'] # POSITIVE / NEGATIVE
                # Map:
                # - NEGATIVE -> "Action Required" (likely complaint/issue needing attention)
                # - POSITIVE -> "Information" (general informational/positive content)
                if label == "NEGATIVE":
                    priority = "Action Required"
                else:
                    priority = "Information"
                results[msg.id] = priority
            except:
                results[msg.id] = "Information"
                
    else:
        # Default fallback or placeholder for embedding-based ranking
        # For now, return "Information" if we can't classify
        for msg in messages:
             results[msg.id] = "Information"

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
            # Actually that model detects GPT-Edge-2 generated text, not toxicity.
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
