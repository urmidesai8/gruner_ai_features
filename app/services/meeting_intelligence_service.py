"""
Meeting Intelligence & Insights: higher-level analytics across people, topics, time, and outcomes.

Produces:
- Talk-time per participant
- Sentiment & tension detection
- Decision velocity
- Blockers / risks mentioned
- Dominant topics
- Human-readable insight bullets
"""
import json
from typing import List, Dict, Optional, Any

from app.services.summarizer import groq_client
from app.services.summarizer import _extract_json_from_response


def generate_meeting_intelligence(
    transcript: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate higher-level meeting analytics: talk-time, sentiment, decisions, blockers, topics, insights.

    Args:
        transcript: Full meeting transcription (with or without speaker labels).
        segments: Optional list of {"start", "end", "text", "speaker"?} for talk-time accuracy.
        model: Optional LLM model name.

    Returns:
        {
            "talk_time_by_participant": [{"participant": str, "percentage": float, "note": str?}, ...],
            "sentiment_insights": [{"segment_or_topic": str, "sentiment": str, "tension_detected": bool, "note": str?}, ...],
            "decision_velocity": {"decisions_count": int, "note": str?, "decisions": [{"decision": str, "context": str?}]?},
            "blockers_mentioned": [{"blocker": str, "context": str?}, ...],
            "dominant_topics": [{"topic": str, "relevance": str?}, ...],
            "insights": ["62% of speaking time by 2 participants", "Negative sentiment spike during QA", ...]
        }
    """
    if not (transcript and transcript.strip()):
        return _empty_intelligence("No transcript provided.")

    prompt = f"""You are a meeting analyst producing higher-level analytics. Analyze this meeting transcript.

MEETING TRANSCRIPT:
---
{transcript[:14000]}
---
(Transcript may be truncated if very long.)

Produce the following. Use the transcript to infer speakers from labels like "Speaker 1:", "John:", "[00:01] John:". If there are no speaker labels, estimate from context or say "Single speaker / no labels".

1. TALK-TIME BY PARTICIPANT: Estimate percentage of speaking time per participant. Sum of percentages should be 100. Include a short "note" if you inferred speakers from context.

2. SENTIMENT & TENSION: Identify moments or topics where sentiment was negative or tense (e.g. disagreement, frustration, conflict). For each, give segment_or_topic, sentiment (positive/negative/neutral), tension_detected (true/false), and a brief note.

3. DECISION VELOCITY: List decisions that were made in the meeting. Include decisions_count and optionally a short note on how quickly decisions were reached. Include a "decisions" array with each decision and optional context.

4. BLOCKERS MENTIONED: List any blockers, risks, or repeated issues mentioned (e.g. "release risk", "QA bottleneck"). Include blocker and optional context.

5. DOMINANT TOPICS: List 3–6 main themes or topics that dominated the discussion, with optional relevance (e.g. "high", "medium").

6. INSIGHTS: Generate 4–8 short, human-readable insight bullets suitable for a dashboard. Examples:
   - "62% of speaking time by 2 participants"
   - "Negative sentiment spike during QA discussion"
   - "Same release risk mentioned in last 3 meetings" (only if the transcript suggests recurrence)
   - "Decision velocity: 3 decisions in first 15 minutes"
   - "Dominant topics: timeline, scope, vendor pricing"

Return ONLY a single JSON object with this exact structure (no markdown, no code fence):
{{
  "talk_time_by_participant": [
    {{ "participant": "<name or Speaker N>", "percentage": <0-100>, "note": "<optional>" }}
  ],
  "sentiment_insights": [
    {{ "segment_or_topic": "<topic or moment>", "sentiment": "positive|negative|neutral", "tension_detected": true|false, "note": "<optional>" }}
  ],
  "decision_velocity": {{
    "decisions_count": <number>,
    "note": "<optional short note on decision speed>",
    "decisions": [
      {{ "decision": "<what was decided>", "context": "<optional>" }}
    ]
  }},
  "blockers_mentioned": [
    {{ "blocker": "<blocker or risk>", "context": "<optional>" }}
  ],
  "dominant_topics": [
    {{ "topic": "<topic>", "relevance": "<optional high|medium|low>" }}
  ],
  "insights": [
    "<insight string 1>",
    "<insight string 2>"
  ]
}}
"""

    try:
        if not groq_client.api_key:
            return _empty_intelligence("GROQ_API_KEY not set.")

        api_params = {
            "model": model or "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a meeting analyst. Output only valid JSON with the exact keys requested.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 2500,
            "response_format": {"type": "json_object"},
        }
        completion = groq_client.chat.completions.create(**api_params)
        response_text = completion.choices[0].message.content.strip()
    except Exception as e:
        return _empty_intelligence(str(e))

    try:
        if "```" in response_text:
            response_text = _extract_json_from_response(response_text)
        raw = json.loads(response_text)
    except json.JSONDecodeError:
        return _empty_intelligence("Invalid JSON from model.")

    # Normalize and ensure required keys
    def safe_list(val, default=None):
        if default is None:
            default = []
        return val if isinstance(val, list) else default

    def safe_dict(val, default=None):
        if default is None:
            default = {}
        return val if isinstance(val, dict) else default

    talk_time = safe_list(raw.get("talk_time_by_participant"))
    sentiment = safe_list(raw.get("sentiment_insights"))
    decision_velocity = safe_dict(raw.get("decision_velocity"))
    if not isinstance(decision_velocity.get("decisions"), list):
        decision_velocity["decisions"] = safe_list(decision_velocity.get("decisions"))
    blockers = safe_list(raw.get("blockers_mentioned"))
    dominant_topics = safe_list(raw.get("dominant_topics"))
    insights = [str(s).strip() for s in raw.get("insights", []) if s]

    return {
        "talk_time_by_participant": talk_time,
        "sentiment_insights": sentiment,
        "decision_velocity": decision_velocity,
        "blockers_mentioned": blockers,
        "dominant_topics": dominant_topics,
        "insights": insights,
    }


def _empty_intelligence(error_msg: str) -> Dict[str, Any]:
    return {
        "talk_time_by_participant": [],
        "sentiment_insights": [],
        "decision_velocity": {"decisions_count": 0, "note": "", "decisions": []},
        "blockers_mentioned": [],
        "dominant_topics": [],
        "insights": [f"Analysis unavailable: {error_msg}"],
    }
