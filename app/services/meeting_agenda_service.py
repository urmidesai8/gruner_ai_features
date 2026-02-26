"""
Agenda vs Discussion Intelligence: compare planned agenda vs actual discussion.

Uses LLM to allocate user-provided actual_meeting_minutes across agenda items
and off-agenda topics, then produces overrun/underrun/missed insights.
"""
import json
from typing import List, Dict, Optional, Any

from app.services.summarizer import groq_client
from app.services.summarizer import _extract_json_from_response


def analyze_agenda_vs_discussion(
    transcript: str,
    agenda_items: List[Dict[str, Any]],
    actual_meeting_minutes: int,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare planned agenda vs actual discussion.

    Args:
        transcript: Full meeting transcription text.
        agenda_items: List of {"title": str, "planned_minutes": int}.
        actual_meeting_minutes: Total actual meeting duration in minutes (user input).
        model: Optional LLM model name.

    Returns:
        {
            "agenda_items": [{"title", "planned_minutes", "actual_minutes", "status"}, ...],
            "off_agenda_topics": [{"title", "actual_minutes"}, ...],
            "insights": ["Agenda overrun: ...", "Missed topic: ...", ...]
        }
    """
    if not (transcript and transcript.strip()):
        return {
            "agenda_items": [],
            "off_agenda_topics": [],
            "insights": ["No transcript provided."],
        }
    if not agenda_items:
        return {
            "agenda_items": [],
            "off_agenda_topics": [],
            "insights": ["No agenda items provided."],
        }
    if actual_meeting_minutes <= 0:
        return {
            "agenda_items": [
                {
                    "title": item.get("title", ""),
                    "planned_minutes": item.get("planned_minutes", 0),
                    "actual_minutes": 0,
                    "status": "missed",
                }
                for item in agenda_items
            ],
            "off_agenda_topics": [],
            "insights": ["Actual meeting duration must be greater than 0."],
        }

    agenda_text = "\n".join(
        f"- {item.get('title', '')} (planned: {item.get('planned_minutes', 0)} min)"
        for item in agenda_items
    )

    prompt = f"""You are analyzing a meeting transcript to compare the PLANNED agenda with what ACTUALLY happened.

PLANNED AGENDA (with planned minutes per topic):
{agenda_text}

ACTUAL TOTAL MEETING DURATION: {actual_meeting_minutes} minutes

MEETING TRANSCRIPT:
---
{transcript[:12000]}
---
(Transcript may be truncated if very long.)

TASK:
1. Allocate the total of {actual_meeting_minutes} minutes across:
   - Each agenda item: estimate how many minutes were actually spent on that topic (based on discussion in the transcript).
   - Any off-agenda topics: topics discussed that were NOT on the agenda, with estimated minutes for each.
2. The SUM of actual_minutes for all agenda items PLUS all off_agenda_topics must equal exactly {actual_meeting_minutes}.
3. If an agenda topic was not discussed at all, give it actual_minutes: 0 and status "missed".
4. For each agenda item set status: "underrun" (actual < planned), "on_track" (actual ≈ planned), "overrun" (actual > planned), or "missed" (actual 0).
5. Generate 2-5 short insight strings (e.g. "Agenda overrun: Release planning (+15 min)", "Missed topic: Risks", "Off-agenda discussion: Vendor pricing (8 min)").

Return ONLY a single JSON object with this exact structure (no markdown, no code fence):
{{
  "agenda_items": [
    {{ "title": "<agenda title>", "planned_minutes": <number>, "actual_minutes": <number>, "status": "underrun|on_track|overrun|missed" }}
  ],
  "off_agenda_topics": [
    {{ "title": "<topic name>", "actual_minutes": <number> }}
  ],
  "insights": [
    "<insight string 1>",
    "<insight string 2>"
  ]
}}
"""

    try:
        if not groq_client.api_key:
            return _fallback_result(agenda_items, "GROQ_API_KEY not set.")

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
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        }
        completion = groq_client.chat.completions.create(**api_params)
        response_text = completion.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_result(agenda_items, str(e))

    # Parse and validate
    try:
        if "```" in response_text:
            response_text = _extract_json_from_response(response_text)
        raw = json.loads(response_text)
    except json.JSONDecodeError:
        return _fallback_result(agenda_items, "Invalid JSON from model.")

    # Normalize: ensure every agenda item appears with actual_minutes and status
    planned_by_title = {item.get("title", "").strip(): item.get("planned_minutes", 0) for item in agenda_items}
    llm_agenda = {x.get("title", "").strip(): x for x in raw.get("agenda_items", [])}

    agenda_items_out = []
    for item in agenda_items:
        title = (item.get("title") or "").strip()
        planned = item.get("planned_minutes", 0)
        llm_row = llm_agenda.get(title, {})
        actual = llm_row.get("actual_minutes")
        if actual is None:
            actual = 0
        status = llm_row.get("status", "missed" if actual == 0 else "on_track")
        if status not in ("underrun", "on_track", "overrun", "missed"):
            status = "missed" if actual == 0 else ("overrun" if actual > planned else ("underrun" if actual < planned else "on_track"))
        agenda_items_out.append({
            "title": title,
            "planned_minutes": planned,
            "actual_minutes": int(actual),
            "status": status,
        })

    off_agenda = []
    for x in raw.get("off_agenda_topics", []):
        t = (x.get("title") or "").strip()
        m = x.get("actual_minutes", 0)
        if t:
            off_agenda.append({"title": t, "actual_minutes": int(m)})

    insights = [str(s).strip() for s in raw.get("insights", []) if s]

    return {
        "agenda_items": agenda_items_out,
        "off_agenda_topics": off_agenda,
        "insights": insights,
    }


def _fallback_result(agenda_items: List[Dict], error_msg: str) -> Dict[str, Any]:
    """Return a safe structure when LLM fails."""
    return {
        "agenda_items": [
            {
                "title": item.get("title", ""),
                "planned_minutes": item.get("planned_minutes", 0),
                "actual_minutes": 0,
                "status": "missed",
            }
            for item in agenda_items
        ],
        "off_agenda_topics": [],
        "insights": [f"Analysis unavailable: {error_msg}"],
    }
