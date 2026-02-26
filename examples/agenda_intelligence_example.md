# Agenda vs Discussion Intelligence – Example for Testing

Use this transcript and input to test the **Agenda vs Discussion Intelligence** feature (UI or API).

---

## 1. Input: Planned agenda and actual meeting time

- **Agenda items** (title + planned minutes):
  - Sprint status — **10 min**
  - Release planning — **15 min**
  - Risks — **5 min**
- **Actual meeting duration:** **48 minutes**

---

## 2. Example transcript (paste into Transcript or use after transcribing)

```
Alright, let's get started. First up is sprint status. We had a good week, most stories are done. John, can you give a quick update? Yeah, we closed out the login bug and the API is ready for QA. Great, that's it for sprint status.

Moving on to release planning. So we need to lock the scope for the March release. The main debate was whether to include the dashboard redesign. Sarah thinks we should slip it to April. After a long discussion we decided to keep it in March but cut the mobile tweaks. We also went through the timeline again and aligned with marketing. Okay, that took longer than expected but we have a clear plan now.

Oh before we wrap, can we talk about vendor pricing? The new quote from Acme came in and it's 12% higher. We spent some time on whether to push back or accept. We'll get procurement to negotiate and circle back next week.

Alright, we're out of time. We didn't get to risks today – we can do that async or next standup. Thanks everyone.
```

---

## 3. Expected output (structure and approximate values)

The LLM may vary slightly; the structure and *type* of insights should match.

### Request (API)

```json
{
  "transcript": "<paste the transcript above>",
  "agenda_items": [
    { "title": "Sprint status", "planned_minutes": 10 },
    { "title": "Release planning", "planned_minutes": 15 },
    { "title": "Risks", "planned_minutes": 5 }
  ],
  "actual_meeting_minutes": 48
}
```

### Expected response shape

```json
{
  "agenda_items": [
    {
      "title": "Sprint status",
      "planned_minutes": 10,
      "actual_minutes": 5,
      "status": "underrun"
    },
    {
      "title": "Release planning",
      "planned_minutes": 15,
      "actual_minutes": 30,
      "status": "overrun"
    },
    {
      "title": "Risks",
      "planned_minutes": 5,
      "actual_minutes": 0,
      "status": "missed"
    }
  ],
  "off_agenda_topics": [
    {
      "title": "Vendor pricing",
      "actual_minutes": 8
    }
  ],
  "insights": [
    "Agenda overrun: Release planning (+15 min)",
    "Missed topic: Risks",
    "Off-agenda discussion: Vendor pricing (8 min)",
    "Sprint status finished under time (5 min vs 10 min planned)"
  ]
}
```

### Notes

- **Sprint status:** Short in the transcript → ~5 min actual, **underrun**.
- **Release planning:** Most of the discussion → ~30 min actual, **overrun**.
- **Risks:** Explicitly not discussed → 0 min, **missed**.
- **Vendor pricing:** Discussed but not on agenda → **off_agenda_topics** with ~8 min.
- Sum of actual minutes should equal **48** (5 + 30 + 0 + 8 + buffer ≈ 48).

---

## 4. How to test in the UI

1. Open **Meeting Recording Transcription** (`/meeting_transcription.html`).
2. Add agenda items (e.g. Sprint status 10, Release planning 15, Risks 5) via **+ Add agenda item**.
3. Set **Actual meeting duration (minutes)** to **48**.
4. Paste the transcript from section 2 (or from `examples/agenda_intelligence_transcript.txt`) into **Paste transcript (for testing)**, then click **Use pasted transcript**. (Or transcribe an audio file that matches the example.)
5. Click **Analyze agenda vs discussion**.
6. Check that you see **overrun** for Release planning, **missed** for Risks, **underrun** for Sprint status, and **Off-agenda: Vendor pricing**.

---

## 5. How to test via API

```bash
curl -X POST http://localhost:8000/api/features/meeting/agenda-intelligence \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Alright, let'\''s get started. First up is sprint status...",
    "agenda_items": [
      {"title": "Sprint status", "planned_minutes": 10},
      {"title": "Release planning", "planned_minutes": 15},
      {"title": "Risks", "planned_minutes": 5}
    ],
    "actual_meeting_minutes": 48
  }'
```

Use the full transcript from section 2 in the `transcript` field. Compare the response to the expected structure in section 3.
