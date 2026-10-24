import logging

from pydantic import BaseModel, Field, ValidationError
from app.services.llm_client import complete_json


logger = logging.getLogger(__name__)


def summarize_meeting(segments: list[dict], speakers: list[dict], actions: list[dict]) -> dict:
    prompt = "Summarize only facts supported by the timestamped meeting transcript. Return JSON with overview, key_discussion_points, decisions, action_items, pending_questions, participants. Use speaker IDs where names are unknown; do not invent decisions."
    partials = []
    for start in range(0, len(segments), 80):
        result = complete_json(prompt, {"segments": segments[start:start + 80], "speakers": speakers, "action_items": actions})
        if result:
            try:
                partials.append(MeetingSummary.model_validate(result).model_dump())
            except ValidationError as exc:
                logger.warning("Summary response validation failed for transcript chunk %s: %s", start // 80 + 1, exc.errors(include_input=False))
    if partials:
        result = partials[0] if len(partials) == 1 else complete_json(
            "Combine these partial meeting notes into a concise JSON meeting summary with overview, key_discussion_points, decisions, action_items, pending_questions, participants. Preserve only supported statements; do not invent decisions.",
            {"partial_summaries": partials, "speakers": speakers, "action_items": actions})
        if result:
            try:
                summary = MeetingSummary.model_validate(result).model_dump()
                summary["action_items"] = actions
                summary["participants"] = speakers
                return summary
            except ValidationError as exc:
                logger.warning("Combined summary response validation failed: %s", exc.errors(include_input=False))
    # Do not label raw transcript excerpts as summary points when generation fails.
    return {"overview": "Transcript available; automated meeting summary is unavailable.",
            "key_discussion_points": [], "decisions": [], "action_items": actions,
            "pending_questions": [], "participants": speakers,
            "summary_note": "Summary generation failed. Check server logs for LLM configuration, request, or response validation errors. No decisions were inferred."}


class MeetingSummary(BaseModel):
    overview: str
    key_discussion_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[dict] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)
    participants: list[dict] = Field(default_factory=list)
