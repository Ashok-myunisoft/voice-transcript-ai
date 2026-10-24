from pydantic import BaseModel
from app.services.llm_client import complete_json


class ActionItem(BaseModel):
    task: str
    owner_speaker_id: str | None = None
    owner_name: str | None = None
    deadline: str | None = None
    status: str | None = None
    evidence: str
    confidence: float = 0.5


def extract_action_items(segments: list[dict], speakers: list[dict]) -> list[dict]:
    prompt = "Extract only explicit commitments as JSON {action_items:[{task,owner_speaker_id,owner_name,deadline,status,evidence,confidence}]}. Never infer unknown owners, deadlines, or status; owner_speaker_id must match transcript speaker IDs."
    if segments:
        valid = []
        known_speakers = {s["speaker_id"] for s in speakers}
        for start in range(0, len(segments), 80):
            result = complete_json(prompt, {"segments": segments[start:start + 80], "speakers": speakers})
            if not result or not isinstance(result.get("action_items"), list):
                continue
            for raw in result["action_items"]:
                try:
                    item = ActionItem.model_validate(raw)
                    if item.owner_speaker_id and item.owner_speaker_id not in known_speakers:
                        item.owner_speaker_id = None
                        item.owner_name = None
                    valid.append(item.model_dump())
                except Exception:
                    continue
        if valid:
            unique = {}
            for item in valid:
                unique[(item["task"].casefold(), item["owner_speaker_id"], item["deadline"])] = item
            return list(unique.values())
    # Conservative baseline: only explicit commitment language is surfaced; no owner/deadline inference.
    import re
    commitments = re.compile(r"\b(?:i\s+will|i'll|we\s+will|we'll)\s+(.+)", re.I)
    names = {s["speaker_id"]: s["name"] for s in speakers}
    items = []
    for segment in segments:
        match = commitments.search(segment["text"])
        if match and len(re.findall(r"\b\w+\b", match.group(1))) >= 2:
            speaker_id = segment.get("speaker_id")
            owner = next((speaker for speaker in speakers if speaker["speaker_id"] == speaker_id), None)
            owner_name = names.get(speaker_id) if owner and owner.get("identity_status") in {"confirmed", "user_corrected"} else None
            items.append(ActionItem(task=segment["text"].strip(), owner_speaker_id=speaker_id,
                                    owner_name=owner_name, evidence=segment["text"],
                                    confidence=0.45).model_dump())
    return items
