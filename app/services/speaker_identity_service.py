import re

# Only accept explicit introduction wording. Treating every "I am X" as a
# self-introduction incorrectly turns ordinary phrases such as "I am sure"
# into speaker names.
INTRO = re.compile(
    r"(?i:\b(?:my name is|this is|i am|i'm)\s+)"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)


def resolve_speakers(segments: list[dict]) -> list[dict]:
    speakers = sorted({s["speaker_id"] for s in segments if s.get("speaker_id")})
    evidence: dict[str, set[str]] = {speaker: set() for speaker in speakers}
    for segment in segments:
        speaker = segment.get("speaker_id")
        match = INTRO.search(segment["text"])
        if speaker in evidence and match:
            evidence[speaker].add(match.group(1).strip().title())
    return [{"speaker_id": speaker,
             "name": next(iter(names)) if len(names) == 1 else None,
             "identity_status": "unconfirmed" if len(names) == 1 else ("conflicting" if names else "unconfirmed"),
             "confidence": 0.5 if len(names) == 1 else 0.0,
             "evidence": "Possible self-introduction; verify name spelling" if len(names) == 1 else None}
            for speaker, names in ((s, evidence[s]) for s in speakers)]
