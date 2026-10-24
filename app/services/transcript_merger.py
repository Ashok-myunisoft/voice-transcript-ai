import logging
import re

logger = logging.getLogger(__name__)


def _normalized_text(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.casefold(), flags=re.UNICODE)


def _speaker_for_interval(start: float, end: float, turns: list[dict]) -> str | None:
    scores: dict[str, float] = {}
    for turn in turns:
        overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if overlap > 0:
            speaker_id = turn["speaker_id"]
            scores[speaker_id] = scores.get(speaker_id, 0.0) + overlap
    if scores:
        speaker_id = max(scores, key=scores.get)
        duration = max(end - start, 0.001)
        if scores[speaker_id] / duration >= 0.2:
            return speaker_id

    # Word alignments can be a few milliseconds outside a diarization boundary.
    midpoint = (start + end) / 2
    nearest = min(
        turns,
        key=lambda turn: min(abs(midpoint - turn["start"]), abs(midpoint - turn["end"]))
        if midpoint < turn["start"] or midpoint > turn["end"] else 0.0,
        default=None,
    )
    if nearest:
        distance = min(abs(midpoint - nearest["start"]), abs(midpoint - nearest["end"]))
        inside = nearest["start"] <= midpoint <= nearest["end"]
        if inside or distance <= 0.25:
            return nearest["speaker_id"]
    return None


def _merge_word_transcript(transcript: list[dict], turns: list[dict]) -> list[dict]:
    words = []
    for segment in transcript:
        segment_words = segment.get("words") or []
        for word in segment_words:
            try:
                start, end = float(word["start"]), float(word["end"])
                text = str(word["text"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start and text.strip():
                words.append({"start": start, "end": end, "text": text,
                              "speaker_id": _speaker_for_interval(start, end, turns)})
        if not segment_words and segment.get("text", "").strip():
            words.append({"start": segment["start"], "end": segment["end"],
                          "text": segment["text"],
                          "speaker_id": _speaker_for_interval(segment["start"], segment["end"], turns)})

    if not words:
        return []
    words.sort(key=lambda word: (word["start"], word["end"]))

    utterances = []
    current = None
    for word in words:
        gap = word["start"] - current["end"] if current else 0.0
        same_speaker = current and word["speaker_id"] == current["speaker_id"]
        if current and (not same_speaker or gap > 1.0):
            current["text"] = current["text"].strip()
            if current["text"]:
                utterances.append(current)
            current = None
        if current is None:
            current = {"start": word["start"], "end": word["end"],
                       "text": word["text"], "speaker_id": word["speaker_id"]}
        else:
            current["end"] = word["end"]
            current["text"] += word["text"]
    if current:
        current["text"] = current["text"].strip()
        if current["text"]:
            utterances.append(current)
    return utterances


def merge_transcript(transcript: list[dict], turns: list[dict]) -> list[dict]:
    if any(segment.get("words") for segment in transcript):
        merged = _merge_word_transcript(transcript, turns)
    else:
        merged = []
    if not merged:
        merged = []
        for segment in transcript:
            speaker_id = _speaker_for_interval(segment["start"], segment["end"], turns)
            merged.append({"start": segment["start"], "end": segment["end"],
                           "text": segment["text"], "speaker_id": speaker_id})

    # Whisper can emit the same sentence twice across adjacent decoding windows.
    # Suppress only exact adjacent repeats assigned to the same diarized speaker.
    deduplicated = []
    for segment in merged:
        if deduplicated:
            previous = deduplicated[-1]
            same_text = _normalized_text(previous["text"]) == _normalized_text(segment["text"])
            close_in_time = segment["start"] <= previous["end"] + 0.5
            same_speaker = previous.get("speaker_id") == segment.get("speaker_id")
            if same_text and close_in_time and same_speaker:
                logger.warning(
                    "Dropping duplicate adjacent transcript segment at %.2f-%.2f seconds",
                    segment["start"], segment["end"],
                )
                continue
        deduplicated.append(segment)
    return deduplicated
