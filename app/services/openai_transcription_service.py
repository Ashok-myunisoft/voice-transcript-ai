"""Transcribe meetings and stabilize speaker labels through OpenAI's Audio API."""
import base64
import difflib
import io
import logging
import shutil
import subprocess
import wave
from pathlib import Path

import httpx

from app.config import (
    OPENAI_API_KEY,
    OPENAI_DIARIZATION_MODEL,
    OPENAI_TRANSCRIPTION_MODEL,
    OPENAI_TRANSCRIPTION_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

API_URL = "https://api.openai.com/v1/audio/transcriptions"
MAX_API_FILE_BYTES = 25 * 1024 * 1024


def _compressed_upload(path: Path) -> Path | None:
    """Compress oversized WAVs to MP3 so they fit the Audio API upload limit."""
    if path.stat().st_size <= MAX_API_FILE_BYTES:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
        except (ImportError, RuntimeError, OSError):
            ffmpeg = None
    if not ffmpeg:
        raise RuntimeError("Audio exceeds OpenAI's 25 MB upload limit and FFmpeg is unavailable for compression")

    compressed = path.with_name(f"{path.stem}.openai.mp3")
    try:
        subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(path), "-codec:a", "libmp3lame", "-b:a", "96k", str(compressed)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if not compressed.is_file() or compressed.stat().st_size > MAX_API_FILE_BYTES:
            raise RuntimeError("Compressed audio still exceeds OpenAI's 25 MB upload limit")
        return compressed
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        compressed.unlink(missing_ok=True)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Could not compress audio for OpenAI upload: {exc}") from exc


def _post_transcription(path: Path, form: dict | list[tuple[str, str]]) -> dict:
    timeout = httpx.Timeout(OPENAI_TRANSCRIPTION_TIMEOUT_SECONDS, connect=30.0)
    compressed_path = _compressed_upload(path)
    upload_path = compressed_path or path
    try:
        # HTTPX's ``data`` argument expects a mapping. A list of pairs is
        # especially needed here because the reference-assisted request may
        # contain repeated field names, so encode all fields as multipart parts.
        fields = form.items() if isinstance(form, dict) else form
        multipart = [(key, (None, value)) for key, value in fields]
        with upload_path.open("rb") as audio_file, httpx.Client(timeout=timeout) as client:
            multipart.append(("file", (upload_path.name, audio_file, "audio/mpeg" if compressed_path else "audio/wav")))
            response = client.post(
                API_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files=multipart,
            )
    finally:
        if compressed_path:
            compressed_path.unlink(missing_ok=True)
    if response.is_error:
        raise RuntimeError(f"OpenAI transcription request failed ({response.status_code}): {response.text[:2000]}")
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("OpenAI returned an invalid JSON transcription response") from exc


def _parse_diarized(payload: dict, duration: float) -> tuple[list[dict], list[dict]]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise RuntimeError("OpenAI diarized response did not include a segments list")

    speaker_ids: dict[str, str] = {}
    transcript = []
    turns = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, float(item["start"]))
            end = min(duration, float(item["end"]))
            text = str(item.get("text", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue

        source_speaker = item.get("speaker")
        if source_speaker is None or not str(source_speaker).strip():
            speaker_id = None
        else:
            source_speaker = str(source_speaker).strip()
            if source_speaker.startswith("SPEAKER_"):
                speaker_id = source_speaker
                speaker_ids.setdefault(source_speaker, speaker_id)
            else:
                if source_speaker not in speaker_ids:
                    speaker_ids[source_speaker] = f"SPEAKER_{len(speaker_ids):02d}"
                speaker_id = speaker_ids[source_speaker]
        turns.append({"start": start, "end": end, "speaker_id": speaker_id})
        if text:
            transcript.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
                "speaker_id": speaker_id,
            })
    return transcript, turns


def _speaker_reference_samples(path: Path, turns: list[dict]) -> list[tuple[str, str]]:
    """Create 2–8 second WAV samples from the longest detected turn per speaker."""
    candidates: dict[str, dict] = {}
    for turn in turns:
        speaker_id = turn.get("speaker_id")
        turn_duration = float(turn["end"]) - float(turn["start"])
        if not speaker_id or turn_duration < 2.0:
            continue
        if speaker_id not in candidates or turn_duration > candidates[speaker_id]["duration"]:
            candidates[speaker_id] = {**turn, "duration": turn_duration}

    selected = sorted(candidates.items())[:4]
    if len(selected) < 2:
        return []

    samples = []
    try:
        with wave.open(str(path), "rb") as source:
            source_channels = source.getnchannels()
            source_sample_width = source.getsampwidth()
            source_rate = source.getframerate()
            for speaker_id, turn in selected:
                clip_duration = min(8.0, turn["duration"])
                clip_start = turn["start"] + max(0.0, (turn["duration"] - clip_duration) / 2)
                first_frame = max(0, int(clip_start * source_rate))
                source.setpos(first_frame)
                frames = source.readframes(int(clip_duration * source_rate))
                sample_buffer = io.BytesIO()
                with wave.open(sample_buffer, "wb") as sample:
                    sample.setnchannels(source_channels)
                    sample.setsampwidth(source_sample_width)
                    sample.setframerate(source_rate)
                    sample.writeframes(frames)
                encoded = base64.b64encode(sample_buffer.getvalue()).decode("ascii")
                samples.append((speaker_id, f"data:audio/wav;base64,{encoded}"))
    except (OSError, wave.Error, EOFError) as exc:
        logger.warning("Could not prepare speaker reference clips: %s", exc)
        return []
    return samples


def _apply_refined_speakers(transcript: list[dict], turns: list[dict]) -> list[dict]:
    """Use reference-assisted labels while preserving the first transcript's words and times."""
    refined = []
    for segment in transcript:
        duration = max(float(segment["end"]) - float(segment["start"]), 0.001)
        overlap_by_speaker: dict[str, float] = {}
        for turn in turns:
            speaker_id = turn.get("speaker_id")
            if not speaker_id:
                continue
            overlap = max(
                0.0,
                min(float(segment["end"]), float(turn["end"]))
                - max(float(segment["start"]), float(turn["start"])),
            )
            if overlap:
                overlap_by_speaker[speaker_id] = overlap_by_speaker.get(speaker_id, 0.0) + overlap
        if overlap_by_speaker:
            candidate = max(overlap_by_speaker, key=overlap_by_speaker.get)
            if overlap_by_speaker[candidate] / duration >= 0.5:
                refined.append({**segment, "speaker_id": candidate})
                continue
        refined.append(segment)
    return refined


def _normalize_word(word: str) -> str:
    return "".join(character for character in word.casefold() if character.isalnum())


def _apply_transcription_text(
    text: str,
    diarized_segments: list[dict],
    duration: float,
) -> list[dict]:
    """Put the transcription model's words into the diarization model's timed turns.

    The transcription endpoint returns text without timestamps. Aligning the two
    outputs by their shared words lets the diarization model remain authoritative
    for timing and speaker IDs, while the transcription model supplies wording.
    """
    output_words = text.split()
    reference_words: list[str] = []
    reference_segments: list[int] = []
    for segment_index, segment in enumerate(diarized_segments):
        for word in str(segment.get("text", "")).split():
            normalized = _normalize_word(word)
            if normalized:
                reference_words.append(normalized)
                reference_segments.append(segment_index)

    normalized_output = [_normalize_word(word) for word in output_words]
    output_indices = [index for index, word in enumerate(normalized_output) if word]
    reference_indices = [index for index, word in enumerate(reference_words) if word]
    matcher = difflib.SequenceMatcher(
        None,
        [normalized_output[index] for index in output_indices],
        [reference_words[index] for index in reference_indices],
        autojunk=False,
    )

    # Anchors map output word positions to the nearest matching diarized word.
    anchors: dict[int, int] = {}
    matched = 0
    for output_start, reference_start, size in matcher.get_matching_blocks():
        matched += size
        for offset in range(size):
            anchors[output_indices[output_start + offset]] = reference_indices[reference_start + offset]

    if not output_words:
        return diarized_segments
    if not reference_words:
        return [{"start": 0.0, "end": round(duration, 2), "text": text.strip(), "speaker_id": None}]
    if not anchors:
        logger.warning("Transcription and diarization text had no matching words; keeping diarization transcript")
        return diarized_segments
    if matched / max(1, len(output_indices)) < 0.2:
        logger.warning("Transcription and diarization text alignment was weak; keeping diarization transcript")
        return diarized_segments

    anchor_positions = sorted(anchors)
    words_by_segment: dict[int, list[str]] = {}
    for output_index, word in enumerate(output_words):
        if output_index in anchors:
            reference_index = anchors[output_index]
        else:
            left = max((position for position in anchor_positions if position < output_index), default=None)
            right = min((position for position in anchor_positions if position > output_index), default=None)
            if left is not None and right is not None:
                fraction = (output_index - left) / (right - left)
                reference_index = round(anchors[left] + fraction * (anchors[right] - anchors[left]))
            elif left is not None:
                reference_index = anchors[left] + output_index - left
            elif right is not None:
                reference_index = anchors[right] - (right - output_index)
            else:
                reference_index = 0
        reference_index = min(len(reference_words) - 1, max(0, reference_index))
        segment_index = reference_segments[reference_index]
        words_by_segment.setdefault(segment_index, []).append(word)

    aligned = []
    for segment_index, segment in enumerate(diarized_segments):
        words = words_by_segment.get(segment_index)
        if words:
            aligned.append({**segment, "text": " ".join(words)})
    return aligned or diarized_segments


def transcribe_and_diarize(audio_path: str) -> tuple[list[dict], int]:
    """Transcribe words with GPT Transcribe and assign turns from the diarization model."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for transcription and diarization")

    path = Path(audio_path)
    with wave.open(str(path), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()

    transcription_form: list[tuple[str, str]] = [
        ("model", OPENAI_TRANSCRIPTION_MODEL),
        ("response_format", "json"),
        ("temperature", "0.2"),
    ]
    logger.info("Transcribing with OpenAI model=%s duration=%.2fs", OPENAI_TRANSCRIPTION_MODEL, duration)
    transcription_payload = _post_transcription(path, transcription_form)
    transcription_text = transcription_payload.get("text")
    if not isinstance(transcription_text, str) or not transcription_text.strip():
        raise RuntimeError("OpenAI transcription model returned no transcript text")

    diarization_form: list[tuple[str, str]] = [
        ("model", OPENAI_DIARIZATION_MODEL),
        ("response_format", "diarized_json"),
    ]
    if duration > 30:
        diarization_form.append(("chunking_strategy", "auto"))

    logger.info("Diarizing with OpenAI model=%s duration=%.2fs", OPENAI_DIARIZATION_MODEL, duration)
    first_payload = _post_transcription(path, diarization_form)
    diarized_transcript, first_turns = _parse_diarized(first_payload, duration)
    first_speaker_count = len({turn["speaker_id"] for turn in first_turns if turn.get("speaker_id")})
    if first_speaker_count < 2 or not diarized_transcript:
        return _apply_transcription_text(transcription_text, diarized_transcript, duration), first_speaker_count

    references = _speaker_reference_samples(path, first_turns)
    if len(references) < 2:
        logger.info("Not enough clean speaker turns for reference-based diarization; keeping first result")
        return _apply_transcription_text(transcription_text, diarized_transcript, duration), first_speaker_count

    reference_form = [
        ("model", OPENAI_DIARIZATION_MODEL),
        ("response_format", "diarized_json"),
    ]
    if duration > 30:
        reference_form.append(("chunking_strategy", "auto"))
    for speaker_id, reference in references:
        reference_form.append(("known_speaker_names[]", speaker_id))
        reference_form.append(("known_speaker_references[]", reference))

    try:
        logger.info("Refining speaker consistency using %d voice references", len(references))
        refined_payload = _post_transcription(path, reference_form)
        refined_transcript, refined_turns = _parse_diarized(refined_payload, duration)
        refined_speaker_count = len({turn["speaker_id"] for turn in refined_turns if turn.get("speaker_id")})
        if refined_transcript and refined_speaker_count >= 2:
            # The reference pass refines only the speaker turns. Text still comes
            # from gpt-transcribe and is aligned to those turns below.
            diarization_transcript = _apply_refined_speakers(diarized_transcript, refined_turns)
            aligned = _apply_transcription_text(transcription_text, diarization_transcript, duration)
            return aligned, refined_speaker_count
        logger.warning("Reference diarization returned incomplete labels; keeping first result")
    except Exception as exc:
        logger.warning("Reference diarization failed; keeping first result: %s", exc)

    return _apply_transcription_text(transcription_text, diarized_transcript, duration), first_speaker_count
