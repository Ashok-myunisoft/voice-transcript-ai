import base64
import binascii
import json
import logging
import re
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import API_KEY, MAX_UPLOAD_BYTES
from app.services.action_item_service import extract_action_items
from app.services.meeting_store import load_metadata, meeting_dir, save_metadata
from app.services.meeting_summary_service import summarize_meeting
from app.services.report_generation_service import generate_reports
from app.services.speaker_identity_service import resolve_speakers
from app.services.openai_transcription_service import transcribe_and_diarize
from app.utils.audio_preprocessor import denoise_audio

logger = logging.getLogger(__name__)
router = APIRouter()


class Base64Audio(BaseModel):
    file_name: str = Field(default="meeting.wav", max_length=255)
    audio_format: str = Field(default="wav", max_length=16)
    audio_base64: str = Field(min_length=1, max_length=4 * ((MAX_UPLOAD_BYTES + 2) // 3))


def _authorize(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")


def _check_meeting_id(meeting_id: str) -> None:
    if not re.fullmatch(r"meeting_[0-9a-f]{32}", meeting_id):
        raise HTTPException(404, "Meeting not found")


def _validate_wav(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getnframes() <= 0 or wav.getframerate() <= 0 or wav.getnchannels() <= 0:
                raise ValueError("Audio is empty or invalid")
            return wav.getnframes() / wav.getframerate()
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid WAV audio content") from exc


def _process_meeting(meeting_id: str, audio_path: Path, duration: float) -> dict:
    meta = load_metadata(meeting_id) or {"meeting_id": meeting_id}
    model_audio_path = audio_path
    try:
        meta.update(status="processing", stage="transcription_and_diarization")
        save_metadata(meeting_id, meta)
        model_audio_path = denoise_audio(audio_path)
        transcript, diarized_speaker_count = transcribe_and_diarize(str(model_audio_path))
        # Decoders can occasionally emit an end time beyond the source duration.
        # Keep persisted transcript timestamps within the actual recording.
        bounded_transcript = []
        for segment in transcript:
            try:
                start = max(0.0, min(float(segment["start"]), duration))
                end = max(0.0, min(float(segment["end"]), duration))
            except (KeyError, TypeError, ValueError):
                logger.warning("Dropping transcript segment with invalid timestamps for %s", meeting_id)
                continue
            if end <= start:
                logger.warning("Dropping empty/out-of-range transcript segment for %s", meeting_id)
                continue
            bounded_segment = {**segment, "start": round(start, 2), "end": round(end, 2)}
            if isinstance(segment.get("words"), list):
                bounded_words = []
                for word in segment["words"]:
                    try:
                        word_start = max(0.0, min(float(word["start"]), duration))
                        word_end = max(0.0, min(float(word["end"]), duration))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if word_end > word_start:
                        bounded_words.append({**word, "start": word_start, "end": word_end})
                bounded_segment["words"] = bounded_words
            bounded_transcript.append(bounded_segment)
        transcript = bounded_transcript
        diarization_status = "completed"
        assigned_speaker_count = len({segment.get("speaker_id") for segment in transcript if segment.get("speaker_id")})
        if assigned_speaker_count == 1:
            logger.warning(
                "OpenAI diarization assigned only one speaker ID to %s; verify if the recording has multiple voices",
                meeting_id,
            )
        logger.info(
            "Transcript speaker assignment for %s: %s",
            meeting_id,
            {speaker_id: sum(1 for segment in transcript if segment.get("speaker_id") == speaker_id)
             for speaker_id in sorted({segment.get("speaker_id") for segment in transcript if segment.get("speaker_id")})},
        )
        speakers = resolve_speakers(transcript)
        actions = extract_action_items(transcript, speakers)
        summary = summarize_meeting(transcript, speakers, actions)
        folder = meeting_dir(meeting_id)
        meta["stage"] = "report_generation"
        save_metadata(meeting_id, meta)
        generate_reports(folder, meeting_id, transcript, speakers, summary, actions, duration)
        data = {"meeting_id": meeting_id, "duration_seconds": duration,
                "transcript": transcript, "speakers": speakers, "summary": summary,
                "action_items": actions, "diarization_status": diarization_status,
                "diarization_speaker_count": diarized_speaker_count,
                "assigned_speaker_count": assigned_speaker_count}
        if assigned_speaker_count == 1:
            data["diarization_warning"] = (
                "OpenAI diarization assigned one speaker. If this recording has multiple voices, "
                "review the speaker labels."
            )
        (folder / "meeting_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        meta.update(status="completed", stage="completed", completed_at=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=duration, diarization_status=diarization_status)
        save_metadata(meeting_id, meta)
        return data
    except Exception as exc:
        logger.exception("Meeting processing failed: %s", meeting_id)
        meta.update(status="failed", stage=meta.get("stage"), error=f"Processing failed ({type(exc).__name__}). Check server logs.")
        save_metadata(meeting_id, meta)
        raise HTTPException(
            status_code=500,
            detail=f"Meeting processing failed ({type(exc).__name__}). Check server logs.",
        ) from exc
    finally:
        save_metadata(meeting_id, meta)
        try:
            audio_path.with_name(f"{audio_path.stem}.denoised.wav").unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary denoised audio for %s", meeting_id)
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary audio for %s", meeting_id)


def _create_meeting(
    audio_bytes: bytes,
    original_name: str,
) -> dict:
    if not audio_bytes:
        raise HTTPException(400, "Audio is empty")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio exceeds the configured upload limit")
    meeting_id = f"meeting_{uuid.uuid4().hex}"
    folder = meeting_dir(meeting_id)
    folder.mkdir(parents=True, exist_ok=False)
    audio_path = folder / "audio.wav"
    audio_path.write_bytes(audio_bytes)
    try:
        duration = _validate_wav(audio_path)
    except ValueError as exc:
        audio_path.unlink(missing_ok=True)
        folder.rmdir()
        raise HTTPException(415, str(exc)) from exc
    metadata = {"meeting_id": meeting_id, "status": "processing", "stage": "queued",
                "original_file_name": Path(original_name).name,
                "created_at": datetime.now(timezone.utc).isoformat()}
    save_metadata(meeting_id, metadata)
    return _process_meeting(meeting_id, audio_path, duration)


@router.post("/transcribe", dependencies=[Depends(_authorize)])
def transcribe(
    file: UploadFile = File(...),
):
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    return _create_meeting(raw, file.filename or "meeting.wav")


@router.post("/transcribe/base64", dependencies=[Depends(_authorize)])
def transcribe_base64(payload: Base64Audio):
    if payload.audio_format.lower() not in {"wav", "wave"} or Path(payload.file_name).suffix.lower() not in {".wav", ".wave"}:
        raise HTTPException(415, "Only WAV audio is currently supported")
    if len(payload.audio_base64) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3):
        raise HTTPException(413, "Audio exceeds the configured upload limit")
    try:
        raw = base64.b64decode(payload.audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(400, "Malformed Base64 audio") from exc
    return _create_meeting(raw, payload.file_name)


@router.get("/meetings/{meeting_id}/status", dependencies=[Depends(_authorize)])
def status(meeting_id: str):
    _check_meeting_id(meeting_id)
    meta = load_metadata(meeting_id)
    if not meta:
        raise HTTPException(404, "Meeting not found")
    response = {k: meta[k] for k in ("meeting_id", "status", "stage", "error", "duration_seconds", "diarization_status") if k in meta}
    if meta["status"] == "completed":
        response["downloads"] = {"transcript": f"/meetings/{meeting_id}/transcript/download",
                                  "report": f"/meetings/{meeting_id}/report/download"}
    return response


def _download(meeting_id: str, filename: str, media_type: str):
    _check_meeting_id(meeting_id)
    meta = load_metadata(meeting_id)
    if not meta:
        raise HTTPException(404, "Meeting not found")
    if meta["status"] != "completed":
        raise HTTPException(409, "Meeting processing is not complete")
    path = meeting_dir(meeting_id) / filename
    if not path.is_file():
        raise HTTPException(404, "Download is unavailable")
    return FileResponse(path, media_type=media_type, filename=filename)


@router.get("/meetings/{meeting_id}/transcript/download", dependencies=[Depends(_authorize)])
def download_transcript(meeting_id: str):
    return _download(meeting_id, "transcript.txt", "text/plain; charset=utf-8")


@router.get("/meetings/{meeting_id}/report/download", dependencies=[Depends(_authorize)])
def download_report(meeting_id: str):
    return _download(meeting_id, "meeting_report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


class SpeakerCorrection(BaseModel):
    name: str = Field(min_length=1, max_length=100)


@router.patch("/meetings/{meeting_id}/speakers/{speaker_id}", dependencies=[Depends(_authorize)])
def correct_speaker(meeting_id: str, speaker_id: str, payload: SpeakerCorrection):
    _check_meeting_id(meeting_id)
    meta = load_metadata(meeting_id)
    if not meta or meta.get("status") != "completed":
        raise HTTPException(404, "Completed meeting not found")
    folder = meeting_dir(meeting_id)
    path = folder / "meeting_data.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    speaker = next((s for s in data["speakers"] if s["speaker_id"] == speaker_id), None)
    if not speaker:
        raise HTTPException(404, "Speaker not found")
    speaker.update(name=payload.name.strip(), identity_status="user_corrected", evidence="User correction", confidence=1.0)
    for action in data["action_items"]:
        if action.get("owner_speaker_id") == speaker_id:
            action["owner_name"] = speaker["name"]
    data["summary"]["participants"] = data["speakers"]
    data["summary"]["action_items"] = data["action_items"]
    generate_reports(folder, meeting_id, data["transcript"], data["speakers"], data["summary"], data["action_items"], data["duration_seconds"])
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return speaker
