import base64
import io
import shutil
import uuid
import wave
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.routes import transcription_route as route
from app.services.speaker_identity_service import resolve_speakers
from app.services.transcript_merger import merge_transcript
from app.utils.audio_splitter import split_audio


@pytest.fixture
def tmp_path():
    # Keep test writes in the repository rather than pytest's restricted system temp directory.
    folder = Path.cwd() / "app" / "storage" / f"test-{uuid.uuid4().hex}"
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        shutil.rmtree(folder)


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 8000)
    return buffer.getvalue()


def test_transcript_merger_uses_overlap_and_keeps_timestamps():
    segments = [{"start": 1.0, "end": 3.0, "text": "hello"}]
    turns = [{"start": 0.8, "end": 2.8, "speaker_id": "SPEAKER_00"}]
    assert merge_transcript(segments, turns) == [{**segments[0], "speaker_id": "SPEAKER_00"}]


def test_splitter_reports_global_offsets_and_cleans_chunks(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes())
    chunks = split_audio(str(source), chunk_length_ms=250)
    assert [offset for _, offset in chunks] == [0.0, 0.25, 0.5, 0.75]
    for path, _ in chunks:
        Path(path).unlink()


def test_identity_requires_self_introduction_from_same_speaker():
    segments = [
        {"speaker_id": "SPEAKER_00", "text": "I'm Ashok."},
        {"speaker_id": "SPEAKER_01", "text": "Ashok will present."},
    ]
    result = resolve_speakers(segments)
    assert result[0]["name"] == "Ashok"
    assert result[1]["name"] is None


def test_wav_upload_and_downloads(monkeypatch, tmp_path):
    import app.services.meeting_store as store
    monkeypatch.setattr(store, "MEETINGS_DIR", tmp_path / "meetings")
    monkeypatch.setattr(route, "transcribe_audio", lambda _: [{"start": 0.0, "end": 0.5, "text": "Hello"}])
    monkeypatch.setattr(route, "diarize_audio", lambda *_, **__: ([], "unavailable"))
    monkeypatch.setattr(route, "extract_action_items", lambda *_: [])
    monkeypatch.setattr(route, "summarize_meeting", lambda *_: {
        "overview": "Test", "key_discussion_points": [], "decisions": [],
        "action_items": [], "pending_questions": [], "participants": []})
    client = TestClient(FastAPI())
    client.app.include_router(route.router)
    response = client.post("/transcribe", files={"file": ("test.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    meeting_id = response.json()["meeting_id"]
    assert client.get(f"/meetings/{meeting_id}/status").json()["status"] == "completed"
    transcript = client.get(f"/meetings/{meeting_id}/transcript/download")
    report = client.get(f"/meetings/{meeting_id}/report/download")
    assert "Hello" in transcript.text
    assert report.headers["content-type"].startswith("application/vnd.openxmlformats-officedocument")
    assert report.content[:2] == b"PK"


def test_base64_rejects_malformed_audio():
    app = FastAPI()
    app.include_router(route.router)
    response = TestClient(app).post("/transcribe/base64", json={"audio_base64": "not base64!", "file_name": "m.wav"})
    assert response.status_code == 400


def test_base64_rejects_non_wav_bytes():
    app = FastAPI()
    app.include_router(route.router)
    response = TestClient(app).post("/transcribe/base64", json={
        "file_name": "meeting.wav", "audio_format": "wav", "audio_base64": base64.b64encode(b"not wav").decode()
    })
    assert response.status_code == 415


def test_runpod_diarization_posts_audio_and_normalizes_output(monkeypatch, tmp_path):
    import app.services.diarization_service as service

    class FakeResponse:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self.data

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, headers, json):
            assert url.endswith("/run")
            assert "audio_base64" in json["input"]
            return FakeResponse({"id": "job-1", "status": "IN_QUEUE"})

        def get(self, url, headers):
            assert url.endswith("/status/job-1")
            return FakeResponse({"status": "COMPLETED", "output": {"segments": [
                {"start": 1.25, "end": 2.5, "speaker": "speaker_1"}
            ]}})

    monkeypatch.setattr(service, "RUNPOD_ENDPOINT_URL", "https://api.runpod.ai/v2/test")
    monkeypatch.setattr(service, "RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(service.httpx, "Client", FakeClient)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(wav_bytes())
    turns, status = service._runpod_diarize(str(audio_path))
    assert status == "completed"
    assert turns == [{"start": 1.25, "end": 2.5, "speaker_id": "SPEAKER_00"}]


def test_base64_wav_upload(monkeypatch, tmp_path):
    import app.services.meeting_store as store
    monkeypatch.setattr(store, "MEETINGS_DIR", tmp_path / "meetings")
    monkeypatch.setattr(route, "transcribe_audio", lambda _: [])
    monkeypatch.setattr(route, "diarize_audio", lambda *_, **__: ([], "unavailable"))
    monkeypatch.setattr(route, "extract_action_items", lambda *_: [])
    monkeypatch.setattr(route, "summarize_meeting", lambda *_: {
        "overview": "Test", "key_discussion_points": [], "decisions": [],
        "action_items": [], "pending_questions": [], "participants": []})
    app = FastAPI()
    app.include_router(route.router)
    response = TestClient(app).post("/transcribe/base64", json={
        "file_name": "meeting.wav", "audio_format": "wav", "audio_base64": base64.b64encode(wav_bytes()).decode()
    })
    assert response.status_code == 200
    assert response.json()["status"] == "processing"

