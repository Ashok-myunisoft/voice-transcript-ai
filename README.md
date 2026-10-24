# Voice Transcript AI

FastAPI service for WAV transcription, optional speaker diarization, transcript-grounded meeting analysis, and downloadable TXT/DOCX reports. Upload requests return the completed meeting JSON after processing finishes.

## Run locally

Use Python 3.12 (the Docker image uses Python 3.12):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python run.py
```

The API listens on `http://127.0.0.1:8000`; Swagger UI is at `/docs`. Configure settings in the process environment (or load `.env` with your preferred environment loader). Set `API_KEY` to protect all endpoints, then send it as `X-API-Key`. If it is unset, API key checks are disabled for local development.

## Upload audio

WAV multipart upload:

```powershell
curl.exe -X POST http://127.0.0.1:8000/transcribe `
  -H "X-API-Key: $env:API_KEY" `
  -F "file=@meeting.wav;type=audio/wav"
```

Base64 JSON upload (the data field must contain the Base64 encoding of the WAV bytes):

```powershell
$audio = [Convert]::ToBase64String([IO.File]::ReadAllBytes("meeting.wav"))
$body = @{
  file_name = "meeting.wav"
  audio_format = "wav"
  audio_base64 = $audio
} | ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:8000/transcribe/base64 -Method Post `
  -Headers @{ "X-API-Key" = $env:API_KEY } -ContentType "application/json" -Body $body
```

The POST request stays open while the audio is processed and returns the full meeting JSON directly:

```json
{
  "meeting_id": "meeting_0123456789abcdef0123456789abcdef",
  "duration_seconds": 183.2,
  "transcript": [],
  "speakers": [],
  "summary": {},
  "action_items": [],
  "diarization_status": "completed"
}
```

The same JSON is also saved in the meeting folder. Allow a sufficiently long client and proxy timeout for the upload request, since transcription, diarization, and summary generation complete before the response is sent.

Optional generated files remain downloadable with:

```text
GET /meetings/{meeting_id}/transcript/download
GET /meetings/{meeting_id}/report/download
```

Speaker corrections use `PATCH /meetings/{meeting_id}/speakers/{speaker_id}` with JSON `{"name":"Ashok"}`; the transcript and report are regenerated.

## Optional models and settings

| Variable | Purpose |
| --- | --- |
| `MAX_UPLOAD_BYTES` | Upload limit; defaults to 100 MiB |
| `STORAGE_DIR` | Persistent meeting storage root |
| `OPENAI_API_KEY` | OpenAI API key for transcription, diarization, and meeting analysis |
| `OPENAI_TRANSCRIPTION_TIMEOUT_SECONDS` | Request timeout for the transcription and diarization call; defaults to 900 seconds |
| `AUDIO_DENOISE_ENABLED`, `AUDIO_DENOISE_REDUCTION_DB` | Run FFmpeg's `afftdn` noise reduction before transcription and diarization; defaults to enabled at 10 dB |
| `API_KEY` | Shared API key; unset means local development without authentication |
| `LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL` | Endpoint overrides; default model is `gpt-4.1-mini` |

The app sends the denoised WAV to `gpt-transcribe` for the transcript, then to `gpt-4o-transcribe-diarize` for speaker turns. The transcription request uses a low temperature to reduce run-to-run variation. Transcript words are aligned in order with the diarization model's segment text; the diarization model supplies timestamps and speaker IDs. If the two transcripts do not align reliably, the app keeps the diarization model's transcript rather than risking incorrect speaker assignments. For recordings longer than 30 seconds, the diarization request uses automatic server-side chunking. When the diarization result identifies at least two speakers with turns long enough for reference clips, the app can make an additional diarization request using short audio samples as known-speaker references. Set `OPENAI_API_KEY` in the backend environment; transcription and diarization fail clearly if it is missing. Speaker labels are anonymous; names are assigned only after the app can resolve identity or a user corrects them.

OpenAI currently lists `gpt-4o-transcribe-diarize` as deprecated, with shutdown scheduled for February 26, 2027. This integration follows the requested model and will need migration before that date.

Both upload endpoints accept WAV audio. Language is inferred automatically. No recording-specific vocabulary or prompt is hardcoded into the transcription request.

With denoising enabled, the app makes a temporary cleaned WAV before sending it to OpenAI. The uploaded audio remains untouched; the temporary copy is removed after processing. The Docker image installs FFmpeg, and local Python installs use the bundled `imageio-ffmpeg` executable if FFmpeg is not on `PATH`. If neither executable is available, processing falls back to the original audio and logs a warning. Denoising can affect quiet speech, so set `AUDIO_DENOISE_ENABLED=false` or lower `AUDIO_DENOISE_REDUCTION_DB` if it reduces speech clarity.

## Tests

```powershell
python -m pytest tests -q
```

The automated suite uses synthetic WAV audio and mocked recognition/diarization. For model integration checks, use a short WAV recording and configure the corresponding model credentials.

## Docker

```powershell
docker build -t voice-transcript-ai .
docker run --rm -p 8000:8000 -e API_KEY="change-me" -v "${PWD}/meeting-storage:/app/app/storage" voice-transcript-ai
```

The background task runs in the API process and is not a durable queue. For production or multi-worker deployments, move processing to a persistent task queue and place storage on a shared persistent volume. Configure API authentication and restrict network access before exposing the service.


