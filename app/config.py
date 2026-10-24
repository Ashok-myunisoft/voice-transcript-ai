import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:
    pass

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "app" / "storage")).resolve()
MEETINGS_DIR = STORAGE_DIR / "meetings"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
# Retained for compatibility with the unused legacy transcription/RunPod modules.
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en").strip() or "en"
AUDIO_DENOISE_ENABLED = os.getenv("AUDIO_DENOISE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
AUDIO_DENOISE_REDUCTION_DB = float(os.getenv("AUDIO_DENOISE_REDUCTION_DB", "10"))
# Retained for compatibility; the upload processing route no longer calls RunPod.
RUNPOD_ENDPOINT_URL = os.getenv("RUNPOD_ENDPOINT_URL", "").rstrip("/")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_MAX_AUDIO_BYTES = int(os.getenv("RUNPOD_MAX_AUDIO_BYTES", "7000000"))
RUNPOD_JOB_TIMEOUT_SECONDS = int(os.getenv("RUNPOD_JOB_TIMEOUT_SECONDS", "1800"))
RUNPOD_POLL_INTERVAL_SECONDS = float(os.getenv("RUNPOD_POLL_INTERVAL_SECONDS", "3"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
OPENAI_TRANSCRIPTION_MODEL = "gpt-transcribe"
OPENAI_DIARIZATION_MODEL = "gpt-4o-transcribe-diarize"
OPENAI_TRANSCRIPTION_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TRANSCRIPTION_TIMEOUT_SECONDS", "900"))
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
API_KEY = os.getenv("API_KEY")
