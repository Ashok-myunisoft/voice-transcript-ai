from functools import lru_cache
import logging
import re
from app.config import (
    MODEL_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
)
from app.utils.audio_splitter import split_audio

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model():
    from faster_whisper import WhisperModel
    return WhisperModel(MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)


def _is_repetition_loop(text: str) -> bool:
    """Drop long segments that consist mostly of one repeated token."""
    tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    if len(tokens) < 8:
        return False
    return max(tokens.count(token) for token in set(tokens)) / len(tokens) >= 0.65


def transcribe_audio(audio_path: str) -> list[dict]:
    selected_language = WHISPER_LANGUAGE
    selected_language = selected_language.strip() or None
    if selected_language and selected_language.strip().lower() in {"auto", "detect"}:
        selected_language = None
    logger.info(
        "Transcribing with faster-whisper model=%s device=%s compute_type=%s language=%s",
        MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
        selected_language or "auto",
    )
    model = _get_model()
    chunks = split_audio(audio_path)
    result: list[dict] = []
    try:
        for chunk_path, offset in chunks:
            segments, _ = model.transcribe(
                chunk_path,
                language=selected_language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
                compression_ratio_threshold=1.8,
                log_prob_threshold=-0.7,
                no_speech_threshold=0.5,
                word_timestamps=True,
            )
            for segment in segments:
                text = segment.text.strip()
                if text and not _is_repetition_loop(text):
                    words = []
                    for word in getattr(segment, "words", None) or []:
                        word_text = getattr(word, "word", "")
                        if not word_text.strip():
                            continue
                        words.append({
                            "start": round(float(word.start) + offset, 3),
                            "end": round(float(word.end) + offset, 3),
                            "text": word_text,
                        })
                    result.append({"start": round(float(segment.start) + offset, 2),
                                   "end": round(float(segment.end) + offset, 2),
                                   "text": text, "words": words})
                elif text:
                    logger.warning(
                        "Dropping repetitive Whisper segment at %.2f-%.2f seconds",
                        float(segment.start) + offset,
                        float(segment.end) + offset,
                    )
    finally:
        for chunk_path, _ in chunks:
            try:
                __import__("pathlib").Path(chunk_path).unlink(missing_ok=True)
            except OSError:
                pass
    result.sort(key=lambda item: (item["start"], item["end"]))
    return result
