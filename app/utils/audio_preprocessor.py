"""Optional speech noise reduction using FFmpeg's FFT denoiser."""
import logging
import shutil
import subprocess
from pathlib import Path

from app.config import AUDIO_DENOISE_ENABLED, AUDIO_DENOISE_REDUCTION_DB

logger = logging.getLogger(__name__)


def denoise_audio(audio_path: str | Path) -> Path:
    """Create a denoised WAV beside the source, or return the source on failure.

    The original audio is kept untouched. FFmpeg is optional for local runs; the
    Docker image installs it. The caller owns cleanup of the generated WAV.
    """
    source = Path(audio_path)
    if not AUDIO_DENOISE_ENABLED:
        return source

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg = get_ffmpeg_exe()
        except (ImportError, RuntimeError, OSError):
            ffmpeg = None
    if not ffmpeg:
        logger.warning("Audio denoising is enabled, but no FFmpeg executable was found; using the original audio")
        return source

    reduction = min(97.0, max(0.01, AUDIO_DENOISE_REDUCTION_DB))
    output = source.with_name(f"{source.stem}.denoised.wav")
    command = [
        ffmpeg,
        "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-af", f"afftdn=nr={reduction:g}:nf=-50:tn=1",
        "-c:a", "pcm_s16le",
        str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
        if output.is_file() and output.stat().st_size > 44:
            logger.info("Applied FFmpeg afftdn noise reduction (%s dB) to %s", reduction, source.name)
            return output
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Audio denoising failed for %s; using original audio: %s", source.name, exc)
    try:
        output.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove incomplete denoised WAV %s", output.name)
    return source
