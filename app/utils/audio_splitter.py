from pathlib import Path
import wave


def split_audio(audio_path: str, chunk_length_ms: int = 600_000) -> list[tuple[str, float]]:
    """Split audio into temporary WAVs and return each chunk's global offset."""
    source = Path(audio_path)
    chunks: list[tuple[str, float]] = []
    with wave.open(str(source), "rb") as original:
        params = original.getparams()
        if params.nframes <= 0 or params.framerate <= 0:
            raise ValueError("Audio is empty")
        chunk_frames = max(1, int(params.framerate * chunk_length_ms / 1000))
        offset = 0
        while offset < params.nframes:
            frames = original.readframes(chunk_frames)
            if not frames:
                break
            path = source.with_name(f"{source.stem}.chunk-{offset}.wav")
            with wave.open(str(path), "wb") as chunk:
                chunk.setparams(params)
                chunk.writeframes(frames)
            chunks.append((str(path), offset / params.framerate))
            offset += chunk_frames
    return chunks
