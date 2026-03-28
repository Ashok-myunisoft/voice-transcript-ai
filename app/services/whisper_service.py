from faster_whisper import WhisperModel
from app.utils.audio_splitter import split_audio

model = WhisperModel("base", device="cpu", compute_type="int8")


def transcribe_audio(audio_path):

    chunks = split_audio(audio_path)

    transcript_lines = []

    for chunk in chunks:

        segments, _ = model.transcribe(chunk)

        for segment in segments:

            text = segment.text.strip()

            if text == "":
                continue

            start = round(segment.start, 2)

            line = f"[{start}] {text}"

            transcript_lines.append(line)

    transcript = "\n".join(transcript_lines)

    return transcript