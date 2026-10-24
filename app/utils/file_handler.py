import os
from pathlib import Path

AUDIO_FOLDER = "storage/audio"
TRANSCRIPT_FOLDER = "storage/transcripts"


def save_audio_file(upload_file):

    file_path = os.path.join(AUDIO_FOLDER, upload_file.filename)

    with open(file_path, "wb") as buffer:
        while True:
            chunk = upload_file.file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            buffer.write(chunk)

    return file_path


def save_transcript(transcript_text, audio_filename):

    base_name = Path(audio_filename).stem
    transcript_file = f"{base_name}.txt"

    transcript_path = os.path.join(TRANSCRIPT_FOLDER, transcript_file)

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    return transcript_path