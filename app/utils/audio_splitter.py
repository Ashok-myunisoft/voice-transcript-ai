from pydub import AudioSegment
import os

def split_audio(audio_path, chunk_length_ms=600000):
    """
    Split audio into chunks
    default = 10 minutes
    """

    audio = AudioSegment.from_file(audio_path)

    chunks = []

    for i in range(0, len(audio), chunk_length_ms):

        chunk = audio[i:i + chunk_length_ms]

        chunk_name = f"{audio_path}_chunk_{i}.wav"

        chunk.export(chunk_name, format="wav")

        chunks.append(chunk_name)

    return chunks