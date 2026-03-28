from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.utils.file_handler import save_audio_file, save_transcript
from app.services.whisper_service import transcribe_audio

router = APIRouter()


def process_audio(audio_path, filename):

    transcript = transcribe_audio(audio_path)

    save_transcript(transcript, filename)


@router.post("/transcribe")
async def transcribe(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
):

    audio_path = save_audio_file(file)

    background_tasks.add_task(
        process_audio,
        audio_path,
        file.filename
    )

    return {
        "message": "Audio uploaded successfully. Transcription started.",
        "file_name": file.filename
    }