from fastapi import FastAPI
from app.routes.transcription_route import router

app = FastAPI(title="AI Transcription Service")

app.include_router(router)