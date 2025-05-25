from fastapi import FastAPI
from src.visi_transcribe.routes.vqa_routes import router as vqa_router
from src.visi_transcribe.routes.image_to_text_routes import router as ic_router
from src.visi_transcribe.routes.translate_routes import router as translation_router

app = FastAPI(
    title="VisiTranscribe-AI",
    version="1.0"
)

app.include_router(ic_router, prefix="/api", tags=["Image Captioning"])
app.include_router(translation_router, prefix="/api", tags=["Translation"])
app.include_router(vqa_router, prefix="/api", tags=["Visual Question Answering"])