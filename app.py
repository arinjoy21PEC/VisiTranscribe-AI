from fastapi import FastAPI, UploadFile, File
from src.visi_transcribe.pipelines.image_captioning_pipeline import run_image_captioning
from src.visi_transcribe.pipelines.translation_pipeline import run_translator
from typing import List, Optional
import yaml

from src.visi_transcribe.routes.vqa_routes import router as vqa_router

app = FastAPI(
    title="VisiTranscribe-AI",
    version="1.0"
)

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

@app.post("/image-to-text/")
async def image_to_text(files: List[UploadFile] = File(...)):
    results = run_image_captioning(files, params['image_captioning_model'])
    return {"results": results}

@app.post("/translate/")
async def translate(texts: List[str], target_lang: Optional[str] = None):
    translations = run_translator(
        texts,
        target_lang,
        params['translation_model'],
        params['language_detection_model']
    )
    return {"translated_text": translations}

app.include_router(vqa_router, prefix="/api", tags=["Visual Question Answering"])