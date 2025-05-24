from fastapi import FastAPI, UploadFile, File
from src.visi_transcribe.pipelines.image_captioning_pipeline import run_image_captioning
from src.visi_transcribe.pipelines.translation_pipeline import run_translator
from typing import List
import yaml

app = FastAPI()

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

@app.post("/image-to-text/")
async def image_to_text(files: List[UploadFile] = File(...)):
    results = run_image_captioning(files, params['image_captioning_model'])
    return {"results": results}

@app.post("/translate/")
async def translate(text: str, target_lang: str):
    translation = run_translator(
        text,
        target_lang,
        params['translation_model']
    )
    return {"translated_text": translation}
