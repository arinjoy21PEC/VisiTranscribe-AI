from fastapi import FastAPI, UploadFile, File
from src.visi_transcribe.pipelines.image_captioning_pipeline import run_image_captioning
from src.visi_transcribe.pipelines.translation_pipeline import run_translator
import yaml

app = FastAPI()

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

@app.post("/image-to-text/")
async def image_to_text(file: UploadFile = File(...)):
    image_bytes = await file.read()
    caption = run_image_captioning(
        image_bytes,
        params['image_captioning_model']
    )

    return {"caption": caption}

@app.post("/translate/")
async def translate(text: str, target_lang: str):
    translation = run_translator(
        text,
        target_lang,
        params['translation_model']
    )
    return {"translated_text": translation}
