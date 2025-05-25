from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from src.visi_transcribe.pipelines.image_captioning_pipeline import run_image_captioning
from src.visi_transcribe.config.configuration import read_params
from src.visi_transcribe.utils.logger import logger

router = APIRouter()

config = read_params()
model_name = config['image_captioning_model']

@router.post("/image-to-text/")
async def image_to_text(files: List[UploadFile] = File(...)):
    try:
        logger.info(f"Received {len(files)} image(s) for captioning.")
        results = run_image_captioning(files, model_name)
        logger.info("Image captioning completed successfully.")
        return {"results": results}
    except Exception as e:
        logger.exception("Image-to-text endpoint failed.")
        raise HTTPException(status_code=500, detail=str(e))
