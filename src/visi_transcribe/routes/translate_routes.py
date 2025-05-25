from fastapi import APIRouter, HTTPException
from typing import List, Optional
from src.visi_transcribe.pipelines.translation_pipeline import run_translator
from src.visi_transcribe.config.configuration import read_params
from src.visi_transcribe.utils.logger import logger

router = APIRouter()

config = read_params()
model_name_1 = config['translation_model']
model_name_2 = config['language_detection_model']

@router.post("/translate/")
async def translate_text(texts: List[str], target_lang: Optional[str] = None):
    try:
        logger.info(f"Received {len(texts)} texts for translation. Target language: {target_lang}")
        translations = run_translator(
            texts,
            target_lang,
            model_name_1,
            model_name_2
        )
        logger.info("Translation completed successfully.")
        return {"translated_texts": translations}
    except Exception as e:
        logger.exception("Translation endpoint failed.")
        raise HTTPException(status_code=500, detail=str(e))
