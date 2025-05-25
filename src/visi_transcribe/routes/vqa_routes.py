from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os
from tempfile import NamedTemporaryFile
from src.visi_transcribe.pipelines.vqa_pipeline import generate_answer
from src.visi_transcribe.utils.logger import logger

router = APIRouter()

@router.post("/vqa/")
async def vqa_endpoint(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        suffix = os.path.splitext(image.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            temp_file.write(await image.read())

        logger.info(f"Received image: {image.filename}, question: {question}")
        answer = generate_answer(temp_path, question)
        os.remove(temp_path)

        return {"question": question, "answer": answer}
    
    except Exception as e:
        logger.exception("VQA endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))