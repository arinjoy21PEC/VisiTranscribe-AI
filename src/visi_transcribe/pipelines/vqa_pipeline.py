from PIL import Image
from src.visi_transcribe.components.vqa_model import VQAModel
from src.visi_transcribe.utils.logger import logger

vqa_handler = VQAModel()

def generate_answer(image_path: str, question: str):
    try:
        image = Image.open(image_path).convert("RGB")
        logger.info(f"Processing image: {image_path}")

        inputs = vqa_handler.get_processor()(image, question, return_tensors="pt").to(vqa_handler.get_device())
        output_ids = vqa_handler.get_model().generate(**inputs)
        answer = vqa_handler.get_processor().batch_decode(output_ids, skip_special_tokens = True)[0].strip()

        logger.info(f"Generated answer: {answer}")

        return answer
    except Exception as e:
        logger.error(f"Error during VQA interface: {e}")
        raise e
    