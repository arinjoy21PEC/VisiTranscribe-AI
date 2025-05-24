from transformers import pipeline
from PIL import Image
import io
from tqdm import tqdm
import time
from src.visi_transcribe.utils.logger import logger

class ImageCaptioner:
    def __init__(self, model_name):
        logger.info(f"Initializing ImageCaptioner with model: {model_name}")

        for _ in tqdm(range(100), desc="Loading Captioning Model", ncols=75):
            time.sleep(0.005)
        self.caption_pipeline = pipeline("image-to-text", model=model_name)

        logger.info("Image Captioning Model Loaded Successfully")

    def generate_caption(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        result = self.caption_pipeline(image)
        caption = result[0]['generated_text']
        return caption
    
    def generate_captions_batch(self, files):
        results = []
        logger.info(f"Processing batch of {len(files)} files")
        for file in files:
            image_bytes = file.file.read()
            caption = self.generate_caption(image_bytes)
            results.append({
                "filename": file.filename,
                "caption": caption
            })
            logger.info(f"Processed {file.filename} : {caption}")

        return results