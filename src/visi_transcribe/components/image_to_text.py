from transformers import pipeline
from PIL import Image
import io

class IamageCaptioner:
    def __init__(self, model_name):
        self.caption_pipeline = pipeline("image-to-text", model=model_name)

    def generate_caption(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        result = self.caption_pipeline(image)
        return result[0]['generated_text']