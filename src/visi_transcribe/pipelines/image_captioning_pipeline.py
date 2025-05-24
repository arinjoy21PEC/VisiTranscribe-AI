from src.visi_transcribe.components.image_to_text import ImageCaptioner

def run_image_captioning(image_bytes, model_name):
    captioner = ImageCaptioner(model_name)
    return captioner.generate_caption(image_bytes)