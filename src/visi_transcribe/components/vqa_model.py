import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from src.visi_transcribe.utils.logger import logger
from src.visi_transcribe.config.configuration import read_params

class VQAModel:
    def __init__(self):
        config = read_params()
        model_name = config['VQA_Model']

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(f"Loading Model '{model_name}' on {device}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

        self.device = device
        logger.info("Model loaded successfully")

    def get_model(self):
        return self.model
    
    def get_processor(self):
        return self.processor
    
    def get_device(self):
        return self.device
    

