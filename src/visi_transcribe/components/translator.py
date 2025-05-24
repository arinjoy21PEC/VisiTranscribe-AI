from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
import time
from src.visi_transcribe.utils.logger import logger

class Translator:
    def __init__(self, model_name):
        logger.info(f"Initializing Translator with model: {model_name}")

        for _ in tqdm(range(100), desc="Loading Translation Model", ncols=75):
            time.sleep(0.005)
        
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        logger.info("Translation Model loaded Successfully")
    
    def translate(self, text, target_lang):
        self.tokenizer.src_lang = "en"
        encoded = self.tokenizer(text, return_tensors="pt")

        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id = self.tokenizer.get_lang_id(target_lang)
        )

        translated = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        logger.info(f"Translated Text: {translated}")
        
        return translated