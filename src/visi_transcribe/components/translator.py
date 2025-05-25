from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
from tqdm import tqdm
import time
from src.visi_transcribe.utils.logger import logger

class Translator:
    def __init__(self, model_name, language_detector):
        logger.info(f"Initializing Translator with model: {model_name}")

        for _ in tqdm(range(100), desc="Loading Translation Model", ncols=75):
            time.sleep(0.005)
        
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.language_detector = pipeline("text-classification", model=language_detector)

        logger.info("Translation Model loaded Successfully")


    def detect_language(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        predictions = self.language_detector(texts)

        if isinstance(predictions, dict) or (len(predictions) > 0 and isinstance(predictions[0], dict)):
            return [pred['label'] for pred in predictions]
        else:
            return [pred[0]['label'] for pred in predictions]
    
    def translate(self, texts, target_lang = None):
        if isinstance(texts, str):
            texts = [texts]
            
        detected_langs = self.detect_language(texts)
        results = []

        for text, src_lang in zip(texts, detected_langs):
            if target_lang is None or target_lang == src_lang:
                logger.info(f"Skipping translation for text already in '{src_lang}'")
                translated = text
            else:
                self.tokenizer.src_lang = src_lang
                encoded = self.tokenizer(text, return_tensors = "pt")
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id = self.tokenizer.get_lang_id(target_lang)
                )
                translated = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                logger.info(f"Translated from {src_lang} to {target_lang} : {translated}")

            results.append({
                "original_text": text,
                "detected_language": src_lang,
                "translated_text": translated
            })
        return results
    
