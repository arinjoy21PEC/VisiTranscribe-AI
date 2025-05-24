from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class Translator:
    def __init__(self, model_name):
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    
    def translate(self, text, target_lang):
        self.tokenizer.src_lang = "en"
        encoded = self.tokenizer(text, return_tensors="pt")

        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id = self.tokenizer.get_lang_id(target_lang)
        )
        
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]