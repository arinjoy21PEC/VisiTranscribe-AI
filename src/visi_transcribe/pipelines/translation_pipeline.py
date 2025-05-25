from src.visi_transcribe.components.translator import Translator

def run_translator(texts, target_lang, model_name, language_detection_model):
    translator = Translator(model_name, language_detection_model)
    return translator.translate(texts, target_lang)