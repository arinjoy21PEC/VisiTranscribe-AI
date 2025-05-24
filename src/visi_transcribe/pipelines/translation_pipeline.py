from src.visi_transcribe.components.translator import Translator

def run_translator(text, target_lang, model_name):
    translator = Translator(model_name)
    return translator.translate(text, target_lang)