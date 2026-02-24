import os
import json

# Dictionary to map language codes to full names for the model's prompt
LANG_MAP = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ks": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "od": "Odia",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu"
}

kwargs_dict = {
    'detectable_content:postscript': 'postscript_marker',
    'keywords:existence': 'keywords',
    'keywords:forbidden_words': 'forbidden_words',
    'keywords:frequency': 'keyword',
    'keywords:letter_frequency': 'letter', # We only consider cases where resp_lang is not English and further ignore this constraint case during backtranslation
    'length_constraints:nth_paragraph_first_word': 'first_word',
    'startend:end_checker': 'end_phrase',
    'detectable_format:multiple_sections': 'section_spliter',
}

def load_translations(filepath=None):
    """Loads translations from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: '{filepath}' contains invalid JSON.")
            raise FileNotFoundError
    else:
        print(f"Warning: '{filepath}' not found.")
        raise FileNotFoundError


def save_translations(translations_dict, filepath=None):
    """Saves translations to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(translations_dict, f, indent=4, ensure_ascii=False)
        print(f"\nTranslations saved successfully to '{filepath}'")
    except IOError as e:
        print(f"Error saving translations to '{filepath}': {e}")