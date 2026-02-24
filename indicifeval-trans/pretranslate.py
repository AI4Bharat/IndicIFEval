import argparse
import json
import os
from tqdm import tqdm

import time

# --- Utility Function ---
from utils import LANG_MAP, save_translations

def read_file_lines(filepath):
    """Reads a file and returns a list of its lines, stripped of whitespace."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

# --- Translation Functions ---

def translate_text_api(client, input_txt, sarvam_translate_kwargs):
    """Translates text using the SarvamAI API."""
    response = client.text.translate(input=input_txt, **sarvam_translate_kwargs)
    time.sleep(0.05)
    return response.translated_text.strip()

def translate_text_hf(model, tokenizer, device, input_txt, tgt_lang_full):
    """Translates text using a local Hugging Face model."""
    messages = [
        {"role": "system", "content": f"Translate the text below to {tgt_lang_full}."},
        {"role": "user", "content": input_txt}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # Using the generation parameters from your original script
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.01,
        num_return_sequences=1
    )
    
    input_ids_len = len(model_inputs.input_ids[0])
    output_ids = generated_ids[0][input_ids_len:].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_text.strip()

# --- Main Execution ---

def main():
    """
    Main function to parse arguments and run the pre-translation process.
    """
    parser = argparse.ArgumentParser(
        description="Translate a text file line-by-line using SarvamAI API or a local Hugging Face model."
    )
    
    # --- CLI Arguments ---
    parser.add_argument("lang_code", type=str, help="Target language code (e.g., 'hi', 'ta').")
    
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--use-api", action="store_true", help="Use the SarvamAI API for translation.")
    method_group.add_argument("--use-hf", action="store_true", help="Use a local Hugging Face model for translation.")
    
    parser.add_argument("--sarvamai_token", type=str, help="Your SarvamAI API subscription key.")
    parser.add_argument("--hf_token", type=str, help="Your Hugging Face Hub token (optional).")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.use_api and not args.sarvamai_token:
        parser.error("--sarvamai_token is required when using --use-api")

    lang_code = args.lang_code
    tgt_lang_full = LANG_MAP.get(lang_code)
    if not tgt_lang_full:
        raise ValueError(f"Language code '{lang_code}' not found in LANG_MAP.")

    # --- File I/O Setup ---
    input_filename = f"languages/en/pretranslations.txt"
    output_filename = f"languages/{lang_code}/pretranslations-{lang_code}.json"
    
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"Input file not found: {input_filename}")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    lines_to_translate = read_file_lines(input_filename)
    translations = []

    print(f"\nStarting translation of {len(lines_to_translate)} sentences to {tgt_lang_full}...")

    # --- Translation Logic ---
    
    # ==> Method 1: SarvamAI API
    if args.use_api:
        target_language_code = lang_code if lang_code != "or" else "od"
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=args.sarvamai_token)
        sarvam_translate_kwargs = {
            "source_language_code": "en-IN",
            "target_language_code": f"{target_language_code}-IN",
            "model": "sarvam-translate:v1",
        }
        for sentence in tqdm(lines_to_translate, desc=f"Translating with API to {tgt_lang_full}", unit="sentence"):
            translation = translate_text_api(client, sentence, sarvam_translate_kwargs)
            translations.append(translation)

    # ==> Method 2: Local Hugging Face Model
    elif args.use_hf:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "sarvamai/sarvam-translate"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model '{model_name}' on {device.type}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, token=args.hf_token
        ).to(device)
        
        for sentence in tqdm(lines_to_translate, desc=f"Translating with HF model to {tgt_lang_full}", unit="sentence"):
            translation = translate_text_hf(model, tokenizer, device, sentence, tgt_lang_full)
            translations.append(translation)
    
    # --- Save Results ---
    # Create a dictionary mapping original sentences to their translations
    save_translations(dict(zip(lines_to_translate, translations)), output_filename)

    print(f"\nTranslation complete! Output saved to '{output_filename}'.")


if __name__ == "__main__":
    main()
