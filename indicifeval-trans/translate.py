import argparse
import os
from tqdm import tqdm
from datasets import load_dataset

import time


# --- Utility Function ---
from utils import LANG_MAP

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
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=False,
    )
    input_ids_len = len(model_inputs.input_ids[0])
    output_ids = generated_ids[0][input_ids_len:].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_text.strip()

# --- Main Execution ---

def main():
    """
    Main function to parse arguments and run the specified translation process.
    """
    parser = argparse.ArgumentParser(description="Translate a dataset using SarvamAI API or a local Hugging Face model.")
    
    # --- CLI Arguments ---
    parser.add_argument("lang_code", type=str, help="Target language code (e.g., 'hi', 'ta').")
    
    # Mutually exclusive group for translation method
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--use-api", action="store_true", help="Use the SarvamAI API for translation.")
    method_group.add_argument("--use-hf", action="store_true", help="Use a local Hugging Face model for translation.")
    
    # Token arguments
    parser.add_argument("--sarvamai_token", type=str, help="Your SarvamAI API subscription key.")
    parser.add_argument("--hf_token", type=str, help="Your Hugging Face Hub token (optional, for gated models).")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.use_api and not args.sarvamai_token:
        parser.error("--sarvamai_token is required when using --use-api")

    lang_code = args.lang_code
    tgt_lang_full = LANG_MAP.get(lang_code)
    if not tgt_lang_full:
        raise ValueError(f"Language code '{lang_code}' not found in LANG_MAP.")

    # --- Dataset Loading and Preparation ---
    input_filename = f"languages/{lang_code}/pretranslated_dataset-{lang_code}.jsonl"
    output_filename = f"languages/{lang_code}/translated_dataset-{lang_code}.jsonl"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    transformed_dataset = load_dataset("ai4bharat/IndicIFEval", "indicifeval-trans", split="en")
    
    # Load pre-translated modifications if the file exists
    if os.path.exists(input_filename):
        modified_data = load_dataset('json', data_files=input_filename, split='train')
        updates_map = {item['key']: item for item in modified_data}
        
        def apply_modifications(example):
            prompt_id = example['key']
            return updates_map.get(prompt_id, example)
            
        pretranslated_dataset = transformed_dataset.map(apply_modifications)
    else:
        print(f"Info: No pre-translation file found at '{input_filename}'. Translating the entire dataset.")
        pretranslated_dataset = transformed_dataset

    lines_to_translate = list(pretranslated_dataset['prompt'])
    translations = []

    print(f"\nStarting translation of {len(lines_to_translate)} sentences to {tgt_lang_full}...")

    # --- Translation Logic ---
    
    # ==> Method 1: SarvamAI API
    if args.use_api:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=args.sarvamai_token)
        sarvam_translate_kwargs = {
            "source_language_code": "en-IN",
            "target_language_code": f"{lang_code}-IN",
            "mode": "formal",
            "model": "sarvam-translate:v1",
            "numerals_format": "international",
            "speaker_gender": "Male",
            "enable_preprocessing": False
        }
        
        for sentence in tqdm(lines_to_translate, desc=f"Translating with API to {tgt_lang_full}", unit="sentence"):
            if sentence.startswith("Translate the following sentence into German and then criticize it"):
                translation1 = translate_text_api(client, sentence.split(": ")[0], sarvam_translate_kwargs)
                translation2 = translate_text_api(client, sentence.split(": ")[1], sarvam_translate_kwargs)
                translation = translation1+": "+translation2
            elif sentence.startswith("Can you provide a translation for \"今天天气很好\" in German"):  
                translation1 = translate_text_api(client, sentence.split("? ")[0].replace("今天天气很好", "@"), sarvam_translate_kwargs)
                translation2 = translate_text_api(client, sentence.split("? ")[1], sarvam_translate_kwargs)
                translation = translation1.replace("@", "今天天气很好")+"? "+translation2
            elif sentence.startswith("Can you expand the following sentences"):
                translation1 = translate_text_api(client, sentence.split(": ")[0], sarvam_translate_kwargs)
                translation2 = translate_text_api(client, sentence.split(": ")[1], sarvam_translate_kwargs)
                translation = translation1+": "+translation2
            elif sentence.startswith("Write a summary of the following text in a funny way"):
                translation1 = translate_text_api(client, sentence.split(": ")[0], sarvam_translate_kwargs)
                translation2 = translate_text_api(client, sentence.split(": ")[1], sarvam_translate_kwargs)
                translation = translation1+": "+translation2
            elif "\n\n" in sentence and any([word in sentence.lower() for word in ["summary", "summarize"]]):
                translation = "\n\n".join([translate_text_api(client, sent, sarvam_translate_kwargs) for sent in sentence.split("\n\n") if sent.strip()])
            else:    
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
            model_name, 
            torch_dtype=torch.bfloat16, 
            token=args.hf_token
        ).to(device)
        
        for sentence in tqdm(lines_to_translate, desc=f"Translating with HF model to {tgt_lang_full}", unit="sentence"):
            translation = translate_text_hf(model, tokenizer, device, sentence, tgt_lang_full)
            translations.append(translation)

    # --- Save Results ---
    translated_dataset = pretranslated_dataset.map(lambda examples, idx: {'prompt': translations[idx]}, with_indices=True)
    translated_dataset.to_pandas().to_json(output_filename, orient="records", lines=True, force_ascii=False)

    print(f"\nTranslation complete! Output saved to '{output_filename}'.")


if __name__ == "__main__":
    main()