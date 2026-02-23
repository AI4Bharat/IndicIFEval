import argparse
import json
import re
import os
import math
import multiprocessing
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

from functools import partial

# --- Configuration ---
CACHE_DIR = ".cache"

# Define characters to be removed from the text during cleaning
import re

REGEX_PATTERNS = {
    "asm": r"[\u0980-\u09FF]+",      # Assamese (Bengali script)
    "ben": r"[\u0980-\u09FF]+",      # Bengali
    "guj": r"[\u0A80-\u0AFF]+",      # Gujarati
    "hin": r"[\u0900-\u097F]+",      # Hindi (Devanagari)
    "kan": r"[\u0C80-\u0CFF]+",      # Kannada
    "mar": r"[\u0900-\u097F]+",      # Marathi (Devanagari)
    "mal": r"[\u0D00-\u0D7F]+",      # Malayalam
    "npi": r"[\u0900-\u097F]+",      # Nepali (Devanagari)
    "ory": r"[\u0B00-\u0B7F]+",      # Odia
    "pan": r"[\u0A00-\u0A7F]+",      # Punjabi (Gurmukhi)
    "san": r"[\u0900-\u097F]+",      # Sanskrit (Devanagari)
    "tam": r"[\u0B80-\u0BFF]+",      # Tamil
    "tel": r"[\u0C00-\u0C7F]+",      # Telugu
    "urd": r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+",  
}

# Precompile for speed
COMPILED = {lang: re.compile(f"^{pattern}$") for lang, pattern in REGEX_PATTERNS.items()}

def is_correct_script(word: str, lang) -> bool:
    """
    Returns True if `word` is written fully in the script used by FLORES language `lang`.
    """
    pattern = COMPILED.get(lang)
    if pattern is None:
        return False
    return pattern.match(word) is not None


# print(is_correct_script("नमस्ते", "hin"))   # True
# print(is_correct_script("hello", "hin"))    # False
# print(is_correct_script("தமிழ்", "tam"))     # True
# print(is_correct_script("മലയാളം", "mal"))     # True
# print(is_correct_script("سلام", "urd"))      # True
# print(is_correct_script("سلام", "hin"))      # False

import regex

# Unicode P = all punctuation marks from all scripts
PUNCT_ALL = regex.compile(r"\p{P}+", flags=regex.UNICODE)

def remove_punctuation(text: str) -> str:
    return PUNCT_ALL.sub("", text)

# --- 1. Define the Worker Function ---
def process_chunk_file(file_path, lang):
    """
    Loads and processes a single JSONL chunk file to count TF and DF.
    
    Args:
        file_path (str): The path to the .jsonl chunk file to process.

    Returns:
        tuple: (local_tf (Counter), local_df (Counter), local_doc_count (int))
    """
    local_tf = Counter()
    local_df = Counter()
    local_doc_count = 0
    
    # Load the specific chunk file assigned to this worker
    # Using try-except block to prevent one bad file from crashing the whole pool
    try:
        dataset_chunk = load_dataset('json', data_files=file_path, split='train', cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"Warning: Failed to load {file_path}. Error: {e}")
        return local_tf, local_df, 0
    
    
    for example in dataset_chunk:
        text = example.get('text')
        if isinstance(text, str):
            local_doc_count += 1

            text = remove_punctuation(text)
            words = text.split()
            
            if not words:
                continue
            
            words = [word.strip() for word in words if is_correct_script(word, lang)]
            
            words = [word for word in words if len(word) > 3]

            # 1. Update Total Frequency (TF) - Count every occurrence
            local_tf.update(words)
            
            # 2. Update Document Frequency (DF) - Count unique words per document
            unique_words_in_doc = set(words)
            local_df.update(unique_words_in_doc)
            
    return local_tf, local_df, local_doc_count

def main():
    """Main function to set up and run the parallel TF-IDF extraction."""
    parser = argparse.ArgumentParser(
        description="Calculate TF-IDF scores from a directory of JSONL chunk files in parallel."
    )
    # parser.add_argument(
    #     "--input_dir",
    #     type=str,
    #     default="data/sangraha-verified-hin/chunks",
    #     help="Directory containing the input .jsonl chunk files."
    # )
    # parser.add_argument(
    #     "--output_file",
    #     type=str,
    #     default="hindi_tfidf_top_words.json",
    #     help="Path for the output JSON file with top words."
    # )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=128,
        help="Number of CPU processes to use."
    )
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=500,
        help="Number of top meaningful words to extract."
    )
    parser.add_argument(
        "-f", "--from_k",
        type=int,
        default=0,
        help="Start index for top k"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language of sangraha split"
    )
    args = parser.parse_args()
    args.input_dir = f"data/sangraha-verified-{args.lang}/chunks"
    args.output_file = f"data/sangraha-verified-{args.lang}/words.txt"

    global GLOBAL_LANG
    GLOBAL_LANG = args.lang

    # --- 2. Find and Validate Input Files ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    chunk_files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.jsonl')])
    
    if not chunk_files:
        print(f"Error: No .jsonl files found in '{args.input_dir}'")
        return

    print(f"Found {len(chunk_files)} chunk files to process in '{args.input_dir}'.")

    # --- 3. Set up and Run Multiprocessing (Map-Reduce) ---
    num_processes = args.num_processes
    print(f"Starting parallel processing with {num_processes} processes.")

    global_tf = Counter()
    global_df = Counter()
    total_docs = 0
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Distribute files to workers
        process_chunk_file_lang = partial(process_chunk_file, lang=args.lang)
        results = list(tqdm(
            pool.imap_unordered(process_chunk_file_lang, chunk_files),
            total=len(chunk_files),
            desc="Processing chunks (TF/DF)"
        ))

    # Combine results
    print("Combining results from all processes...")
    for tf_counts, df_counts, doc_count in tqdm(results, desc="Merging counters"):
        global_tf.update(tf_counts)
        global_df.update(df_counts)
        total_docs += doc_count

    print(f"Total Documents Processed: {total_docs:,}")
    print(f"Unique Vocabulary Size: {len(global_tf):,}")

    # --- 4. Calculate TF-IDF Scores ---
    print("Computing TF-IDF scores...")
    tfidf_scores = {}

    if total_docs == 0:
        print("Error: No documents were processed.")
        return

    for word, tf in global_tf.items():
        df = global_df[word]
        
        # Calculate IDF: log10(Total Documents / Document Frequency)
        idf = math.log10(total_docs / df) if df > 0 else 0
        
        # Calculate Score: TF * IDF
        score = tf * idf
        
        tfidf_scores[word] = score

    # --- 5. Sort and Save ---
    print(f"Sorting to find top {args.top_k} meaningful words...")
    
    # Sort dictionary by score in descending order
    sorted_words = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
    top_k_words = sorted_words[args.from_k:args.from_k+args.top_k]

    # Prepare output data structure
    output_data = {}
    
    print("\n--- Top Meaningful Words (TF-IDF) ---")
    header = f"{'Rank':<6} | {'Word':<20} | {'TF':<10} | {'DF':<10} | {'Score':<10}"
    print(header)
    print("-" * len(header))

    for rank, (word, score) in enumerate(top_k_words, 1):
        tf = global_tf[word]
        df = global_df[word]
        print(f"{rank:<6} | {word:<20} | {tf:<10} | {df:<10} | {score:<10.2f}")
        
        output_data[word] = {
            "score": score,
            "tf": tf,
            "df": df
        }

    print(f"\nSaving top {args.top_k} words from index {args.from_k} to '{args.output_file}'...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        # json.dump(output_data, f, ensure_ascii=False, indent=4)
        for word in output_data.keys():
            f.write(f"{word}\n")

    print("🎉 Script finished successfully.")

# --- Main execution block ---
if __name__ == "__main__":
    main()