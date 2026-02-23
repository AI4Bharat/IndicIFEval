import argparse
import os
import re
import functools
import multiprocessing
import random
from datasets import load_dataset
from tqdm import tqdm
import json

random.seed(42)

# --- Configuration ---
CACHE_DIR = ".cache"

LANG_SCRIPT = {
    "asm": "asm_Beng",   # Assamese (Bengali script)
    "ben": "ben_Beng",   # Bengali
    "guj": "guj_Gujr",   # Gujarati
    "hin": "hin_Deva",   # Hindi
    "kan": "kan_Knda",   # Kannada
    "mar": "mar_Deva",   # Marathi
    "mal": "mal_Mlym",   # Malayalam
    "npi": "npi_Deva",   # Nepali
    "ory": "ory_Orya",   # Odia
    "pan": "pan_Guru",   # Punjabi (Gurmukhi)
    "san": "san_Deva",   # Sanskrit
    "tam": "tam_Taml",   # Tamil
    "tel": "tel_Telu",   # Telugu
    "urd": "urd_Arab",   # Urdu (Arabic script)
}


from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA

def search_chunk_file_para(file_path, exact_paragraphs, exact_sentences, lang):
    """
    Worker function that loads a single JSONL chunk file and searches it for
    paragraphs matching the specified criteria.

    Args:
        file_path (str): Path to the .jsonl chunk file to process.
        exact_paragraphs (int): The exact number of paragraphs the doc must contain.
        exact_sentences (int): The exact number of sentences a paragraph must contain.

    Returns:
        A list of (id, text) tuples that match the criteria.
    """
    
    local_results = []

    # Load the single chunk file assigned to this worker.
    dataset_chunk = load_dataset('json', data_files=file_path, split='train', cache_dir=CACHE_DIR)

    for example in dataset_chunk:
        doc_id = example.get('id')
        text = example.get('text').strip()

        if not isinstance(text, str) or not doc_id:
            continue

        # Skip very long documents
        if len(text) > 1000*exact_paragraphs:
            continue 

        paragraphs = text.split('\n\n')

        if not len(paragraphs) == exact_paragraphs:
            continue

        paragraphs = [paragraph.replace("\n", " ") for paragraph in paragraphs]
        text = "\n\n".join(paragraphs)
    
        # Check sentence count criteria
        num_sentences = sum([len(sentence_split(paragraph, lang=LANG_SCRIPT[lang], delim_pat=DELIM_PAT_NO_DANDA)) for paragraph in paragraphs])

        # Store the result only if it is an exact match or if exact_sentences requirement is not applied.
        if num_sentences == exact_sentences or exact_sentences == 0:
            local_results.append((doc_id, text))
    
    return local_results

def main():
    """Main function to set up and run the parallel search."""
    parser = argparse.ArgumentParser(
        description="Find a random sample of docs from a directory of JSONL chunks."
    )
    # parser.add_argument(
    #     "--input_dir",
    #     type=str,
    #     default="data/sangraha-verified-hin/chunks/",
    #     help="Directory containing the input .jsonl chunk files."
    # )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language"
    )
    parser.add_argument(
        "-p", "--paragraphs",
        type=int,
        required=True,
        help="The number of paragraphs a doc should have."
    )
    parser.add_argument(
        "-s", "--sentences",
        type=int,
        required=True,
        help="The target number of sentences the doc must EXACTLY contain (assume 1 paragraph)"
    )
    parser.add_argument(
        "-k", "--limit",
        type=int,
        default=5,
        help="The maximum number of random docs to display (default: 5)."
    )
    parser.add_argument(
        "-n", "--num_proc",
        type=int,
        default=128,
        help="The number of cpu processes to use. Defaults to all available cores."
    )
    # parser.add_argument(
    #     "-o", "--output_path",
    #     type=str,
    #     default="data/sangraha-verified-hin/samples2.jsonl",
    #     help="The output path to append example rows (jsonl)."
    # )
    args = parser.parse_args()

    args.input_dir = f"data/sangraha-verified-{args.lang}/chunks/"
    args.input_path = f"data/sangraha-verified-{args.lang}/samples1.jsonl" 
    args.output_path = f"data/sangraha-verified-{args.lang}/samples2.jsonl"

    # --- 1. Find Input Files ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    chunk_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
    
    if not chunk_files:
        print(f"Error: No .jsonl files found in '{args.input_dir}'")
        return

    print(f"Found {len(chunk_files)} chunk files to process in '{args.input_dir}'.")


    # --- 2. Set up for Multiprocessing  ---
    num_processes = args.num_proc
    print(f"Starting parallel search with {num_processes} processes...")
    
    # Use functools.partial to pre-fill arguments for the worker function.
    worker_func = functools.partial(
        search_chunk_file_para, 
        exact_sentences=args.sentences,
        exact_paragraphs=args.paragraphs,
        lang=args.lang
    )
    all_found_paragraphs = []
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # The list of chunk_files is the iterable that gets distributed among workers.
        results_from_workers = list(tqdm(
            pool.imap_unordered(worker_func, chunk_files),
            total=len(chunk_files),
            desc=f"Searching for relevant docs (sentences={args.sentences}, paragraphs={args.paragraphs})"
        ))

    # --- 3. Combine, Remove existing paragraphs, Shuffle, and Limit Results ---
    print("\nSearch complete. Processing results...")
    
    # Flatten the list of lists into a single list of results.
    for worker_result in results_from_workers:
        all_found_paragraphs.extend(worker_result)
    
    total_found = len(all_found_paragraphs)

    if True: # TODO: add a boolean argparse flag instead
         ids_to_ignore = set()
         with open(args.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                ids_to_ignore.add(json.loads(line)['id'].split("_")[-1])
    
    # Shuffle the list of all found paragraphs randomly.
    random.shuffle(all_found_paragraphs)

    # Limit the number of results to display.
    results_to_display = all_found_paragraphs[:args.limit]

    # --- 4. Store the Final Output ---
    print(f"Found a total of {total_found:,} docs with {args.paragraphs} paragraphs and {args.sentences} sentences.")

    if not results_to_display:
        print("Nothing to store.")
    else:
        with open(args.output_path, 'a', encoding='utf-8') as f_out:
            for doc_id, paragraph in results_to_display:
                if doc_id in ids_to_ignore:
                    continue
                record = {'id': f"sangraha_verified_{args.lang}_{args.paragraphs}_{args.sentences}_{doc_id}",
                          'text': paragraph,
                          'sentences': args.sentences,
                          'paragraphs': args.paragraphs 
                        }
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Appended data to {args.output_path}")


if __name__ == "__main__":
    main()