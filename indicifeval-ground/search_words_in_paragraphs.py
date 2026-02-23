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

from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA

def search_chunk_file_keyword(file_path, target_word, exact_count, exact_sentences):
    """
    Worker function that loads a single JSONL chunk file and searches it for
    paragraphs matching the specified criteria.

    Args:
        file_path (str): Path to the .jsonl chunk file to process.
        target_word (str): The word to search for.
        exact_count (int): The exact number of times the word must appear.
        exact_sentences (int): The approximate number of sentences a paragraph must contain.

    Returns:
        A list of (id, text) tuples that match the criteria.
    """
    # Compile a regex pattern for whole-word, case-insensitive matching.
    # pattern = re.compile(r'\b' + re.escape(target_word) + r'\b', re.IGNORECASE)
    pattern = re.compile(r'(?<!\w)' + re.escape(target_word) + r'(?!\w)', re.IGNORECASE)

    
    num_paragraphs=1
    
    local_results = []

    # Load the single chunk file assigned to this worker.
    dataset_chunk = load_dataset('json', data_files=file_path, split='train', cache_dir=CACHE_DIR)

    for example in dataset_chunk:
        doc_id = example.get('id')
        text = example.get('text').strip()

        if not isinstance(text, str) or not doc_id:
            continue

        # Skip very long documents
        if len(text) > 1000*num_paragraphs:
            continue 

        paragraphs = text.split('\n\n')
        exact_sentences = exact_sentences*len(paragraphs)
        if not len(paragraphs) == num_paragraphs:
            continue

        if not any(["\n" in paragraph for paragraph in paragraphs]):
            continue
    
        # Check sentence count criteria
        num_sentences = sum([len(sentence_split(paragraph, lang="hin_Deva", delim_pat=DELIM_PAT_NO_DANDA)) for paragraph in paragraphs])
        if not ((exact_sentences - 2)*num_paragraphs <= num_sentences <= (exact_sentences + 2)*num_paragraphs):
            continue
        
        # Find all non-overlapping matches and get the count.
        # count = len(pattern.findall(paragraph))
        count = text.count(f' {target_word} ')

        # Store the result only if the count is an exact match.
        if count == exact_count:
            local_results.append((doc_id, text))
    
    return local_results

def main():
    """Main function to set up and run the parallel search."""
    parser = argparse.ArgumentParser(
        description="Find a random sample of docs from a directory of JSONL chunks."
    )
    parser.add_argument(
        "keyword",
        type=str,
        help="The keyword to search for in the dataset."
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
        "-c", "--count",
        type=int,
        required=True,
        help="The exact number of times the keyword must appear in a doc."
    )
    parser.add_argument(
        "-p", "--paragraphs",
        type=int,
        default=1,
        help="The number of paragraphs a doc should have."
    )
    parser.add_argument(
        "-s", "--sentences",
        type=int,
        default=6,
        help="The target number of sentences the doc must contain (allows a margin of +/- 2)."
    )
    parser.add_argument(
        "-k", "--limit",
        type=int,
        default=1,
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
    args.output_path = f"data/sangraha-verified-{args.lang}/samples1.jsonl"

    # --- 1. Find Input Files ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    chunk_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
    
    if not chunk_files:
        print(f"Error: No .jsonl files found in '{args.input_dir}'")
        return

    print(f"Found {len(chunk_files)} chunk files to process in '{args.input_dir}'.")

    # --- 2. Set up for Multiprocessing ---
    num_processes = args.num_proc
    print(f"Starting parallel search with {num_processes} processes...")
    
    # Use functools.partial to pre-fill arguments for the worker function.
    worker_func = functools.partial(
        search_chunk_file_keyword, 
        target_word=args.keyword, 
        exact_count=args.count,
        exact_sentences=args.sentences,
    )

    all_found_paragraphs = []
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # The list of chunk_files is the iterable that gets distributed among workers.
        results_from_workers = list(tqdm(
            pool.imap_unordered(worker_func, chunk_files),
            total=len(chunk_files),
            desc=f"Searching for '{args.keyword}' (count={args.count}, sentences≈{args.sentences*args.paragraphs}, paragraphs={args.paragraphs})"
        ))

    # --- 3. Combine, Shuffle, and Limit Results ---
    print("\nSearch complete. Processing results...")
    
    # Flatten the list of lists into a single list of results.
    for worker_result in results_from_workers:
        all_found_paragraphs.extend(worker_result)
    
    total_found = len(all_found_paragraphs)

    # Shuffle the list of all found paragraphs randomly.
    random.shuffle(all_found_paragraphs)

    # Limit the number of results to display.
    results_to_display = all_found_paragraphs[:args.limit]

    # --- 4. Store the Final Output ---
    print(f"Found a total of {total_found:,} docs where '{args.keyword}' appears exactly {args.count} time(s) in a {args.paragraphs} paragraph doc.")

    if not results_to_display:
        print("Nothing to store.")
    else:
        with open(args.output_path, 'a', encoding='utf-8') as f_out:
            for doc_id, paragraph in results_to_display:
                record = {'id': f"sangraha_verified_{args.lang}_{args.keyword}_{args.paragraphs}_{args.count}_{doc_id}",
                          'text': paragraph,
                          'keyword': args.keyword,
                          'paragraphs': args.paragraphs,
                          'frequency': args.count 
                        }
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Appended data to {args.output_path}")


if __name__ == "__main__":
    main()