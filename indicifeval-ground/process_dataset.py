import argparse
import json
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import load_dataset, Dataset
from tqdm import tqdm

CACHE_DIR = ".cache"

def process_chunk(shard: Dataset, id_column: str, text_column: str, output_path: str):
    """
    Processes a single shard of the dataset and saves it to a JSONL file.
    This function is executed by each worker process.

    Args:
        shard (Dataset): A shard of the main Hugging Face dataset.
        id_column (str): The name of the column to be used as 'id'.
        text_column (str): The name of the column to be used as 'text'.
        output_path (str): The path to save the output JSONL chunk file.
    """
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for record in shard:
            # Create a new dictionary with standardized keys
            standardized_record = {
                'id': record[id_column],
                'text': record[text_column]
            }
            # Convert the dictionary to a JSON string and write it as a new line
            f_out.write(json.dumps(standardized_record, ensure_ascii=False) + '\n')
    return output_path

def main():
    """
    Main function to parse command-line arguments and run the parallel processing.
    """
    parser = argparse.ArgumentParser(
        description="Load a Hugging Face dataset, standardize columns, and save as JSONL chunks in parallel.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--id_column",
        type=str,
        default="doc_id",
        help="Name of the existing column to be standardized as 'id'."
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the existing column to be standardized as 'text'."
    )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="data/sangraha-verified-tam/",
    #     help="Path to the directory to save the output JSONL chunks."
    # )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language of sangraha split"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=128,
        help="Number of worker processes to use. Defaults to the number of CPU cores."
    )

    args = parser.parse_args()

    args.output_dir = f"data/sangraha-verified-{args.lang}/chunks"

    # Determine the number of processes to use
    num_processes = args.num_processes
    print(f"Using {num_processes} worker processes.")

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved in '{args.output_dir}'")

    print("Loading the dataset...")
    # This part remains sequential as the dataset needs to be loaded first
    if args.lang == "npi":
        temp_lang = "nep"
    elif args.lang == "ory":
        temp_lang = "ori"
    else:
        temp_lang = args.lang 
    dataset = load_dataset(
        "ai4bharat/sangraha",
        data_dir=f"verified/{temp_lang}",
        cache_dir=CACHE_DIR,
        split="train"
    )
    print("Dataset loaded successfully.")

    # Prepare arguments for each chunk processing task
    tasks = []
    for i in range(num_processes):
        # The .shard() method is a highly efficient way to split the work
        shard = dataset.shard(num_shards=num_processes, index=i)
        output_path = os.path.join(args.output_dir, f"chunk_{i:03d}.jsonl")
        tasks.append((shard, args.id_column, args.text_column, output_path))

    # Use a ProcessPoolExecutor to manage the worker processes
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks to the process pool
        futures = [executor.submit(process_chunk, *task) for task in tasks]

        # Use tqdm to create a progress bar for completed tasks
        progress_bar = tqdm(total=len(tasks), desc="Processing chunks")
        for future in as_completed(futures):
            # As each task completes, update the progress bar
            result_path = future.result()
            # print(f"Chunk successfully saved to {result_path}") # Optional: uncomment for verbose logging
            progress_bar.update(1)
        progress_bar.close()

    print(f"\nProcessing complete. All chunks saved in '{args.output_dir}'.")


if __name__ == "__main__":
    # This is crucial for multiprocessing to work correctly on all platforms
    multiprocessing.set_start_method("fork")
    main()