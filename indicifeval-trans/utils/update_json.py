import argparse
import json
import sys
from utils import save_translations, load_translations

def update_json_value(filepath, source_key, target_value):
    """
    Updates the value of a specific key in a JSON file.

    Args:
        filepath (str): The path to the JSON file.
        source_key (str): The key whose value needs to be updated.
        target_value (str): The new value to set for the key.
    """
    filepath=f"languages/{filepath}/pretranslations-{filepath}.json"
    data = load_translations(filepath)

    if source_key in data:
        data[source_key] = target_value
        save_translations(data, filepath)
        print(f"Successfully updated key '{source_key}' to value '{target_value}' in '{filepath}'.")
    else:
        print(f"Error: The key '{source_key}' was not found in the JSON file.")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Update a key's value in a JSON file.")
    parser.add_argument("filepath", help="The path to the JSON file.")
    parser.add_argument("source_key", help="The key to update.")
    parser.add_argument("target_value", help="The new value for the key.")

    args = parser.parse_args()
    update_json_value(args.filepath, args.source_key, args.target_value)
