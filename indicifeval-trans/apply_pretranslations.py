from datasets import load_dataset
import re
from utils import load_translations, kwargs_dict
from sys import argv

import re

def replace_phrase(text_string, input_phrase, target_phrase):
    # Return original string if input phrase is empty to avoid errors
    if not input_phrase:
        return text_string
        
    # Escape special characters in the input phrase (e.g., '.')
    escaped_phrase = re.escape(input_phrase)
    
    # Replace escaped spaces with a regex pattern for one or more whitespace characters
    flexible_whitespace_phrase = re.sub(r'\\ ', r'\\s+', escaped_phrase)
    
    # Conditionally add word boundaries. This prevents issues when a phrase
    # starts or ends with a non-alphanumeric character (like '?').
    start_boundary = r'\b' if input_phrase[0].isalnum() else ''
    end_boundary = r'\b' if input_phrase[-1].isalnum() else ''
    
    # Build the final pattern
    pattern_string = start_boundary + flexible_whitespace_phrase + end_boundary
    
    # Compile the regex for case-insensitive matching
    pattern = re.compile(pattern_string, re.IGNORECASE)
    
    # Substitute all occurrences of the pattern with the target phrase
    return pattern.sub(target_phrase, text_string)

def validate_one_to_one_mapping(data):
    if len(data.values()) != len(set(data.values())):
        duplicate_values = set()
        seen_values = {}
        for key, value in data.items():
            if value in seen_values:
                duplicate_values.add(value)
            else:
                seen_values[value] = key
        
        error_msg = "The dictionary does not have a one-to-one mapping. The following values are mapped to multiple keys:\n"
        for value in duplicate_values:
            keys = [k for k, v in data.items() if v == value]
            error_msg += f"  - Value '{value}' is mapped to keys {keys}\n"
            
        print(error_msg)
    else:
        print("Validation successful: The dictionary has a one-to-one mapping.")


if __name__ == "__main__":

    transformed_dataset = load_dataset("ai4bharat/IndicIFEval", "indicifeval-trans", split="en")
    
    lang_code = argv[1]

    output_filename = f'languages/{lang_code}/pretranslated_dataset-{lang_code}.jsonl'

    pretranslations = load_translations(f"languages/{lang_code}/pretranslations-{lang_code}.json")
    letter_mapping = load_translations(f'languages/{lang_code}/letter_mapping-{lang_code}.json')
    validate_one_to_one_mapping(pretranslations)
    validate_one_to_one_mapping(letter_mapping)
    prompt_list = list(transformed_dataset["prompt"])
    kwargs_list = list(transformed_dataset["kwargs"])

    pretranslated_prompt_list = []
    pretranslated_kwargs_list = []

    # print(kwargs_dict)

    for ex in transformed_dataset:
        pretranslated_prompt = ex["prompt"]
        pretranslated_kwargs = ex["kwargs"]
        for j, instruction_id in enumerate(ex["instruction_id_list"]):
            if instruction_id in list(kwargs_dict.keys()):
                if instruction_id == 'keywords:letter_frequency':
                    keyword = ex["kwargs"][j][kwargs_dict['keywords:letter_frequency']]
                    if ex["resp_lang"] != 'en' and keyword not in ["!", "#"]:
                        pretranslated_prompt = replace_phrase(pretranslated_prompt, keyword, letter_mapping[keyword])
                        pretranslated_kwargs[j][kwargs_dict[instruction_id]] = letter_mapping[keyword]
                else:
                    words_phrases = ex["kwargs"][j][kwargs_dict[instruction_id]]
                    if isinstance(words_phrases, list):
                        for k, keyword in enumerate(words_phrases):
                            # Replace all occurrences of the pattern in the prompt with the translated word/phrase'
                            pretranslated_prompt = replace_phrase(pretranslated_prompt, keyword, pretranslations[keyword])
                            pretranslated_kwargs[j][kwargs_dict[instruction_id]][k] = pretranslations[keyword]
                    else:
                        # Replace all occurrences of the pattern in the prompt with the translated word/phrase'
                        pretranslated_prompt = replace_phrase(pretranslated_prompt, words_phrases, pretranslations[words_phrases])
                        pretranslated_kwargs[j][kwargs_dict[instruction_id]] = pretranslations[words_phrases]
            if instruction_id == "detectable_format:constrained_response":
                for response in ["My answer is no.", "My answer is yes.", "My answer is maybe."]:
                    pretranslated_prompt = replace_phrase(pretranslated_prompt, response, pretranslations[response])

        pretranslated_kwargs_list.append(pretranslated_kwargs)
        pretranslated_prompt_list.append(pretranslated_prompt)

    pretranslations = dict(zip(prompt_list, pretranslated_prompt_list))
    
    # save_translations(pretranslations, f'languages/{lang_code}/pretranslated_dataset-{lang_code}.json')
    
    pretranslated_dataset = transformed_dataset.map(lambda examples, idx: {'prompt': pretranslated_prompt_list[idx], 'kwargs': pretranslated_kwargs_list[idx]}, with_indices=True)

    pretranslated_dataset.to_pandas().to_json(output_filename, orient="records", lines=True, force_ascii=False)
