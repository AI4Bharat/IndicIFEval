from datasets import load_dataset
import re
from utils import load_translations
from sys import argv
from utils import kwargs_dict

del kwargs_dict['keywords:letter_frequency']

def backreplace_phrase(text_string, input_phrase, target_phrase):
  """
  Replaces a phrase, preserving any surrounding single or double quotes.
  """
  escaped_phrase = re.escape(input_phrase)
  flexible_whitespace_phrase = re.sub(r'\\ ', r'\\s+', escaped_phrase)
  
  # Expanded Unicode ranges for scripts of the 22 scheduled languages of India.
  indian_scripts = (
      r'[\u0900-\u097F'  # Devanagari (Hindi, Marathi, Sanskrit, etc.)
      r'\u0980-\u09FF'  # Bengali-Assamese
      r'\u0A00-\u0A7F'  # Gurmukhi (Punjabi)
      r'\u0A80-\u0AFF'  # Gujarati
      r'\u0B00-\u0B7F'  # Odia
      r'\u0B80-\u0BFF'  # Tamil
      r'\u0C00-\u0C7F'  # Telugu
      r'\u0C80-\u0CFF'  # Kannada
      r'\u0D00-\u0D7F'  # Malayalam
      r'\u0600-\u06FF'  # Arabic (Urdu, Kashmiri, Sindhi)
      r'\u0750-\u077F'  # Arabic Supplement (for extended characters)
      r'\u08A0-\u08FF'  # Arabic Extended-A (for extended characters)
      r'\uABC0-\uABFF'  # Meetei Mayek (Manipuri)
      r'\u1C50-\u1C7F]' # Ol Chiki (Santali)
  )
  
  # Suffix is non-greedy (*?) so it doesn't consume a trailing quote.
  advanced_suffix = r'(?:(?!' + indian_scripts + r').)*?'

  # The main pattern is wrapped in capturing groups for the quotes.
  pattern = re.compile(
      r'([\'"]?)' +              # Group 1: Optional leading quote
      r'(' +                      # Group 2: The main text to be replaced
        r'\b' +
        flexible_whitespace_phrase +
        advanced_suffix +
        r'(?=\W|$)' +
      r')' +
      r'([\'"]?)',               # Group 3: Optional trailing quote
      re.IGNORECASE | re.UNICODE # Added re.UNICODE for proper word boundary handling
  )

  # The replacement string uses backreferences \1 and \3 to restore the quotes.
  replacement_string = r'\1' + target_phrase + r'\3'
  
  # Perform both steps of the main replacement and whitespace normalization.
  final_text = pattern.sub(replacement_string, text_string)

  # Define the pattern to find the target_phrase followed by non-English characters.
  # We escape the target_phrase to handle any special regex characters it might contain.
  escaped_target_phrase = re.escape(target_phrase)
  truncation_pattern = re.compile(f'({escaped_target_phrase})[^a-zA-Z\\s.'+"'"+'"]*')
  
  # Find and print the strings that will be truncated.
  for match in truncation_pattern.finditer(final_text):
      if match.group(0) != match.group(1):
              print(f"Truncating: '{match.group(0)}' to '{match.group(1)}'")

  # Perform the specific truncation for the target_phrase.
  text_after_truncation = truncation_pattern.sub(r'\1', final_text)
    
  return text_after_truncation


if __name__=="__main__":
    lang_code = argv[1]

    transformed_dataset = load_dataset("ai4bharat/IndicIFEval", "indicifeval-trans", split="en")

    input_filename = f"languages/{lang_code}/translated_dataset-{lang_code}.jsonl"
    output_filename = f"languages/{lang_code}/posttranslated_dataset-{lang_code}.jsonl"

    modified_data = load_dataset('json', data_files=input_filename, split='train')

    updates_map = {item['key']: item for item in modified_data}

    def apply_modifications(example):
        prompt_id = example['key']
        if prompt_id in updates_map:
            return updates_map[prompt_id]
        return example
    
    translated_dataset = transformed_dataset.map(apply_modifications)

    
    pretranslations = load_translations(f"languages/{lang_code}/pretranslations-{lang_code}.json")
    letter_mapping = load_translations(f'languages/{lang_code}/letter_mapping-{lang_code}.json')

    backtranslations = {v:k for k,v in pretranslations.items()}

    backletters = {v:k for k,v in letter_mapping.items()}

    prompt_list = translated_dataset["prompt"]
    kwargs_list = translated_dataset["kwargs"]

    posttranslated_prompt_list = []
    posttranslated_kwargs_list = []

    repeat_prompt_loc = {1012: ["\n", 0], 1129: ["\n", 0], 1139: ["\n", 0], 1281: ["\n", 0], 1480: ["\n", 0], 
             1518: ["\n", 0], 1561: ["\n", 0], 1627: ["\n", 0], 1656: ["\n", 0], 1906: ["\n", 0], 
             1922: ["\n", 0], 2063: ["\n", 0], 2071: ["\n", 0], 2192: ["\n", 0], 2337: ["\n", 0], 
             2355: ["\n", 0], 2465: ["\n", 0], 2482: ["\n", 0], 2585: ["\n", 0], 2713: ["\n", 0], 
             2739: ["\n", 0], 288: ["\n", 0], 2904: ["\n", 0], 3224: ["\n", 0], 3272: ["\n", 0], 
             332: ["\n", 0], 3369: ["\n", 0], 3371: ["\n", 0], 3505: ["\n", 0], 3563: ["\n", 0], 
             3633: ["\n", 0], 3718: ["\n", 0], 1546: ["\n\n", -1], 16: ["\n\n", -1], 1686: ["\n\n", -1], 
             1944: ["\n\n", -1], 2305: ["\n\n", -1], 2359: ["\n\n", -1], 3484: ["\n\n", -1], 3305: ["\n\n", -1],
             374: ["\n", 0]}


    for idx, ex in enumerate(translated_dataset):
        posttranslated_prompt = prompt_list[idx]
        posttranslated_kwargs = ex["kwargs"]

        if ex["resp_lang"] != "src":
            for j, instruction_id in enumerate(ex["instruction_id_list"]):
                if instruction_id in list(kwargs_dict.keys()):
                    words_phrases = ex["kwargs"][j][kwargs_dict[instruction_id]]
                    if isinstance(words_phrases, list):
                        for k, keyword in enumerate(words_phrases):
                            # Replace all occurrences of the pattern in the prompt with the backtranslated word/phrase'
                            if f"'{keyword}'" in posttranslated_prompt:
                                posttranslated_prompt = posttranslated_prompt.replace(f"'{keyword}'", f"'{backtranslations[keyword]}'")
                            elif f'"{keyword}' in posttranslated_prompt:
                                posttranslated_prompt = posttranslated_prompt.replace(f'"{keyword}"', f"'{backtranslations[keyword]}'")
                            else:
                                posttranslated_prompt = backreplace_phrase(posttranslated_prompt, keyword, backtranslations[keyword])
                            posttranslated_kwargs[j][kwargs_dict[instruction_id]][k] = backtranslations[keyword]
                    else:
                        # Replace all occurrences of the pattern in the prompt with the backtranslated word/phrase'
                        if f"'{words_phrases}'" in posttranslated_prompt:
                            posttranslated_prompt = posttranslated_prompt.replace(f"'{words_phrases}'", f"'{backtranslations[words_phrases]}'")
                        elif f'"{words_phrases}"' in posttranslated_prompt:
                            posttranslated_prompt = posttranslated_prompt.replace(f'"{words_phrases}"', f'"{backtranslations[words_phrases]}"')
                        else:
                            posttranslated_prompt = backreplace_phrase(posttranslated_prompt, words_phrases, backtranslations[words_phrases])
                        posttranslated_kwargs[j][kwargs_dict[instruction_id]] = backtranslations[words_phrases]

                if instruction_id == "keywords:letter_frequency" and ex['resp_lang'] != 'en':
                    letter = ex["kwargs"][j]["letter"]
                    if letter not in ["#", "!"]:
                        posttranslated_prompt = posttranslated_prompt.replace(letter, backletters[letter])
                        posttranslated_kwargs[j]["letter"] = backletters[letter]

        
        if "combination:repeat_prompt" in ex["instruction_id_list"]:
            j = ex["instruction_id_list"].index("combination:repeat_prompt")
            prompt_to_repeat = ex["kwargs"][j]["prompt_to_repeat"]
            repeat_prompt_ = repeat_prompt_loc[int(ex["key"])]
            posttranslated_kwargs[j]["prompt_to_repeat"] = posttranslated_prompt.split(repeat_prompt_[0])[repeat_prompt_[1]]

        if "startend:end_checker" in ex["instruction_id_list"]:
            j = ex["instruction_id_list"].index("startend:end_checker")
            if posttranslated_prompt.count('"') == 2:
                posttranslated_prompt = re.sub(r'(")[^"]*(")', rf'\1{ex["kwargs"][j]["end_phrase"]}\2', posttranslated_prompt, count=1)
        
        posttranslated_kwargs_list.append(posttranslated_kwargs)
        posttranslated_prompt_list.append(posttranslated_prompt)
        
    posttranslated_dataset = translated_dataset.map(lambda examples, idx: {'prompt': posttranslated_prompt_list[idx], 'kwargs': posttranslated_kwargs_list[idx]}, with_indices=True)

    posttranslated_dataset.to_pandas().to_json(output_filename, orient="records", lines=True, force_ascii=False)
