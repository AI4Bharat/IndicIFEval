import sys

from utils import load_translations

# if len(sys.argv) > 1:
#     sys.stdout = open(sys.argv[1], "w")

lang_code = sys.argv[1]

from datasets import load_dataset

# The local JSONL file containing your manual modifications
modified_jsonl_path = sys.argv[2]
# modified_jsonl_path = "transformed_default_ifeval.jsonl"

transformed_dataset = load_dataset('json', data_files=modified_jsonl_path, split='train')

# constrained_response = {'hi':["मेरा जवाब है, हाँ", "मेरा जवाब है, नहीं", "मेरा जवाब है, शायद"],
#                         'ta':["என் பதில் 'ஆம்'", "என் பதில் 'இல்லை'", "என் பதில் 'இருக்கலாம்'"],
#                         'ml': ["എന്റെ ഉത്തരം അതെ എന്നാണ്", "എൻ്റെ ഉത്തരം ഇല്ല എന്നാണ്", "എൻ്റെ ഉത്തരം ആകാം എന്നാണ്"],
#                         'te': ["నా సమాధానం 'అవును'", "నా సమాధానం 'కాదు'", "నా సమాధానం 'కావచ్చు'"],
#                         'kn': ["ನನ್ನ ಉತ್ತರ ಹೌದು", "ನನ್ನ ಉತ್ತರ ಇಲ್ಲ", "ನನ್ನ ಉತ್ತರ ಬಹುಶಃ ಹೌದು"]
# }

pretranslations = load_translations(f"languages/{lang_code}/pretranslations-{lang_code}.json")

constrained_response_list = ["My answer is yes.", "My answer is no.", "My answer is maybe."]
for i in range(3):
    constrained_response_list[i] =  pretranslations[constrained_response_list[i]]

# ==================== detectable_content:postscript ====================
x = transformed_dataset.filter(lambda ex: "detectable_content:postscript" in ex["instruction_id_list"])
print(f"No. of examples for detectable_content:postscript: {len(x)}")
x_mismatched = x.filter(lambda ex: ex["kwargs"][ex["instruction_id_list"].index("detectable_content:postscript")]["postscript_marker"] not in ex["prompt"])
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== detectable_format:constrained_response ====================
x = transformed_dataset.filter(lambda ex: "detectable_format:constrained_response" in ex["instruction_id_list"])
print(f"No. of examples for detectable_format:constrained_response: {len(x)}")
x_mismatched = x.filter(lambda ex: not all([keyword in ex["prompt"] for keyword in constrained_response_list]))
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== keywords:existence ====================
x = transformed_dataset.filter(lambda ex: "keywords:existence" in ex["instruction_id_list"])
print(f"No. of examples for keywords:existence: {len(x)}")
x_mismatched = x.filter(lambda ex: not all(keyword.lower() in ex["prompt"].lower() for keyword in ex["kwargs"][ex["instruction_id_list"].index("keywords:existence")]["keywords"]))
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
x_mismatched = x.filter(lambda ex: len(ex["kwargs"][ex["instruction_id_list"].index("keywords:existence")]["keywords"]) != len(set(ex["kwargs"][ex["instruction_id_list"].index("keywords:existence")]["keywords"])))
if len(x_mismatched) != 0:
    print(f"Warning: keys {list(x_mismatched['key'])} has duplicate keywords")
print("\n======================================================\n")


# ==================== keywords:forbidden_words ====================
x = transformed_dataset.filter(lambda ex: "keywords:forbidden_words" in ex["instruction_id_list"])
print(f"No. of examples for keywords:forbidden_words: {len(x)}")
x_mismatched = x.filter(lambda ex: not all(keyword.lower() in ex["prompt"].lower() for keyword in ex["kwargs"][ex["instruction_id_list"].index("keywords:forbidden_words")]["forbidden_words"]))
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
x_mismatched = x.filter(lambda ex: len(ex["kwargs"][ex["instruction_id_list"].index("keywords:forbidden_words")]["forbidden_words"]) != len(set(ex["kwargs"][ex["instruction_id_list"].index("keywords:forbidden_words")]["forbidden_words"])))
if len(x_mismatched) != 0:
    print(f"Warning: keys {list(x_mismatched['key'])} has duplicate keywords")
print("\n======================================================\n")


# ==================== keywords:frequency ====================
x = transformed_dataset.filter(lambda ex: "keywords:frequency" in ex["instruction_id_list"])
print(f"No. of examples for keywords:frequency: {len(x)}")
x_mismatched = x.filter(lambda ex: ex["kwargs"][ex["instruction_id_list"].index("keywords:frequency")]["keyword"].lower() not in ex["prompt"].lower())
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== keywords:letter_frequency ====================
x = transformed_dataset.filter(lambda ex: "keywords:letter_frequency" in ex["instruction_id_list"])
print(f"No. of examples for keywords:letter_frequency: {len(x)}")
x_mismatched = x.filter(lambda ex: ex["kwargs"][ex["instruction_id_list"].index("keywords:letter_frequency")]["letter"].lower() not in ex["prompt"].lower())
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== length_constraints:nth_paragraph_first_word ====================
x = transformed_dataset.filter(lambda ex: "length_constraints:nth_paragraph_first_word" in ex["instruction_id_list"])
print(f"No. of examples for length_constraints:nth_paragraph_first_word: {len(x)}")
x_mismatched = x.filter(lambda ex: ex["kwargs"][ex["instruction_id_list"].index("length_constraints:nth_paragraph_first_word")]["first_word"].lower() not in ex["prompt"].lower())
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== startend:end_checker ====================
x = transformed_dataset.filter(lambda ex: "startend:end_checker" in ex["instruction_id_list"])
print(f"No. of examples for startend:end_checker: {len(x)}")
x_mismatched = x.filter(lambda ex: ex["kwargs"][ex["instruction_id_list"].index("startend:end_checker")]["end_phrase"].lower() not in ex["prompt"].lower())
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== combination:repeat_prompt ====================
x = transformed_dataset.filter(lambda ex: "combination:repeat_prompt" in ex["instruction_id_list"])
print(f"No. of examples for combination:repeat_prompt: {len(x)}")
x_mismatched = x.filter(lambda ex: ex["kwargs"][ex["instruction_id_list"].index("combination:repeat_prompt")]["prompt_to_repeat"].lower() not in ex["prompt"].lower())
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== combination:two_responses ====================
import re
x = transformed_dataset.filter(lambda ex: "combination:two_responses" in ex["instruction_id_list"])
print(f"No. of examples for combination:two_responses: {len(x)}")
x_mismatched = x.filter(lambda ex: not bool(re.search(r'(?<!\*)\*{6}(?!\*)', ex["prompt"])))
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")



# ==================== detectable_format:multiple_sections ====================
x = transformed_dataset.filter(lambda ex: "detectable_format:multiple_sections" in ex["instruction_id_list"])
print(f"No. of examples for detectable_format:multiple_sections: {len(x)}")
x_mismatched = x.filter(lambda ex: ex["kwargs"][ex["instruction_id_list"].index("detectable_format:multiple_sections")]["section_spliter"].lower() not in ex["prompt"].lower())
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== detectable_format:json_format ====================
import re
x = transformed_dataset.filter(lambda ex: "detectable_format:json_format" in ex["instruction_id_list"])
print(f"No. of examples for detectable_format:json_format: {len(x)}")
x_mismatched = x.filter(lambda ex: not bool(re.search(r'(?<!`)`{3}(?!`)', ex["prompt"])))
x_mismatched = x_mismatched.filter(lambda ex: "JSON" not in ex["prompt"])
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== detectable_format:number_bullet_lists ====================
import re
x = transformed_dataset.filter(lambda ex: "detectable_format:number_bullet_lists" in ex["instruction_id_list"])
print(f"No. of examples for detectable_format:number_bullet_lists: {len(x)}")
x_mismatched = x.filter(lambda ex: not bool(re.search(r'(?<!\*)\*{1}(?!\*)', ex["prompt"])))
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== detectable_format:number_bullet_lists ====================
import re
x = transformed_dataset.filter(lambda ex: "detectable_format:number_highlighted_sections" in ex["instruction_id_list"])

print(f"No. of examples for detectable_format:number_highlighted_sections: {len(x)}")
x_mismatched = x.filter(lambda ex: not all(re.fullmatch(r'\*.+\*', part) for part in re.findall(r'\*.*?\*', ex["prompt"])))
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


# ==================== detectable_format:title ====================
import re
x = transformed_dataset.filter(lambda ex: "detectable_format:title" in ex["instruction_id_list"])
print(f"No. of examples for detectable_format:title: {len(x)}")
x_mismatched = x.filter(lambda ex: not all(re.fullmatch(r'<<.+>>', part) for part in re.findall(r'<<.*?>>', ex["prompt"])))
print(f"No. of unexpected rows: {len(x_mismatched)}")
print(f"Keys of unexpected rows: {list(x_mismatched['key'])}")
print("\n======================================================\n")


if len(sys.argv) > 1:
    sys.stdout.close()
