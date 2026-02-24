from datasets import load_dataset
from apply_pretranslations import kwargs_dict
from sys import argv

if __name__ == "__main__":
    transformed_dataset = load_dataset("ai4bharat/IndicIFEval", "indicifeval-trans", split="en")

    wordf = []

    for ex in transformed_dataset:
        for j, instruction_id in enumerate(ex["instruction_id_list"]):
            if instruction_id in list(kwargs_dict.keys()):
                words_phrases = ex["kwargs"][j][kwargs_dict[instruction_id]]
                if isinstance(words_phrases, list):
                    for key in words_phrases:
                        wordf.append(key)
                else:
                    wordf.append(words_phrases)

wordf = list(set(wordf))
wordf.extend(["My answer is yes.", "My answer is no.", "My answer is maybe."])

with open("languages/en/pretranslations.txt", 'w') as f:
    for key in wordf:
        if len(key) != 1: # for single characters
            f.write(key+"\n")
    