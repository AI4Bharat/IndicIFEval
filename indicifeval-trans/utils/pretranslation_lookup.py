from datasets import load_dataset
from apply_pretranslations import kwargs_dict
from sys import argv

if __name__ == "__main__":
    transformed_dataset = load_dataset("ai4bharat/IndicIFEval", "indicifeval-trans", split="en")

    word = argv[1].strip()

    for ex in transformed_dataset:
        for j, instruction_id in enumerate(ex["instruction_id_list"]):
            if instruction_id in list(kwargs_dict.keys()):
                words_phrases = ex["kwargs"][j][kwargs_dict[instruction_id]]
                if isinstance(words_phrases, list):
                    for key in words_phrases:
                        if key == word:
                            print(ex["prompt"]+"\n\n")
                else:
                    if words_phrases == word:
                        print(ex["prompt"]+"\n\n")