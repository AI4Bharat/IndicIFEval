# IndicIFEval

## Dataset Setup
1. Prepare your input file. Make sure you have a text file at the path `languages/en/pretranslations.txt` with one English wprd/phrase per line.
2. Run `pretranslate.py`:
    - To use the SarvamAI API:
    ```python pretranslate.py hi --use-api --sarvamai_token "YOUR_API_KEY"```
    - To use a Hugging Face model:
    ```python pretranslate.py hi --use-hf```
3. Modify the pretranslations at `languages/xx/pretranslations.json` (where `xx` is your language code). You can use  ```python pretranslation_lookup.py <word>``` wherever required.
4. After verification/modification apply the pretranslations to the English dataset.
    ```python apply_pretranslations.py xx`
5. Translate the pretranslated dataset to language `xx` by running `translate.py` with similar cli arguments as pretranslate.py
6. After translations, perform post-translations.
    ```python apply_posttranslation.py xx```
7. The resulting datasets will be saved at `languages/en/posttranslated_dataset-xx.jsonl` with similar names at each stage (`pretranslated`/`translated`/`posttranslated`).
8. You can now use the post-translated dataset to make manual modifications (post-editing translations).
9. Before verifying the post-translated dataset ensure that all the IFEval constraint arguments are coherent to the prompt. You can check this by running the following command:
    ```python constraint_verification.py <optional_filename_for_stdout.txt>```
