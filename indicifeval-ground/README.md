This contains scripts to process IndicIFEval-Ground. It extracts context documents based on specific text properties and formulates prompts to test instruction-following capabilities.

## Prerequisites

You must install the datasets, indic-nlp-library, tqdm, and regex Python packages before running the scripts.

## Usage

Follow these ordered steps to execute the full pipeline for a target language. The examples below use Hindi, denoted by the language code hin.

### Step 1: Download and Process Dataset
Download the Hugging Face dataset and split it into chunked files.

```bash
python process_dataset.py --lang hin --num_processes 128
```

### Step 2: Extract Top Vocabulary
Analyze the corpus to find meaningful words using TF-IDF. This generates a text file containing the top words to be used for keyword constraints.

```bash
python get_word_samples_tfidf.py --lang hin --num_processes 128 --top_k 500
```

### Step 3: Extract Keyword Frequency Samples
Search the processed chunks for documents where a specific word appears an exact number of times. This creates the first set of samples.

```bash
python search_words_in_paragraphs.py नमस्ते --lang hin --count 3 --paragraphs 1 --sentences 6
```

### Step 4: Extract Structural Constraint Samples
Find documents that exactly match specific sentence and paragraph length constraints. This skips documents already collected in the previous step and creates the second set of samples.

```bash
python search_paragraphs_sentences.py --lang hin --paragraphs 2 --sentences 10
```

### Step 5: Extract First Word Samples
Find documents that start with a specific target word. This creates the third set of samples.
```bash
python search_firstword_in_paragraphs.py नमस्ते --lang hin --count 1
```
