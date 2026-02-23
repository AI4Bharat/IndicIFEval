from datasets import load_dataset
import json
import random
from collections import defaultdict

# Set a seed for reproducibility if needed, or remove for true randomness
random.seed(42)

LANG_FULLNAME = {
    "asm": "Assamese",
    "ben": "Bengali",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "mar": "Marathi",
    "mal": "Malayalam",
    "npi": "Nepali",
    "ory": "Odia",
    "pan": "Punjabi",
    "san": "Sanskrit",
    "tam": "Tamil",
    "tel": "Telugu",
    "urd": "Urdu"
}

GENERATION_STYLES_CONTEXT = {
    "Summarization": "required context",
    "Formal Rewrite": "required context",
    "Informal Rewrite": "required context",

    "Articles or Blog Post": "context not required",
    "Screenplay": "context not required",
    "Email": "context not required",
    "Step-by-Step Instructions": "context not required",
    "Poem": "context not required",
    "Social Media Post": "context not required",
    "News Article": "context not required",
    "Creative Story": "context not required",
    "Dialogue Generation": "context not required",
    "Speech or Presentation": "context not required",
    "Question Generation": "context not required",
    "Essay": "context not required",
    "Technical Report": "context not required",
}

# Separate styles into two lists for efficient sampling
CONTEXT_REQUIRED_STYLES = [s for s, t in GENERATION_STYLES_CONTEXT.items() if t == "required context"]
CONTEXT_NOT_REQUIRED_STYLES = [s for s, t in GENERATION_STYLES_CONTEXT.items() if t == "context not required"]

def get_weighted_style():
    """
    Returns a tuple (style_name, is_context_required)
    Probability: 0.1 for context required, 0.9 for context not required.
    """
    if random.random() < 0.1:
        return random.choice(CONTEXT_REQUIRED_STYLES), True
    else:
        return random.choice(CONTEXT_NOT_REQUIRED_STYLES), False

def get_context_instruction(style, is_context_required):
    """
    Generates the specific instruction block for the prompt based on requirement.
    """
    if is_context_required:
        return (
            f"The intended style of the task is '{style}'. "
            "This instruction MUST require the provided context to be performed (e.g., rewriting or summarizing the context). "
            "You MUST explicitly write '[INSERT CONTEXT HERE]' in your generated instruction where the context should be placed."
        )
    else:
        return (
            f"The intended style of the task is '{style}'. "
            "The instruction MUST be grounded in the themes and topics of the provided context, but it must be answerable WITHOUT providing the context text itself. "
            "Do NOT include '[INSERT CONTEXT HERE]' or any placeholder references to the text. The user will not see the context."
        )

def prompt_contains_keyword(input_data, lang, style, is_context_required):
    context_instruction = get_context_instruction(style, is_context_required)
    
    prompt = f"""Role: You are an expert prompt engineer specializing in creating evaluation prompts for Large Language Models (LLMs). Your task is to design a high-quality, creative prompt for the IFEval benchmark (in {lang}), which is designed to test an LLM's ability to follow specific instructions and constraints.

Objective: I will provide you with a Constraint and an Example Context (in {lang}). Based on this input, you will generate a new prompt that instructs an LLM to perform a task while adhering to the given constraint. 

Strict Requirement: {context_instruction}

[START OF INPUT]
Constraint to Test:
{input_data["keyword"]} should appear {input_data["frequency"]} time(s).

Example Context (for grounding/theme):
{input_data["text"]}
[END OF INPUT]

Your Task:
Based on the Constraint to Test and the Example Context provided above, generate a single, new prompt in JSON format.

[START OF OUTPUT]
```json
{{
    "instruction": "[generated instruction with mention of constraint]",
    "constraint": "{input_data["keyword"]} should appear {input_data["frequency"]} time(s)."
}}
```
[END OF OUTPUT]

Requirements for the Generated Prompt:
- Natural Integration: The constraint must be woven seamlessly and logically into the prompt's instructions.
- Clear Task: The prompt must define a clear and achievable task for the LLM.
- Creative Context: The prompt should be engaging and provide a context that makes the constraint feel purposeful rather than arbitrary.
- No Meta-Language: Do not mention "LLM," "AI," "test," "evaluation," or "IFEval." The prompt should feel like a genuine request.
- Output: Provide only the generated prompt without any additional explanation or commentary in the form of json format containing two keys:
    1) `instruction`: The generated instruction string.
    2) `constraint`: The constraint description string.
- Safety: The generated instruction should be safe to use. If the context itself is unsafe or controversial, please generate "This context contains controversial themes" only.
"""
    return prompt

def prompt_forbidden_keyword(input_data, lang, style, is_context_required):
    context_instruction = get_context_instruction(style, is_context_required)
    
    prompt = f"""Role: You are an expert prompt engineer specializing in creating evaluation prompts for Large Language Models (LLMs). Your task is to design a high-quality, creative prompt for the IFEval benchmark (in {lang}), which is designed to test an LLM's ability to follow specific instructions and constraints.

Objective: I will provide you with a Constraint and an Example Context (in {lang}). Based on this input, you will generate a new prompt that instructs an LLM to perform a task while adhering to the given constraint.

Strict Requirement: {context_instruction}

[START OF INPUT]
Constraint to Test:
{input_data["keyword"]} should not appear in the response (forbidden keyword).

Example Context (for grounding/theme):
{input_data["text"]}
[END OF INPUT]

Your Task:
Based on the Constraint to Test and the Example Context provided above, generate a single, new prompt in JSON format.

[START OF OUTPUT]
```json
{{
    "instruction": "[generated instruction with mention of constraint]",
    "constraint": "{input_data["keyword"]} should not appear in the response."
}}
```
[END OF OUTPUT]

Requirements for the Generated Prompt:
- Natural Integration: The constraint must be woven seamlessly and logically into the prompt's instructions.
- Clear Task: The prompt must define a clear and achievable task for the LLM.
- Creative Context: The prompt should be engaging and provide a context that makes the constraint feel purposeful rather than arbitrary.
- No Meta-Language: Do not mention "LLM," "AI," "test," "evaluation," or "IFEval." The prompt should feel like a genuine request.
- Output: Provide only the generated prompt without any additional explanation or commentary in the form of json format containing two keys:
    1) `instruction`: The generated instruction string.
    2) `constraint`: The constraint description string.
- Safety: The generated instruction should be safe to use. If the context itself is unsafe or controversial, please generate "This context contains controversial themes" only.
"""
    return prompt


def prompt_sentences_paragraph(input_data, lang, style, is_context_required):
    context_instruction = get_context_instruction(style, is_context_required)

    prompt = f"""Role: You are an expert prompt engineer specializing in creating evaluation prompts for Large Language Models (LLMs). Your task is to design a high-quality, creative prompt for the IFEval benchmark (in {lang}), which is designed to test an LLM's ability to follow specific instructions and constraints.

Objective: I will provide you with a Constraint and an Example Context (in {lang}). Based on this input, you will generate a new prompt that instructs an LLM to perform a task while adhering to the given constraint.

Strict Requirement: {context_instruction}

[START OF INPUT]
Constraint to Test:
The response should have {input_data["sentences"]} sentence(s) and {input_data["paragraphs"]} paragraph(s).

Example Context (for grounding/theme):
{input_data["text"]}
[END OF INPUT]

Your Task:
Based on the Constraint to Test and the Example Context provided above, generate a single, new prompt in JSON format.

[START OF OUTPUT]
```json
{{
    "instruction": "[generated instruction with mention of constraint]",
    "constraint": "The response should have {input_data["sentences"]} sentence(s) and {input_data["paragraphs"]} paragraph(s)."
}}
```
[END OF OUTPUT]

Requirements for the Generated Prompt:
- Natural Integration: The constraint must be woven seamlessly and logically into the prompt's instructions.
- Clear Task: The prompt must define a clear and achievable task for the LLM.
- Creative Context: The prompt should be engaging and provide a context that makes the constraint feel purposeful rather than arbitrary.
- No Meta-Language: Do not mention "LLM," "AI," "test," "evaluation," or "IFEval." The prompt should feel like a genuine request.
- Output: Provide only the generated prompt without any additional explanation or commentary in the form of json format containing two keys:
    1) `instruction`: The generated instruction string.
    2) `constraint`: The constraint description string.
- Safety: The generated instruction should be safe to use. If the context itself is unsafe or controversial, please generate "This context contains controversial themes" only.
"""

    if input_data['sentences'] == 0:
        prompt = prompt.replace(" 0 sentence(s) and", "")
        prompt = prompt.replace('"The response should have 0 sentence(s) and', '"The response should have') # Adjust constraint string in example
    if input_data['paragraphs'] == 1:
        prompt = prompt.replace(" and 1 paragraph(s)", "")
        prompt = prompt.replace(' and 1 paragraph(s)."', '."') # Adjust constraint string in example

    return prompt


def prompt_firstword(input_data, lang, style, is_context_required):
    context_instruction = get_context_instruction(style, is_context_required)

    prompt = f"""Role: You are an expert prompt engineer specializing in creating evaluation prompts for Large Language Models (LLMs). Your task is to design a high-quality, creative prompt for the IFEval benchmark (in {lang}), which is designed to test an LLM's ability to follow specific instructions and constraints.

Objective: I will provide you with a Constraint and an Example Context (in {lang}). Based on this input, you will generate a new prompt that instructs an LLM to perform a task while adhering to the given constraint.

Strict Requirement: {context_instruction}

[START OF INPUT]
Constraint to Test:
{input_data["keyword"]} should be the first word in the response.

Context for Grounding (for grounding/theme):
{input_data["text"]}
[END OF INPUT]

Your Task:
Based on the Constraint to Test and the Example Context provided above, generate a single, new prompt in JSON format.

[START OF OUTPUT]
```json
{{
    "instruction": "[generated instruction with mention of constraint]",
    "constraint": "{input_data["keyword"]} should be the first word in the response."
}}
```
[END OF OUTPUT]

Requirements for the Generated Prompt:
- Natural Integration: The constraint must be woven seamlessly and logically into the prompt's instructions.
- Clear Task: The prompt must define a clear and achievable task for the LLM.
- Creative Context: The prompt should be engaging and provide a context that makes the constraint feel purposeful rather than arbitrary.
- No Meta-Language: Do not mention "LLM," "AI," "test," "evaluation," or "IFEval." The prompt should feel like a genuine request.
- Output: Provide only the generated prompt without any additional explanation or commentary in the form of json format containing two keys:
    1) `instruction`: The generated instruction string.
    2) `constraint`: The constraint description string.
- Safety: The generated instruction should be safe to use. If the context itself is unsafe or controversial, please generate "This context contains controversial themes" only.
"""
    return prompt


if __name__ == "__main__":
    langs = ["asm", "ben", "guj", "hin", "kan", "mar", "mal", "npi", "ory", "pan", "san", "tam", "tel", "urd"]
    
    REQUESTS = []
    IDS = []
    
    # Track statistics: style_stats[lang][style_name] = count
    style_stats = defaultdict(lambda: defaultdict(int))

    for lang in langs:
        print(f"Processing {lang}...")

        data_path = f"data/sangraha-verified-{lang}"

        # SAMPLES-1: Keyword Frequency
        try:
            samples1 = load_dataset("json", data_files=f"{data_path}/samples1.jsonl", split="train")
            for row in samples1:
                # Generate a style per request for better distribution
                style, is_context_required = get_weighted_style()
                style_stats[lang][style] += 1
                
                if row['frequency'] == 6:
                    prompt = prompt_forbidden_keyword(row, LANG_FULLNAME[lang], style, is_context_required)
                else:
                    prompt = prompt_contains_keyword(row, LANG_FULLNAME[lang], style, is_context_required)

                REQUESTS.append({"key": row['id'], "request": {"contents": [{"parts": [{"text": prompt}], "role": "user"}], "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.9, "topP": 0.95}}})
                IDS.append(row['id'])
        except Exception as e:
            print(f"Skipping samples1 for {lang}: {e}")

        # SAMPLES-2: Sentences/Paragraphs
        try:
            samples2 = load_dataset("json", data_files=f"{data_path}/samples2.jsonl", split="train")
            for row in samples2:
                style, is_context_required = get_weighted_style()
                style_stats[lang][style] += 1
                prompt = prompt_sentences_paragraph(row, LANG_FULLNAME[lang], style, is_context_required)

                REQUESTS.append({"key": row['id'], "request": {"contents": [{"parts": [{"text": prompt}], "role": "user"}], "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.9, "topP": 0.95}}})
                IDS.append(row['id'])
        except Exception as e:
            print(f"Skipping samples2 for {lang}: {e}")

        # SAMPLES-3: First Word
        try:
            samples3 = load_dataset("json", data_files=f"{data_path}/samples3.jsonl", split="train")
            for row in samples3:
                style, is_context_required = get_weighted_style()
                style_stats[lang][style] += 1
                prompt = prompt_firstword(row, LANG_FULLNAME[lang], style, is_context_required)

                REQUESTS.append({"key": row['id'], "request": {"contents": [{"parts": [{"text": prompt}], "role": "user"}], "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.9, "topP": 0.95}}})
                IDS.append(row['id'])
        except Exception as e:
            print(f"Skipping samples3 for {lang}: {e}")

        
    seen = set()
    UNIQUE_REQUESTS = []
    
    for item in REQUESTS:
        if item['key'] not in seen:
            seen.add(item['key'])
            UNIQUE_REQUESTS.append(item)

    print(f"Total Unique Requests: {len(UNIQUE_REQUESTS)}")

    # Create a JSONL file
    length = len(UNIQUE_REQUESTS)
    with open("ifeval-batch-requests.jsonl", "w") as f:
        for i, req in enumerate(UNIQUE_REQUESTS):
            f.write(json.dumps(req, ensure_ascii=False) + '\n')
            if i % 100 == 0:
                print(f"\rWriting file: {i}/{length}", end='\r')
    print("\nDone!")

    # Print Style Summary
    print("\n" + "="*50)
    print("STYLE DISTRIBUTION SUMMARY PER LANGUAGE")
    print("="*50)

    for lang in langs:
        print(f"\nLanguage: {LANG_FULLNAME[lang]} ({lang})")
        total_requests = sum(style_stats[lang].values())
        
        if total_requests == 0:
            print("  No requests processed.")
            continue
            
        # Sort by count descending
        sorted_styles = sorted(style_stats[lang].items(), key=lambda x: x[1], reverse=True)
        
        for style, count in sorted_styles:
            percentage = (count / total_requests) * 100
            print(f"  - {style}: {count} ({percentage:.2f}%)")
            
    print("\n" + "="*50)