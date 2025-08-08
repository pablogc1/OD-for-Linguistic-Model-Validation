#!/usr/bin/env python3
# generate_ai_definitions_batched_v4.py
#
# FINAL VERSION: Includes a pre-run cost estimator and user confirmation step.
# It uses the tiktoken library for accurate token counting.

import pathlib
import json
import time
import re
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm
import tiktoken # <-- Import the official tokenizer

# ==============================================================================
#                      Configuration
# ==============================================================================

# --- API and Model ---
API_KEY      = ""
MODEL = "gpt-4o"
TEMP = 0.0
MAX_RETRY = 3
BACKOFF = 2

# --- Batching Configuration ---
BATCH_SIZE = 100

# --- File Paths and Word Selection ---
SOURCE_DEFINITIONS_FILE = pathlib.Path("/Users/pablo/Downloads/extracted_definitions.txt")
OUTPUT_AI_DEFINITIONS_FILE = pathlib.Path("/Users/pablo/Downloads/ai_generated_definitions.txt")
START_INDEX = 1830
END_INDEX = 21112
WORD_LIMIT = None

# --- Stricter System Prompt ---
# --- Stricter System Prompt with 4-Word Rule and Better Example ---
SYSTEM_PROMPT = """
You are a lexicographer creating a dataset for a computational linguistics project. Your task is to provide a definition for each word you receive.
Follow these rules with ABSOLUTE STRICTNESS. There are NO exceptions.
--- RULES ---
1.  **ALLOWED WORD TYPES:** The definition MUST be a space-separated list containing ONLY these three types of words: Nouns, Adjectives, Verbs in their base infinitive form.
2.  **FORBIDDEN WORD TYPES:** You are ABSOLUTELY FORBIDDEN from using: Adverbs (NO 'not', 'very'), Prepositions (NO 'by', 'with', 'in'), Conjunctions (NO 'and', 'or'), Articles (NO 'a', 'the'), Pronouns, or any conjugated verb forms.
3.  **FORMAT:** You MUST return a single, raw JSON array. Each object must contain the "word" and its "definition".
4.  **DEFINITION LENGTH:** Each definition MUST be EXACTLY 5 words long.
--- EXAMPLES ---
Example Request: ["invisible", "gravity"]
Example CORRECT Output: [{"word": "invisible", "definition": "impossible see perceive physical eye"}, {"word": "gravity", "definition": "universal natural force attract physical body"}]
Example INCORRECT Output: [{"word": "invisible", "definition": "not possible perceive by eye"}] (WRONG: not 5 words, uses forbidden 'not' and 'by')
Produce only the final JSON array and no other text.
""".strip()

# ==============================================================================
#                 *** NEW: COST ESTIMATION FUNCTION ***
# ==============================================================================
def estimate_cost(words_to_process):
    """Calculates and displays the estimated cost for the job."""
    print("\n--- Calculating Cost Estimate ---")
    
    # Official pricing for gpt-4o
    PRICE_INPUT_PER_MILLION = 5.00
    PRICE_OUTPUT_PER_MILLION = 15.00
    
    # Estimate for an average output definition per word
    AVG_OUTPUT_TOKENS_PER_WORD = 25 

    try:
        encoding = tiktoken.encoding_for_model(MODEL)
    except Exception:
        # Fallback for older tiktoken versions
        encoding = tiktoken.get_encoding("cl100k_base")

    # Calculate input tokens
    tokens_in_system_prompt = len(encoding.encode(SYSTEM_PROMPT))
    total_input_tokens = 0
    num_batches = 0
    
    for i in range(0, len(words_to_process), BATCH_SIZE):
        batch = words_to_process[i : i + BATCH_SIZE]
        user_content = json.dumps(batch)
        total_input_tokens += tokens_in_system_prompt + len(encoding.encode(user_content))
        num_batches += 1

    # Calculate estimated output tokens
    total_output_tokens = len(words_to_process) * AVG_OUTPUT_TOKENS_PER_WORD
    
    # Calculate final cost
    input_cost = (total_input_tokens / 1_000_000) * PRICE_INPUT_PER_MILLION
    output_cost = (total_output_tokens / 1_000_000) * PRICE_OUTPUT_PER_MILLION
    total_cost = input_cost + output_cost

    print(f"Model: {MODEL}")
    print(f"Words to process: {len(words_to_process):,}")
    print(f"Batches to run: {num_batches:,}")
    print("-" * 35)
    print(f"Estimated INPUT tokens: {total_input_tokens:,} (~${input_cost:.4f})")
    print(f"Estimated OUTPUT tokens: {total_output_tokens:,} (~${output_cost:.4f})")
    print("-" * 35)
    print(f"ESTIMATED TOTAL COST: ${total_cost:.2f}")
    print("-" * 35)

    return total_cost

def parse_json_array_from_response(text: str):
    """Safely extracts a JSON array from a string."""
    match = re.search(r'\[.*\]', text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError: return None
    return None

# ==============================================================================
#                           Main Execution
# ==============================================================================
def main():
    if "YOUR_API_KEY" in API_KEY:
        print("--> FATAL ERROR: Please replace 'sk-YOUR_API_KEY_HERE' with your actual OpenAI API key.")
        exit(1)

    if not SOURCE_DEFINITIONS_FILE.exists():
        print(f"--> FATAL ERROR: Source file '{SOURCE_DEFINITIONS_FILE}' not found.")
        exit(1)

    print(f"--> Loading headwords from '{SOURCE_DEFINITIONS_FILE}'...")
    all_headwords = [line.split(":", 1)[0].strip() for line in open(SOURCE_DEFINITIONS_FILE, "r", encoding="utf-8") if ":" in line]
    target_words = all_headwords[START_INDEX - 1 : END_INDEX]
    
    if WORD_LIMIT: target_words = target_words[:WORD_LIMIT]

    processed_words = set()
    if OUTPUT_AI_DEFINITIONS_FILE.exists():
        with open(OUTPUT_AI_DEFINITIONS_FILE, "r", encoding="utf-8") as f_read:
            processed_words = {line.split(":", 1)[0].strip() for line in f_read if ":" in line}
        print(f"--> Found {len(processed_words)} words already processed.")

    words_to_process = [w for w in target_words if w not in processed_words]
    if not words_to_process:
        print("\n--> All target words have already been processed. Nothing to do.")
        return
        
    # --- *** NEW: ESTIMATE COST AND ASK FOR CONFIRMATION *** ---
    estimate_cost(words_to_process)
    
    proceed = input("--> Do you want to proceed with this job? (yes/no): ").lower()
    if proceed != 'yes':
        print("--> Aborting job as requested.")
        return
    
    print("\n--> Starting API calls...")
    # --- End of new section ---

    client = OpenAI(api_key=API_KEY)
    
    with open(OUTPUT_AI_DEFINITIONS_FILE, "a", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(words_to_process), BATCH_SIZE), desc="Processing Batches"):
            batch = words_to_process[i : i + BATCH_SIZE]
            user_content = json.dumps(batch)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
            
            tries = 0
            response_text = None
            while True:
                try:
                    response = client.chat.completions.create(model=MODEL, temperature=TEMP, messages=messages)
                    response_text = response.choices[0].message.content
                    break
                except (RateLimitError, APIError) as e:
                    tries += 1
                    if tries > MAX_RETRY:
                        print(f"\n--> API failure for batch starting with '{batch[0]}'. Skipping. Error: {e}")
                        break
                    time.sleep(BACKOFF ** tries)
            
            if response_text is None: continue

            results_list = parse_json_array_from_response(response_text)

            if results_list:
                for item in results_list:
                    if isinstance(item, dict) and "word" in item and "definition" in item:
                        f_out.write(f"{item['word']}: {item['definition'].strip()}\n")
                f_out.flush()
            else:
                print(f"\n--> Warning: Could not parse a valid JSON array for batch starting with '{batch[0]}'. Skipping.")

    print("\n--> Process finished successfully.")
    print(f"--> AI-generated lexicon saved to '{OUTPUT_AI_DEFINITIONS_FILE}'")

if __name__ == '__main__':
    main()