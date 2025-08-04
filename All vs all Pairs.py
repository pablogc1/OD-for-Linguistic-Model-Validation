
################################################################################################################################################################################

[w007104@login2 ~]$ cat sbatch_cleaner.slurm
#!/bin/bash
#SBATCH --job-name=clean_definitions
#SBATCH --output=clean_definitions_%j.out
#SBATCH --error=clean_definitions_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Purge modules and load the same Python environment as your other jobs
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare

# Execute the Python script.
# **IMPORTANT**: Make sure the python script is named 'clean_definitions_with_logging.py'
# or change the name in the line below to match.
python3 -u clean_definitions_with_logging.py
[w007104@login2 ~]$ cat clean_definitions_with_logging.py
import os
from tqdm import tqdm
from collections import Counter

# ==============================================================================
#                      Configuration & File Paths
# ==============================================================================
# --- File names are hardcoded here to match your workflow ---
# This is the original, uncleaned definitions file that the script will read.
INPUT_DEFINITIONS_FILE = "extracted_definitions.txt"

# This is the new, cleaned file that the script will create.
OUTPUT_CLEANED_FILE = "extracted_definitions_cleaned.txt"

# --- New Log File Names ---
DANGLING_WORDS_LOG_FILE = "dangling_words_log.txt"
REMOVED_HEADWORDS_LOG_FILE = "removed_headwords_log.txt"

# ==============================================================================
#                            Main Cleaning Logic
# ==============================================================================
def main():
    """
    Main function to load, iteratively clean, and save the definitions file,
    while logging all removed words for detailed analysis.
    """
    if not os.path.exists(INPUT_DEFINITIONS_FILE):
        print(f"--> FATAL ERROR: Input file '{INPUT_DEFINITIONS_FILE}' not found in the current directory.")
        print("--> Please ensure the definitions file is present before running.")
        exit(1) # Exit with an error code

    # --- 1. Initial Load ---
    print(f"--> Loading initial definitions from '{INPUT_DEFINITIONS_FILE}'...")
    definitions = {}
    original_order = [] # Keep track of the original word order
    with open(INPUT_DEFINITIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line: continue
            
            head, def_text = line.split(":", 1)
            head = head.strip().lower()
            tokens = def_text.strip().split()

            if head not in definitions: # Avoid duplicates if any
                original_order.append(head)
            definitions[head] = tokens
    
    initial_word_count = len(definitions)
    print(f"--> Loaded {initial_word_count} initial definitions.")

    # --- Initialize logging data structures ---
    all_dangling_words_found = Counter()
    all_removed_headwords = set()

    # --- 2. Iterative Pruning Loop ---
    print("\n--> Starting iterative pruning process...")
    iteration = 0
    while True:
        iteration += 1
        
        valid_headwords = set(definitions.keys())
        
        print(f"\n--- Iteration {iteration} (Current vocabulary size: {len(valid_headwords)}) ---")
        
        # --- Step A: Prune dangling words ---
        for headword in tqdm(list(definitions.keys()), desc="Pruning dangling words"):
            # Identify dangling words for this specific definition
            dangling_in_this_def = [token for token in definitions[headword] if token not in valid_headwords]
            if dangling_in_this_def:
                all_dangling_words_found.update(dangling_in_this_def)
            
            # Create the new, cleaned definition
            definitions[headword] = [token for token in definitions[headword] if token in valid_headwords]
        
        print(f"   -> Pruning pass complete. Found {len(all_dangling_words_found)} unique dangling words so far.")

        # --- Step B: Remove headwords with now-empty definitions ---
        headwords_to_remove = {
            headword for headword, definition_tokens in definitions.items() if not definition_tokens
        }
        
        if headwords_to_remove:
            print(f"   -> Found {len(headwords_to_remove)} headwords with newly empty definitions. Removing them.")
            all_removed_headwords.update(headwords_to_remove)
            for headword in headwords_to_remove:
                del definitions[headword]
        else:
            print("   -> No empty definitions found in this pass.")

        # --- Step C: Check for termination condition ---
        if not headwords_to_remove:
            print("\n--> Pruning complete. Lexicon is now stable and self-contained.")
            break
            
    # --- 3. Final Report and Save ---
    final_word_count = len(definitions)
    total_words_removed = initial_word_count - final_word_count
    
    print("\n" + "="*50)
    print("                 CLEANING SUMMARY")
    print("="*50)
    print(f"Initial Vocabulary Size: {initial_word_count}")
    print(f"Final Vocabulary Size:   {final_word_count}")
    print(f"Total Headwords Removed: {total_words_removed}")
    print("="*50)

    # --- 4. Write Log Files ---
    print(f"\n--> Writing {len(all_dangling_words_found)} unique dangling words to '{DANGLING_WORDS_LOG_FILE}'...")
    with open(DANGLING_WORDS_LOG_FILE, "w", encoding="utf-8") as f_log:
        f_log.write("# Word, Count (how many times it appeared as a dangling word)\n")
        # Sort by frequency, descending
        for word, count in all_dangling_words_found.most_common():
            f_log.write(f"{word}, {count}\n")
            
    print(f"--> Writing {len(all_removed_headwords)} removed headwords to '{REMOVED_HEADWORDS_LOG_FILE}'...")
    with open(REMOVED_HEADWORDS_LOG_FILE, "w", encoding="utf-8") as f_log:
        f_log.write("# Headwords removed because their definitions became empty\n")
        # Sort alphabetically for easy reading
        for word in sorted(list(all_removed_headwords)):
            f_log.write(f"{word}\n")

    # --- 5. Save Cleaned Definitions File ---
    print(f"\n--> Saving {final_word_count} cleaned definitions to '{OUTPUT_CLEANED_FILE}'...")
    with open(OUTPUT_CLEANED_FILE, "w", encoding="utf-8") as f_out:
        for head in original_order:
            if head in definitions:
                cleaned_def_text = " ".join(definitions[head])
                f_out.write(f"{head}: {cleaned_def_text}\n")

    print("\n--> Process finished successfully.")


if __name__ == '__main__':
    main()

################################################################################################################################################################################

[w007104@login2 ~]$ cat sbatch_pairs.slurm
#!/bin/bash
#SBATCH --job-name=od_pairs
#SBATCH --output=od_pairs_%A_%a.out
#SBATCH --error=od_pairs_%A_%a.err
#SBATCH --array=1-40                     # <--- Now you only need to change this line
#SBATCH --time=100:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=16

module purge
module load Python/3.10.8-GCCcore-12.2.0-bare

# The --total_jobs argument now automatically gets the size of the array.
# This is much safer and easier to maintain.
python3 -u run_od_analysis.py \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    --total_jobs ${SLURM_ARRAY_TASK_COUNT} \
    --mode pairs \
    --num_workers ${SLURM_CPUS_PER_TASK}
[w007104@login2 ~]$ cat run_od_analysis.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ontological Differentiation (OD) Large-Scale Analysis Script (V5)

This script performs two main OD analyses, designed for parallel execution on a
SLURM cluster and leveraging multiprocessing within each job.

This version includes:
- Performance optimization in the core differentiation engine.
- Captures and outputs the final termination level for SOD and WOD.

Modes:
1.  '--mode pairs': Calculates SOD, WOD, termination levels, and pairwise GOD 
    for all unique pairs in the vocabulary slice. This single run produces 
    all data needed for both 'all-vs-all' and 'one-vs-all' post-analysis. 
    It also triggers a one-off canary test.
2.  '--mode single_god': Calculates the GOD for each individual word.
"""

import os
import argparse
from collections import Counter
import itertools
from tqdm import tqdm
from multiprocessing import Pool

# ===============================================
#   CORE ONTOLOGICAL DIFFERENTIATION LOGIC
# ===============================================

def process_GOD(words_to_expand, sets_dict, max_level=100):
    all_elements_seen = set(words_to_expand)
    elements_to_open_now = set(words_to_expand)
    for level in range(1, max_level + 1):
        elements_for_next_level = set()
        for word in elements_to_open_now:
            elements_for_next_level.update(sets_dict.get(word, []))
        new_elements = elements_for_next_level - all_elements_seen
        if not new_elements:
            return level
        all_elements_seen.update(new_elements)
        elements_to_open_now = new_elements
    return max_level

def _run_differentiation_engine(seed_a, seed_b, sets_dict, omega_god, diff_type, verbose=False):
    E, U, R = {}, {}, {}
    log_steps = []
    E[(0, 1)], E[(0, 2)] = Counter([seed_a]), Counter([seed_b])
    
    # --- PERFORMANCE OPTIMIZATION: Initialize global counters before the loop ---
    global_E_side1 = E[(0, 1)].copy()
    global_E_side2 = E[(0, 2)].copy()

    for side in [1, 2]:
        U[(0, side)], R[(0, side)] = E[(0, side)].copy(), Counter()
    
    if verbose: log_steps.append(f"Level 0:\n  U_1: {dict(U[(0,1)])}\n  U_2: {dict(U[(0,2)])}")

    for level in range(1, omega_god + 2):
        for side in [1, 2]:
            E[(level, side)] = Counter()
            for elem, count in E.get((level - 1, side), {}).items():
                for e in sets_dict.get(elem, []):
                    E[(level, side)][e] += count
            U[(level, side)], R[(level, side)] = E[(level, side)].copy(), Counter()

        # --- PERFORMANCE OPTIMIZATION: Update global counters incrementally ---
        global_E_side1 += E.get((level, 1), Counter())
        global_E_side2 += E.get((level, 2), Counter())
        global_E_total = global_E_side1 + global_E_side2

        # Cancellation logic
        for m in range(level + 1):
            for side in [1, 2]:
                words_to_cancel = []
                for u_word in U.get((m, side), {}):
                    if diff_type == 'WOD':
                        if global_E_total[u_word] > 1: words_to_cancel.append(u_word)
                    elif diff_type == 'SOD':
                        check_set = global_E_side2 if side == 1 else global_E_side1
                        if check_set[u_word] > 0: words_to_cancel.append(u_word)
                for u_word in words_to_cancel:
                    count = U[(m, side)].pop(u_word)
                    R[(m, side)][u_word] += count
        
        if verbose:
            log_steps.append(f"Level {level}:")
            for m_log in range(level + 1):
                u1_str = str(dict(U.get((m_log,1),{})))
                r1_str = str(dict(R.get((m_log,1),{})))
                u2_str = str(dict(U.get((m_log,2),{})))
                r2_str = str(dict(R.get((m_log,2),{})))
                log_steps.append(f"  (m={m_log}) U_1: {u1_str}, R_1: {r1_str}")
                log_steps.append(f"  (m={m_log}) U_2: {u2_str}, R_2: {r2_str}")
            log_steps.append("-" * 20)

        # Termination condition check (successful)
        if any(not U.get((m, s), True) for m, s in U.keys()):
            score = sum(count * lvl for (lvl, s), r_set in R.items() for word, count in r_set.items())
            if verbose: return score, level, "\n".join(log_steps)
            return score, level
        
        # Termination condition check (GOD rule)
        if level > omega_god:
            if verbose: return -1, level, "\n".join(log_steps) + "\nTERMINATION: Invalid run (GOD rule)."
            return -1, level
            
    # Fallback termination (if loop completes, which should be rare)
    final_level = omega_god + 1
    return -1, final_level

def process_WOD(seed_a, seed_b, sets_dict, omega_god, verbose=False):
    return _run_differentiation_engine(seed_a, seed_b, sets_dict, omega_god, 'WOD', verbose)
def process_SOD(seed_a, seed_b, sets_dict, omega_god, verbose=False):
    return _run_differentiation_engine(seed_a, seed_b, sets_dict, omega_god, 'SOD', verbose)

# ===============================================
#   WORKER FUNCTIONS FOR MULTIPROCESSING
# ===============================================
worker_sets_dict = None
worker_word_map = None

def init_worker(sets_dict, word_map):
    global worker_sets_dict, worker_word_map
    worker_sets_dict = sets_dict
    worker_word_map = word_map

def worker_process_pair(pair):
    word_a, word_b = pair
    omega_god = process_GOD([word_a, word_b], worker_sets_dict)
    
    sod_score, sod_term_level = process_SOD(word_a, word_b, worker_sets_dict, omega_god)
    wod_score, wod_term_level = process_WOD(word_a, word_b, worker_sets_dict, omega_god)
    
    idx1 = worker_word_map[word_a]
    idx2 = worker_word_map[word_b]
    
    # Output format: idx1 idx2 sod_score sod_term_level wod_score wod_term_level omega_god
    return f"{idx1} {idx2} {sod_score} {sod_term_level} {wod_score} {wod_term_level} {omega_god}\n"

def worker_process_single_god(word):
    god_score = process_GOD([word], worker_sets_dict)
    idx = worker_word_map[word]
    return f"{idx} {god_score}\n"
    
# ===============================================
#   UTILITY AND I/O FUNCTIONS
# ===============================================

def read_definitions(file_path):
    definitions, ordered_words = {}, []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line: continue
            head, def_text = line.split(":", 1)
            head = head.strip().lower()
            tokens = def_text.strip().split()
            if head not in definitions:
                ordered_words.append(head)
            definitions[head] = tokens
    return definitions, ordered_words

def run_canary_test(sets_dict):
    """Runs a single, verbose test case and saves a detailed log."""
    print("Running canary test for 'money' vs 'business'...")
    word_a, word_b = "money", "business"
    log_filename = "canary_test_od_log.txt"

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"CANARY TEST: {word_a} vs {word_b}\n" + "="*40 + "\n\n")

        f.write("--- GOD Calculation ---\n")
        omega_god = process_GOD([word_a, word_b], sets_dict)
        f.write(f"Result: ω_GOD = {omega_god}\n\n")

        f.write("--- WOD Calculation (verbose) ---\n")
        wod_score, wod_term_level, wod_log = process_WOD(word_a, word_b, sets_dict, omega_god, verbose=True)
        f.write(wod_log)
        f.write(f"\nFinal WOD Score: {wod_score} (Terminated at level {wod_term_level})\n\n")

        f.write("--- SOD Calculation (verbose) ---\n")
        sod_score, sod_term_level, sod_log = process_SOD(word_a, word_b, sets_dict, omega_god, verbose=True)
        f.write(sod_log)
        f.write(f"\nFinal SOD Score: {sod_score} (Terminated at level {sod_term_level})\n")

    print(f"Canary test complete. Detailed log saved to '{log_filename}'.")

# ===============================================
#   MAIN EXECUTION BLOCK
# ===============================================

def main():
    parser = argparse.ArgumentParser(description="Run large-scale Ontological Differentiation analysis (V5).")
    parser.add_argument("--job_id", type=int, required=True, help="Current job index (1-based).")
    parser.add_argument("--total_jobs", type=int, required=True, help="Total number of parallel jobs.")
    parser.add_argument("--mode", type=str, required=True, choices=['pairs', 'single_god'], help="The analysis mode to run.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of CPU cores to use within this job.")
    args = parser.parse_args()

    # --- Data Loading ---
    input_file = "extracted_definitions.txt"
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found."); return

    print("Loading definitions...")
    definitions, ordered_words = read_definitions(input_file)
    word_to_index_map = {word: i + 1 for i, word in enumerate(ordered_words)}
    print(f"Loaded {len(definitions)} definitions.")
    
    start_idx, end_idx = 1830, 21112
    analysis_vocab = ordered_words[start_idx-1 : end_idx]
    print(f"Analysis will be on {len(analysis_vocab)} words (indices {start_idx} to {end_idx}).")

    # --- Canary Test Trigger ---
    # Run the canary test only once, on the first job of the 'pairs' analysis.
    if args.job_id == 1 and args.mode == 'pairs':
        run_canary_test(definitions)

    # --- Mode Dispatch ---
    if args.mode == 'pairs':
        output_file = f"pairs_results_job_{args.job_id}.txt"
        all_pairs = list(itertools.combinations(analysis_vocab, 2))
        chunk_size = (len(all_pairs) + args.total_jobs - 1) // args.total_jobs
        start = (args.job_id - 1) * chunk_size
        end = min(len(all_pairs), start + chunk_size)
        items_for_this_job = all_pairs[start:end]
        worker_func = worker_process_pair

    elif args.mode == 'single_god':
        output_file = f"single_god_results_job_{args.job_id}.txt"
        chunk_size = (len(analysis_vocab) + args.total_jobs - 1) // args.total_jobs
        start = (args.job_id - 1) * chunk_size
        end = min(len(analysis_vocab), start + chunk_size)
        items_for_this_job = analysis_vocab[start:end]
        worker_func = worker_process_single_god
    
    else: # Should not happen due to argparse choices
        print(f"Error: Unknown mode '{args.mode}'")
        return

    print(f"Job {args.job_id}: Running '{args.mode}' mode for {len(items_for_this_job)} items using {args.num_workers} workers...")
    
    # --- Multiprocessing Execution ---
    with Pool(processes=args.num_workers, initializer=init_worker, initargs=(definitions, word_to_index_map)) as pool, open(output_file, "w") as f:
        results = list(tqdm(pool.imap_unordered(worker_func, items_for_this_job), total=len(items_for_this_job), desc=f"Job {args.job_id} ({args.mode})"))
        f.writelines(results)
    
    print(f"Job {args.job_id}: Analysis complete. Results in {output_file}")

if __name__ == '__main__':
    main()

################################################################################################################################################################################

[w007104@login2 ~]$ cat aggregate_and_filter.sh
#!/bin/bash

# ==============================================================================
# Post-Processing Script for Ontological Differentiation Results
#
# This script does two things:
# 1. Merges the temporary, per-job output files into final result files.
# 2. Extracts the specific "one-vs-all" results for a target word from
#    the final merged 'pairs' file.
#
# Usage:
#   bash aggregate_and_filter.sh
# ==============================================================================

echo "Starting post-processing..."

# --- Configuration ---
# The target word for which to extract "one-vs-all" results.
TARGET_WORD="word"

# The file containing the full list of words to find the index.
# This must be the same definition file used for the main analysis.
DEFINITIONS_FILE="extracted_definitions.txt"

# --- Task 1: Merge Per-Job Results ---

echo "[Task 1/3] Merging 'pairs' results..."
# Check if temporary files exist before trying to merge
if ls pairs_results_job_*.txt 1> /dev/null 2>&1; then
    cat pairs_results_job_*.txt > pairs_results.txt
    rm pairs_results_job_*.txt
    echo "  -> Merged into pairs_results.txt and removed temporary files."
else
    echo "  -> No 'pairs_results_job_*.txt' files found to merge. Skipping."
fi

echo "[Task 2/3] Merging 'single_god' results..."
if ls single_god_results_job_*.txt 1> /dev/null 2>&1; then
    cat single_god_results_job_*.txt > single_god_results.txt
    rm single_god_results_job_*.txt
    echo "  -> Merged into single_god_results.txt and removed temporary files."
else
    echo "  -> No 'single_god_results_job_*.txt' files found to merge. Skipping."
fi


# --- Task 2: Extract "One-vs-All" Results for the Target Word ---

echo "[Task 3/3] Extracting results for target word: '${TARGET_WORD}'"

# First, find the 1-based index of the target word.
# We use `grep -n` to get the line number, which corresponds to the index.
# `grep -w` ensures we match the whole word.
# `cut -d: -f1` extracts just the line number.
WORD_INDEX=$(grep -n -w "^${TARGET_WORD}:" "${DEFINITIONS_FILE}" | cut -d: -f1)

# Check if the word was found.
if [ -z "${WORD_INDEX}" ]; then
    echo "  -> ERROR: Target word '${TARGET_WORD}' not found in '${DEFINITIONS_FILE}'."
    echo "  -> Aborting extraction."
    exit 1
fi

echo "  -> Found index for '${TARGET_WORD}': ${WORD_INDEX}"

# Now, use grep to find all lines in the final results file that contain this index.
# The pattern `(^| )${WORD_INDEX}( |$)` is a robust way to find the index as a whole word,
# preventing accidental matches (e.g., finding index '190' inside '1900').
# It looks for the index at the start of a line OR preceded by a space,
# AND at the end of a line OR followed by a space.
grep -E "(^| )${WORD_INDEX}( |$)" pairs_results.txt > "one_vs_all_${TARGET_WORD}.txt"

echo "  -> Extraction complete. Results saved to 'one_vs_all_${TARGET_WORD}.txt'."
echo ""
echo "Post-processing finished successfully."