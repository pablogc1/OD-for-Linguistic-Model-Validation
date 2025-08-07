# ==============================================================================
#                  LEXICAL PROPERTY CORRELATION PIPELINE
#                  (Multi-Corpus, Automated - Script 2 of 3)
# ==============================================================================
#
# ############################################################################
# #                       HOW TO RUN THIS SCRIPT                             #
# ############################################################################
#
#  This script is fully automated.
#
#  1. Ensure the list of corpus names and the base directory in the
#     "USER CONFIGURATION" section below are correct.
#
#  2. Press the standard "Run file" button in Spyder (or F5).
#
#  The script will loop through each corpus, perform the complete
#  lexical correlation analysis, and save all plots and data tables into
#  a new subfolder called "lexical_correlation_analysis" inside each
#  main corpus directory.
#
# ############################################################################

import pandas as pd
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, f_oneway

# ==============================================================================
#                      ******** USER CONFIGURATION ********
# ==============================================================================
#
# List all the corpus folder names you want to analyze.
CORPORA_TO_ANALYZE = [
    "Ground Filtered",
    "Random Removal",
    "Targeted Removal",
    "Null Model",
    "AI Generated"
]

# Set the base directory where all your main corpus folders are located.
BASE_RESULTS_DIR = "/Volumes/PortableSSD/OD Results"
#
# ==============================================================================


# ==============================================================================
#                      Global Settings (No need to change)
# ==============================================================================
OUTPUT_SUBFOLDER_NAME = "lexical_correlation_analysis" # All plots/data will go here
CHUNK_SIZE = 1_000_000
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plot_dpi = 150
HEADER_NAMES_PAIRS = [
    'idx1', 'idx2', 'sod_score', 'sod_term_level',
    'wod_score', 'wod_term_level', 'omega_god'
]

# ==============================================================================
#                 Extended Lexical Property Extractor
# ==============================================================================
def get_lexical_properties_extended(def_filepath):
    """Reads a definitions file to calculate a rich set of lexical properties."""
    print(f"-> Analyzing lexical properties from '{os.path.basename(def_filepath)}'...")
    index_to_word, word_to_index, definitions = {}, {}, {}
    with open(def_filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or ":" not in line: continue
            head, def_text = line.split(":", 1)
            head = head.strip().lower()
            tokens = def_text.strip().split()
            idx = i + 1
            index_to_word[idx], word_to_index[head], definitions[idx] = head, idx, tokens
            
    def_lengths = {idx: len(tokens) for idx, tokens in definitions.items()}
    word_frequencies = Counter(token for tokens_list in definitions.values() for token in tokens_list)
    
    extended_properties = {}
    for idx, def_tokens in tqdm(definitions.items(), desc="  Calculating metrics", leave=False):
        token_freqs = [word_frequencies.get(token, 0) for token in def_tokens]
        token_lens = [def_lengths.get(word_to_index.get(token), 0) for token in def_tokens]
        
        extended_properties[idx] = {
            'Word_Index': idx, # Keep index for merging
            'Word': index_to_word.get(idx),
            'Def_Length': def_lengths.get(idx, 0),
            'Frequency': word_frequencies.get(index_to_word.get(idx, ""), 0),
            'Avg_Def_Token_Freq': np.mean(token_freqs) if token_freqs else 0,
            'Max_Def_Token_Freq': np.max(token_freqs) if token_freqs else 0,
            'Avg_Def_Token_Length': np.mean(token_lens) if token_lens else 0
        }
    print(f"-> Successfully processed properties for {len(definitions)} words.")
    return extended_properties

# ==============================================================================
#               Function for Correlation with P-Values
# ==============================================================================
def correlation_matrix_with_pvalues(df, columns):
    """Calculates a correlation matrix and a corresponding p-value matrix."""
    df_valid = df[columns].dropna()
    n_cols = len(columns)
    corr_matrix = pd.DataFrame(np.ones((n_cols, n_cols)), index=columns, columns=columns)
    pval_matrix = pd.DataFrame(np.ones((n_cols, n_cols)), index=columns, columns=columns)

    for i in range(n_cols):
        for j in range(i, n_cols):
            if i == j: continue
            corr, pval = pearsonr(df_valid.iloc[:, i], df_valid.iloc[:, j])
            corr_matrix.iloc[i, j] = corr_matrix.iloc[j, i] = corr
            pval_matrix.iloc[i, j] = pval_matrix.iloc[j, i] = pval
    return corr_matrix, pval_matrix

# ==============================================================================
#                                Main Analysis
# ==============================================================================
def run_analysis_for_corpus(corpus_name, corpus_dir):
    """
    Main function to perform a multi-faceted analysis for a single corpus.
    """
    corpus_tag = corpus_name.lower().replace(" ", "_")
    analysis_output_dir = os.path.join(corpus_dir, OUTPUT_SUBFOLDER_NAME)
    os.makedirs(analysis_output_dir, exist_ok=True)
    print(f"Output will be saved to: {analysis_output_dir}")

    # --- Step 1: Locate Input Files ---
    def_filename = f"extracted_definitions_{corpus_tag}.txt"
    def_filepath = os.path.join(corpus_dir, def_filename)
    if not os.path.exists(def_filepath):
        def_filepath = os.path.join(corpus_dir, "extracted_definitions_cleaned.txt")
        if not os.path.exists(def_filepath):
            print(f"ERROR: No definitions file found in '{corpus_dir}'. Skipping corpus.")
            return

    results_filename = f"pairs_results_{corpus_tag}.txt"
    results_filepath = os.path.join(corpus_dir, results_filename)
    if not os.path.exists(results_filepath):
        results_filepath = os.path.join(corpus_dir, "pairs_results.txt")
        if not os.path.exists(results_filepath):
            print(f"ERROR: No results file found in '{corpus_dir}'. Skipping corpus.")
            return

    # --- Open the main report file to capture all text output ---
    report_filepath = os.path.join(analysis_output_dir, f"report_full_lexical_analysis_{corpus_tag}.txt")
    with open(report_filepath, "w", encoding="utf-8") as report_file:
        report_file.write(f"LEXICAL CORRELATION ANALYSIS REPORT\nCorpus: {corpus_name}\n\n")

        # --- Steps 2 & 3: Data Loading, Processing, and DataFrame Assembly ---
        extended_lexical_props = get_lexical_properties_extended(def_filepath)
        if extended_lexical_props is None: return

        sod_sums, wod_sums, pair_counts = defaultdict(int), defaultdict(int), defaultdict(int)
        print(f"-> Processing large results file '{os.path.basename(results_filepath)}'...")
        reader = pd.read_csv(results_filepath, sep=' ', header=None, names=HEADER_NAMES_PAIRS, chunksize=CHUNK_SIZE, on_bad_lines='warn', comment='#')
        for chunk in tqdm(reader, desc="  Analyzing chunks", leave=False):
            valid_chunk = chunk[chunk['sod_score'] != -1].copy()
            for row in valid_chunk.itertuples(index=False):
                sod_sums[row.idx1] += row.sod_score; sod_sums[row.idx2] += row.sod_score
                wod_sums[row.idx1] += row.wod_score; wod_sums[row.idx2] += row.wod_score
                pair_counts[row.idx1] += 1; pair_counts[row.idx2] += 1

        print("-> Combining all metrics into a final DataFrame...")
        analysis_data = []
        for idx, count in pair_counts.items():
            if count > 0:
                word_props = extended_lexical_props.get(idx, {}).copy()
                word_props.update({'Avg_SOD': sod_sums[idx] / count, 'Avg_WOD': wod_sums[idx] / count})
                analysis_data.append(word_props)
        
        df_analysis = pd.DataFrame(analysis_data).set_index('Word')
        df_analysis['Log_Avg_SOD'] = np.log10(df_analysis['Avg_SOD'] + 1)
        df_analysis['Log_Avg_WOD'] = np.log10(df_analysis['Avg_WOD'] + 1)
        report_file.write(f"Analysis was performed on {len(df_analysis)} words.\n\n")

        # --- Save the full analysis dataframe to a CSV file ---
        lexical_props_csv_path = os.path.join(analysis_output_dir, f"lexical_properties_with_avg_od_{corpus_tag}.csv")
        df_analysis.to_csv(lexical_props_csv_path, float_format="%.4f")
        print(f"-> Full lexical properties data frame saved to CSV.")

        # --- Step 4: Correlation Analysis ---
        report_file.write("="*80 + "\n                CORRELATION ANALYSIS RESULTS (EXTENDED)\n" + "="*80 + "\n")
        correlation_cols = ['Log_Avg_SOD', 'Log_Avg_WOD', 'Def_Length', 'Frequency', 'Avg_Def_Token_Freq', 'Max_Def_Token_Freq', 'Avg_Def_Token_Length']
        valid_cols = [col for col in correlation_cols if col in df_analysis.columns]
        
        corr_matrix, pval_matrix = correlation_matrix_with_pvalues(df_analysis, valid_cols)
        
        report_file.write("\n--- Extended Correlation Matrix (Pearson's r) ---\n")
        report_file.write(corr_matrix.to_string(float_format="%.3f") + "\n")
        report_file.write("\n--- P-values for Correlation ---\n")
        report_file.write(pval_matrix.to_string(float_format="%.3g") + "\n")
        
        corr_csv_path = os.path.join(analysis_output_dir, f"report_correlation_matrix_{corpus_tag}.csv")
        corr_matrix.to_csv(corr_csv_path, float_format="%.4f")
        print(f"-> Correlation matrix saved to CSV.")

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f'Correlation Matrix of Lexical and OD Properties\n({corpus_name})', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_output_dir, f"plot_extended_correlation_heatmap_{corpus_tag}.png"), dpi=plot_dpi)
        plt.show(block=False)

        # --- Step 4B: Generate a Gallery of Individual Scatter Plots ---
        print("-> Generating a gallery of key relationship plots...")
        gallery_dir = os.path.join(analysis_output_dir, "plot_gallery")
        os.makedirs(gallery_dir, exist_ok=True)
        core_var = 'Log_Avg_SOD'
        
        ##################################################################################
        #                   *** THIS IS THE CORRECTED SECTION ***                          #
        ##################################################################################
        # The list comprehension now correctly checks 'col' instead of 'prop'
        plot_props = [col for col in valid_cols if col != core_var and 'Word_Index' not in col]
        for prop in plot_props:
        ##################################################################################
        
            g = sns.jointplot(data=df_analysis, x=prop, y=core_var, kind='scatter', height=8, joint_kws={'alpha': 0.1, 's': 15})
            g.fig.suptitle(f'Relationship between {core_var} and {prop}\n({corpus_name})', y=1.02, fontsize=16)
            g.savefig(os.path.join(gallery_dir, f"plot_joint_{core_var}_vs_{prop}_{corpus_tag}.png"), dpi=plot_dpi)
            plt.close(g.fig)

        # --- Step 5: Binned Categorical Analysis ---
        report_file.write("\n" + "="*80 + "\n                   BINNED CATEGORICAL ANALYSIS\n" + "="*80 + "\n")
        df_analysis['Freq_Bin'] = pd.qcut(df_analysis['Frequency'].rank(method='first'), q=4, labels=['Lowest Freq', 'Low Freq', 'High Freq', 'Highest Freq'])
        df_analysis['Len_Bin'] = pd.qcut(df_analysis['Def_Length'].rank(method='first'), q=4, labels=['Shortest Defs', 'Short Defs', 'Long Defs', 'Longest Defs'])
        
        plt.figure(figsize=(14, 9))
        sns.boxplot(data=df_analysis, x='Freq_Bin', y='Log_Avg_SOD', palette='viridis')
        plt.title(f'Avg. SOD Score by Word Frequency Category\n({corpus_name})', fontsize=18)
        plt.savefig(os.path.join(analysis_output_dir, f"plot_boxplot_by_frequency_{corpus_tag}.png"), dpi=plot_dpi)
        plt.show(block=False)
        
        plt.figure(figsize=(14, 9))
        sns.violinplot(data=df_analysis, x='Len_Bin', y='Log_Avg_SOD', palette='plasma', inner='quartile')
        plt.title(f'Avg. SOD Score by Definition Length Category\n({corpus_name})', fontsize=18)
        plt.savefig(os.path.join(analysis_output_dir, f"plot_violinplot_by_length_{corpus_tag}.png"), dpi=plot_dpi)
        plt.show(block=False)

        # --- Step 6: Statistical Significance of Binned Groups (ANOVA) ---
        report_file.write("\n" + "="*80 + "\n        STATISTICAL SIGNIFICANCE OF BINNED GROUPS (ANOVA)\n" + "="*80 + "\n")
        
        freq_groups = [df_analysis['Log_Avg_SOD'][df_analysis['Freq_Bin'] == cat] for cat in df_analysis['Freq_Bin'].cat.categories]
        f_stat_freq, p_val_freq = f_oneway(*freq_groups)
        report_file.write("\n--- ANOVA Test: Log_Avg_SOD across Frequency Bins ---\n")
        report_file.write(f"F-statistic: {f_stat_freq:.4f}\nP-value: {p_val_freq:.4g}\n")
        
        len_groups = [df_analysis['Log_Avg_SOD'][df_analysis['Len_Bin'] == cat] for cat in df_analysis['Len_Bin'].cat.categories]
        f_stat_len, p_val_len = f_oneway(*len_groups)
        report_file.write("\n--- ANOVA Test: Log_Avg_SOD across Definition Length Bins ---\n")
        report_file.write(f"F-statistic: {f_stat_len:.4f}\nP-value: {p_val_len:.4g}\n")

        report_file.write("\n\n" + "="*80 + "\n                           INTERPRETATION NOTE\n" + "="*80 + "\n")
        report_file.write("A statistically significant p-value (e.g., < 0.05) does not imply a large or meaningful effect size.\n")

# ==============================================================================
#                      MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """Main function to loop through corpora and execute the analysis."""
    for corpus_name in CORPORA_TO_ANALYZE:
        corpus_dir = os.path.join(BASE_RESULTS_DIR, corpus_name)
        
        print("\n" + "#"*80)
        print(f"###   STARTING ANALYSIS FOR CORPUS: {corpus_name.upper()}   ###")
        print("#"*80)
        
        if not os.path.isdir(corpus_dir):
            print(f"ERROR: Corpus directory not found at '{corpus_dir}'. Skipping.")
            continue
            
        run_analysis_for_corpus(corpus_name, corpus_dir)
        
        plt.close('all')

    print("\n\n################################################")
    print("###   ALL ANALYSES COMPLETED SUCCESSFULLY   ###")
    print("################################################")

if __name__ == '__main__':
    main()