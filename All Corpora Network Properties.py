# ==============================================================================
#                  NETWORK PROPERTY CORRELATION PIPELINE
#                  (Multi-Corpus, Automated - Script 3 of 3)
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
#  The script will loop through each corpus. For each one, it will
#  calculate (or load from cache) network properties and correlate them
#  with OD scores. All plots and reports will be saved into a new
#  subfolder called "network_correlation_analysis".
#
# ############################################################################

import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
import networkx as nx

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
OUTPUT_SUBFOLDER_NAME = "network_correlation_analysis" # All plots/data will go here
CHUNK_SIZE = 1_000_000
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plot_dpi = 150
HEADER_NAMES_PAIRS = [
    'idx1', 'idx2', 'sod_score', 'sod_term_level',
    'wod_score', 'wod_term_level', 'omega_god'
]

# ==============================================================================
#                 Network Properties Calculation & Caching
# ==============================================================================
def calculate_network_properties(def_filepath, cache_filepath):
    """
    Builds a NetworkX graph, calculates key metrics, and caches the results.
    """
    if os.path.exists(cache_filepath):
        print(f"-> Loading cached network properties from '{os.path.basename(cache_filepath)}'...")
        return pd.read_csv(cache_filepath).set_index('Word')

    print("-> Cache not found. Calculating network properties (this may take a while)...")
    
    # Step 1: Build the graph
    definitions = {}
    with open(def_filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or ":" not in line: continue
            head, def_text = line.split(":", 1)
            definitions[head.strip().lower()] = def_text.strip().split()
    G = nx.DiGraph()
    all_words = list(definitions.keys())
    G.add_nodes_from(all_words)
    for headword, tokens in tqdm(definitions.items(), desc="  Adding edges", leave=False):
        for token in tokens:
            if G.has_node(token): G.add_edge(headword, token)

    # Step 2: Calculate network metrics
    print("  -> Calculating metrics (Degree, PageRank, Centrality)...")
    betweenness_sample_size = min(2000, int(len(G) * 0.1)) # Sensible sample size for speed
    
    df_net = pd.DataFrame(index=all_words)
    df_net['In_Degree'] = pd.Series(dict(G.in_degree()))
    df_net['Out_Degree'] = pd.Series(dict(G.out_degree()))
    df_net['Clustering_Coeff'] = pd.Series(nx.clustering(G.to_undirected()))
    df_net['PageRank'] = pd.Series(nx.pagerank(G, alpha=0.85))
    print("    - Calculating closeness centrality (can be slow)...")
    df_net['Closeness_Centrality'] = pd.Series(nx.closeness_centrality(G))
    print(f"    - Calculating betweenness centrality (VERY slow, using k={betweenness_sample_size})...")
    df_net['Betweenness_Centrality'] = pd.Series(nx.betweenness_centrality(G, k=betweenness_sample_size, seed=42))
    
    # Step 3: Cache the results
    df_net.index.name = 'Word'
    df_net.to_csv(cache_filepath)
    print(f"-> Network properties calculated and saved to cache.")
    return df_net

# ==============================================================================
#               Correlation with P-Values (Helper Function)
# ==============================================================================
def correlation_matrix_with_pvalues(df, columns):
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
    """Main function to run the full analysis pipeline for a single corpus."""
    corpus_tag = corpus_name.lower().replace(" ", "_")
    analysis_output_dir = os.path.join(corpus_dir, OUTPUT_SUBFOLDER_NAME)
    os.makedirs(analysis_output_dir, exist_ok=True)
    print(f"Output will be saved to: {analysis_output_dir}")

    # --- Step 1: Locate Input Files with Fallbacks ---
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

    # --- Open the main report file ---
    report_filepath = os.path.join(analysis_output_dir, f"report_full_network_analysis_{corpus_tag}.txt")
    with open(report_filepath, "w", encoding="utf-8") as report_file:
        report_file.write(f"NETWORK CORRELATION ANALYSIS REPORT\nCorpus: {corpus_name}\n\n")

        # Step 2: Get Network Properties (from cache or by calculation)
        cache_filepath = os.path.join(analysis_output_dir, f"network_properties_{corpus_tag}.csv")
        df_network = calculate_network_properties(def_filepath, cache_filepath)

        # Step 3: Calculate Average OD Scores
        print(f"-> Processing large results file '{os.path.basename(results_filepath)}'...")
        word_to_index = {word: i + 1 for i, word in enumerate(df_network.index)}
        index_to_word = {i + 1: word for word, i in word_to_index.items()}
        sod_sums, pair_counts = defaultdict(float), defaultdict(int)
        
        reader = pd.read_csv(results_filepath, sep=' ', header=None, names=HEADER_NAMES_PAIRS, chunksize=CHUNK_SIZE, on_bad_lines='warn', comment='#')
        for chunk in tqdm(reader, desc="  Aggregating SOD", leave=False):
            valid_chunk = chunk[chunk['sod_score'] != -1]
            for row in valid_chunk.itertuples(index=False):
                word1, word2 = index_to_word.get(row.idx1), index_to_word.get(row.idx2)
                if word1: sod_sums[word1] += row.sod_score; pair_counts[word1] += 1
                if word2: sod_sums[word2] += row.sod_score; pair_counts[word2] += 1
        avg_sod = {word: sod_sums[word] / pair_counts[word] for word in sod_sums if pair_counts[word] > 0}
        
        # Step 4: Combine all metrics into a single DataFrame
        print("-> Combining network metrics and OD scores...")
        df_analysis = df_network.copy()
        df_analysis['Avg_SOD'] = pd.Series(avg_sod)
        df_analysis.dropna(subset=['Avg_SOD'], inplace=True)
        df_analysis['Log_Avg_SOD'] = np.log10(df_analysis['Avg_SOD'] + 1)
        report_file.write(f"Final analysis performed on {len(df_analysis)} words.\n\n")

        # Step 5: Correlation Analysis
        report_file.write("="*80 + "\n                NETWORK CORRELATION ANALYSIS RESULTS\n" + "="*80 + "\n")
        correlation_cols = ['Log_Avg_SOD', 'In_Degree', 'Out_Degree', 'Clustering_Coeff', 'PageRank', 'Closeness_Centrality', 'Betweenness_Centrality']
        valid_cols = [col for col in correlation_cols if col in df_analysis.columns]
        
        corr_matrix, pval_matrix = correlation_matrix_with_pvalues(df_analysis, valid_cols)
        
        report_file.write("\n--- Correlation Matrix (Pearson's r) ---\n")
        report_file.write(corr_matrix.to_string(float_format="%.3f") + "\n")
        report_file.write("\n--- P-values for Correlation ---\n")
        report_file.write(pval_matrix.to_string(float_format="%.3g") + "\n")
        
        # Save correlation matrix to CSV
        corr_csv_path = os.path.join(analysis_output_dir, f"report_correlation_matrix_{corpus_tag}.csv")
        corr_matrix.to_csv(corr_csv_path, float_format="%.4f")
        print(f"-> Correlation matrix saved to CSV.")

        # Plot Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='vlag', fmt=".2f", linewidths=.5, center=0)
        plt.title(f'Correlation of Network Properties and OD Score\n({corpus_name})', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_output_dir, f"plot_network_correlation_heatmap_{corpus_tag}.png"), dpi=plot_dpi)
        plt.show(block=False)

        # Step 6: Plot Gallery
        print("-> Generating gallery of network relationship plots...")
        gallery_dir = os.path.join(analysis_output_dir, "network_plot_gallery")
        os.makedirs(gallery_dir, exist_ok=True)
        core_var = 'Log_Avg_SOD'
        for prop in [col for col in valid_cols if col != core_var]:
            g = sns.jointplot(data=df_analysis, x=prop, y=core_var, kind='scatter', height=8, joint_kws={'alpha': 0.1, 's': 15})
            g.fig.suptitle(f'Relationship between {core_var} and {prop}\n({corpus_name})', y=1.02, fontsize=16)
            if any(p in prop for p in ['Degree', 'PageRank', 'Centrality']):
                g.ax_joint.set_xscale('log'); g.ax_marg_x.set_xscale('log') # Log scale for skewed metrics
            g.savefig(os.path.join(gallery_dir, f"plot_joint_{core_var}_vs_{prop}_{corpus_tag}.png"), dpi=plot_dpi)
            plt.close(g.fig)

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