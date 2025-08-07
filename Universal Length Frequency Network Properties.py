# ==============================================================================
#                      UNIVERSALITY ANALYSIS PIPELINE (V4)
#
# This script tests whether observed relationships between semantic, lexical,
# and network properties are "universal" or "corpus-dependent".
#
# V4 Fixes:
#  - THE DEFINITIVE FIX for the statsmodels parsing error.
#  - Creates a temporary, clean DataFrame for the model with simple column
#    names ('Response', 'Predictor') to guarantee the formula is parsed correctly.
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import statsmodels.formula.api as smf
from tqdm import tqdm
from matplotlib.lines import Line2D
# ==============================================================================
#                      ******** USER CONFIGURATION ********
# ==============================================================================
CORPORA_TO_ANALYZE = [
    "Ground Filtered",
    "Random Removal",
    "Targeted Removal",
    "Null Model",
    "AI Generated"
]
BASE_RESULTS_DIR = "/Volumes/PortableSSD/OD Results"
UNIVERSALITY_OUTPUT_DIR = "_UNIVERSALITY_ANALYSIS_"
PROPERTIES_TO_TEST = {
    'Def_Length': 'Definition Length',
    'Frequency': 'Word Frequency in Definitions',
    'PageRank': 'PageRank Centrality',
    'In_Degree': 'In-Degree',
    'Out_Degree': 'Out-Degree',
    'Betweenness_Centrality': 'Betweenness Centrality'
}

# <<< MODIFICATION 1: FINAL COLOR ASSIGNMENTS >>>
# Colors are explicitly defined with the requested changes.
# - Targeted Removal gets the previous "bright green".
# - Null Model gets a new, "very bright" lime green.
# - AI Generated gets a new, "very bright" red.
COLOR_CONFIG = {
    "Ground Filtered":  '#440154',  # Viridis Dark Purple
    "Random Removal":   '#31688e',  # Viridis Blue-Teal
    "Targeted Removal": '#28a745',  # Bright Green
    "Null Model":       '#FF0000',  # Very Bright Lime Green
    "AI Generated":     '#7CFC00'   # Very Bright Red
}
# ==============================================================================
#                         ANALYSIS HELPER FUNCTIONS
# ==============================================================================

def find_file_with_fallback(directory, preferred_name, fallback_name):
    preferred_path = os.path.join(directory, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path
    fallback_path = os.path.join(directory, fallback_name)
    return fallback_path if os.path.exists(fallback_path) else None

def load_and_combine_data():
    """Loads and merges the detailed data from all corpora into a single DataFrame."""
    all_dfs = []
    print("--- Phase 1: Aggregating Data From All Corpora ---")

    for corpus_name in tqdm(CORPORA_TO_ANALYZE, desc="Loading and combining corpora"):
        lexical_dir = os.path.join(BASE_RESULTS_DIR, corpus_name, "lexical_correlation_analysis")
        network_dir = os.path.join(BASE_RESULTS_DIR, corpus_name, "network_correlation_analysis")

        lex_path = find_file_with_fallback(lexical_dir, f"lexical_properties_with_avg_od_{corpus_name.lower().replace(' ', '_')}.csv", "lexical_properties_with_avg_od.csv")
        net_path = find_file_with_fallback(network_dir, f"network_properties_{corpus_name.lower().replace(' ', '_')}.csv", "network_properties.csv")

        if not lex_path or not net_path:
            print(f"\nWarning: Missing data files for '{corpus_name}'. Skipping.")
            continue

        df_lex = pd.read_csv(lex_path)
        df_net = pd.read_csv(net_path)

        df_merged = pd.merge(df_lex, df_net, on='Word', how='inner')
        df_merged['Corpus'] = corpus_name
        all_dfs.append(df_merged)

    if not all_dfs:
        print("\nFATAL ERROR: No data could be loaded. Aborting.")
        return None

    master_df = pd.concat(all_dfs, ignore_index=True)

    master_df['Corpus_Formula'] = master_df['Corpus'].str.replace(' ', '_')
    category_order = [name.replace(' ', '_') for name in CORPORA_TO_ANALYZE]
    master_df['Corpus_Formula'] = pd.Categorical(master_df['Corpus_Formula'], categories=category_order, ordered=False)

    print(f"\nSuccessfully created master DataFrame with {len(master_df):,} total rows.")
    return master_df

def perform_universality_test(df, x_var, y_var, x_label, output_dir):
    """Performs the visualization and statistical test for one relationship."""
    safe_x_var = x_var.replace('[', '').replace(']', '').replace('<', '')

    test_cols = [x_var, y_var, 'Corpus', 'Corpus_Formula']
    df_test = df[test_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_test) < 20:
        print(f"  -> Skipping {y_var} vs {x_var}: Not enough valid data points ({len(df_test)}) after cleaning.")
        return

    # <<< MODIFICATION 2: CONDITIONAL FILENAME GENERATION >>>
    # Check if the "AI Generated" corpus is present in the data for this specific plot.
    # If so, append '1' to the filename.
    plot_title = f'Relationship between {y_var} and {x_label}'
    base_filename = f"plot_universality_{y_var}_vs_{safe_x_var}"
    
    if 'AI Generated' in df_test['Corpus'].unique():
        plot_filename = os.path.join(output_dir, f"{base_filename}1.png")
    else:
        plot_filename = os.path.join(output_dir, f"{base_filename}.png")
    # <<< END OF FILENAME MODIFICATION >>>

    # --- Visualization with Dynamic Palette ---
    corpora_in_plot = sorted(df_test['Corpus'].unique(), key=lambda c: CORPORA_TO_ANALYZE.index(c))
    current_palette = {corpus: COLOR_CONFIG[corpus] for corpus in corpora_in_plot}

    g = sns.lmplot(data=df_test, x=x_var, y=y_var, hue='Corpus',
                   hue_order=corpora_in_plot,
                   palette=current_palette,
                   height=8, aspect=1.5,
                   scatter_kws={'alpha': 0.05, 's': 5},
                   line_kws={'lw': 3},
                   legend=False)

    g.fig.suptitle(plot_title, y=1.02, fontsize=20)
    g.set_axis_labels(x_label, f"Log10(Average {y_var.split('_')[1]} + 1)", fontsize=16)

    if 'Frequency' in x_var or 'Degree' in x_var or 'Centrality' in x_var or 'PageRank' in x_var:
        g.ax.set_xscale('log')

    legend_handles = [Line2D([0], [0], color=current_palette[name], lw=3) for name in corpora_in_plot]
    plt.legend(handles=legend_handles, labels=corpora_in_plot, title='Corpus',
               loc='center left', bbox_to_anchor=(1.02, 0.6), frameon=True)

    plt.tight_layout()
    g.savefig(plot_filename, dpi=150)
    plt.show(block=False)

    # --- Statistical Test (ANCOVA using OLS) ---
    report_filename = os.path.join(output_dir, f"stats_report_{y_var}_vs_{safe_x_var}.txt")
    
    df_model = df_test.copy()
    df_model.rename(columns={y_var: 'Response', x_var: 'Predictor'}, inplace=True)

    formula_safe = "Response ~ Predictor * Corpus_Formula"
    
    try:
        model = smf.ols(formula_safe, data=df_model).fit()
        summary = model.summary()

        with open(report_filename, 'w') as f:
            f.write(f"UNIVERSALITY ANALYSIS REPORT\n\n")
            f.write(f"Dependent Variable (Response): {y_var}\n")
            f.write(f"Predictor Variable (Predictor): {x_var} ({x_label})\n")
            f.write(f"Model: {formula_safe}\n")
            f.write(f"Reference Corpus: {df_test['Corpus_Formula'].cat.categories[0]}\n")
            f.write("="*80 + "\n\n")
            f.write(str(summary))
            f.write("\n\n" + "="*80 + "\n")
            f.write("HOW TO INTERPRET THIS REPORT:\n\n")
            f.write("The key is to look at the P>|t| values for the INTERACTION terms.\n")
            f.write("These look like `Predictor:Corpus_Formula[T.Corpus_Name]`.\n\n")
            f.write(" - If the P-value for an interaction term is LOW (e.g., < 0.05),\n")
            f.write("   it means the relationship's SLOPE is SIGNIFICANTLY DIFFERENT for that corpus\n")
            f.write("   compared to the reference ('Ground_Filtered').\n\n")
            f.write(" - If many interaction terms are significant, the relationship is NOT UNIVERSAL.\n")
            f.write("   It is CORPUS-DEPENDENT.\n")
            
    except Exception as e:
        print(f"  -> Could not perform statistical test for {y_var} vs {x_var}: {e}")
        with open(report_filename, 'w') as f:
            f.write(f"Statistical test failed for {y_var} vs {x_var}.\nError: {e}\nFormula attempted: {formula_safe}\n")

# ==============================================================================
#                               MAIN EXECUTION
# ==============================================================================

def main():
    """Main function to run the universality analysis pipeline."""
    
    os.makedirs(UNIVERSALITY_OUTPUT_DIR, exist_ok=True)
    print(f"Starting universality analysis. Outputs will be in '{UNIVERSALITY_OUTPUT_DIR}'\n")

    master_df = load_and_combine_data()
    
    if master_df is None:
        return

    print("\n--- Phase 2: Running Universality Tests for All Properties ---")
    
    sod_output_dir = os.path.join(UNIVERSALITY_OUTPUT_DIR, "SOD_Analysis")
    wod_output_dir = os.path.join(UNIVERSALITY_OUTPUT_DIR, "WOD_Analysis")
    os.makedirs(sod_output_dir, exist_ok=True)
    os.makedirs(wod_output_dir, exist_ok=True)
    
    for x_var, x_label in tqdm(PROPERTIES_TO_TEST.items(), desc="Testing relationships"):
        if x_var not in master_df.columns:
            print(f"\nSkipping '{x_var}': column not found in master dataframe.")
            continue
        
        tqdm.write(f" -> Testing SOD vs. {x_label}...")
        perform_universality_test(master_df, x_var, 'Log_Avg_SOD', x_label, sod_output_dir)

        tqdm.write(f" -> Testing WOD vs. {x_label}...")
        perform_universality_test(master_df, x_var, 'Log_Avg_WOD', x_label, wod_output_dir)

    plt.close('all')
    print("\n#################################################")
    print("###   UNIVERSALITY ANALYSIS SCRIPT FINISHED   ###")
    print("#################################################")

if __name__ == '__main__':
    main()