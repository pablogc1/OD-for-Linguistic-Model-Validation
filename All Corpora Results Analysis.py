# ==============================================================================
#           STRUCTURAL BAND & TERMINATION LEVEL ANALYSIS PIPELINE
#                  (Multi-Corpus, Automated - Script 1 of 3)
# ==============================================================================
#
# ############################################################################
# #                       HOW TO RUN THIS SCRIPT                             #
# ############################################################################
#
#  This script is fully automated.
#
#  1. Ensure the list of corpus names in the "USER CONFIGURATION"
#     section below is correct.
#
#  2. Press the standard "Run file" button in Spyder (or F5).
#
#  The script will loop through each corpus, run the complete analysis,
#  and save all plots and data tables into a new, organized subfolder
#  called "structural_analysis_output" inside each main corpus directory.
#
# ############################################################################

import matplotlib
matplotlib.use('Qt5Agg') # Forcing an interactive backend if default isn't working

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import warnings

# --- Compatibility Block for different SciPy versions ---
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.ndimage import gaussian_filter1d
try:
    from scipy.signal import PeakPropertyWarning
except ImportError:
    class PeakPropertyWarning(Warning):
        pass
# ---------------------------------------------------------


# ==============================================================================
#                      ******** USER CONFIGURATION ********
# ==============================================================================
#
# List all the corpus folder names you want to analyze.
# The script will loop through this list automatically.
#
CORPORA_TO_ANALYZE = [
    "Ground Filtered",
    "Random Removal",
    "Targeted Removal",
    "Null Model",
    "AI Generated"
]
#
# ==============================================================================


# ==============================================================================
#                      Global Settings (No need to change)
# ==============================================================================
BASE_RESULTS_DIR = "/Volumes/PortableSSD/OD Results"
OUTPUT_SUBFOLDER_NAME = "structural_analysis_output" # All plots/data will go here

HEADER_NAMES_PAIRS = [
    'idx1', 'idx2', 'sod_score', 'sod_term_level',
    'wod_score', 'wod_term_level', 'omega_god'
]
PAIRS_DTYPES = {
    'idx1': 'int32', 'idx2': 'int32',
    'sod_score': 'int64', 'sod_term_level': 'int16',
    'wod_score': 'int64', 'wod_term_level': 'int16',
    'omega_god': 'int16'
}
HEADER_NAMES_SINGLE_GOD = ['idx', 'god_score']
SINGLE_GOD_DTYPES = {'idx': 'int32', 'god_score': 'int16'}

HISTOGRAM_BINS = 500
CHUNK_SIZE = 1_000_000
SCATTER_SAMPLE_SIZE = 100_000
plt.style.use('seaborn-v0_8-whitegrid')
plot_dpi = 200

# ==============================================================================
#                      Analysis Helper Functions
# ==============================================================================
def find_peaks_with_detrending(H, sigma=30):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PeakPropertyWarning)
        distance_from_peak, min_peak_width = 2, 1
        log_H = np.log1p(H); baseline_log = gaussian_filter1d(log_H, sigma=sigma)
        H_detrended_log = log_H - baseline_log
        peaks, _ = find_peaks(H_detrended_log, prominence=0.1, width=min_peak_width, distance=distance_from_peak)
    return peaks, H_detrended_log, baseline_log

def analyze_peak_structure(peaks, H, bin_centers, column_prefix, analysis_output_dir, corpus_name, corpus_tag):
    if len(peaks) < 2:
        print("   -> Not enough peaks (<2) for structural analysis.")
        return None
    peak_locations_log = bin_centers[peaks]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PeakPropertyWarning)
        prominences_val, _, _ = peak_prominences(H, peaks)
        widths_bins = peak_widths(H, peaks, rel_height=0.5)[0]
    separations = np.diff(peak_locations_log)
    peak_df = pd.DataFrame({
        'Peak Index': peaks, 'Log10 Score': peak_locations_log,
        'Separation to Next': np.append(separations, [np.nan]),
        'Prominence': prominences_val, 'Width (bins)': widths_bins
    })
    
    csv_path = os.path.join(analysis_output_dir, f"report_peak_analysis_{corpus_tag}_{column_prefix}.csv")
    txt_path = os.path.join(analysis_output_dir, f"report_peak_analysis_{corpus_tag}_{column_prefix}.txt")
    peak_df.to_csv(csv_path, index=False, float_format="%.4f")
    with open(txt_path, 'w') as f:
        f.write(f"--- Peak Structure Analysis ({corpus_name} - {column_prefix.upper()}) ---\n\n")
        f.write(peak_df.to_string(index=False, float_format="%.4f"))
    print(f"   -> Peak analysis reports saved.")
    return peak_df

def create_scatter_sample_safely(filepath, num_samples, header_names, dtypes):
    print(f"   -> Creating scatter sample (target size: {num_samples})...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
    except FileNotFoundError: return pd.DataFrame(columns=header_names)
    if total_lines == 0: return pd.DataFrame(columns=header_names)
    if total_lines <= num_samples:
        return pd.read_csv(filepath, sep=' ', header=None, names=header_names, dtype=dtypes, on_bad_lines='warn')
    p = num_samples / total_lines; sampled_chunks = []
    reader = pd.read_csv(filepath, sep=' ', header=None, names=header_names, dtype=dtypes, on_bad_lines='skip', chunksize=CHUNK_SIZE)
    for chunk in tqdm(reader, total=int(np.ceil(total_lines/CHUNK_SIZE)), desc="Sampling lines", leave=False):
        sampled_chunks.append(chunk.sample(frac=p, replace=False, random_state=42))
    if not sampled_chunks: return pd.DataFrame(columns=header_names)
    return pd.concat(sampled_chunks, ignore_index=True)

# ==============================================================================
#      Core Analysis & Plotting Functions
# ==============================================================================

def analyze_and_plot_all(filepath, analysis_name, column_prefix, analysis_output_dir, corpus_name, corpus_tag):
    """
    This is the main function that performs the entire structural analysis for one metric (e.g., SOD).
    """
    print("\n" + "="*80)
    print(f"Processing: '{analysis_name}' on column prefix '{column_prefix.upper()}'")
    print("="*80)
    
    score_col, term_level_col = f'{column_prefix}_score', f'{column_prefix}_term_level'

    # Step 1: Find data range for binning
    print(f"[1/7] Pre-scanning for data range ('{score_col}')...")
    min_val, max_val = np.inf, -np.inf
    reader = pd.read_csv(filepath, sep=' ', header=None, names=HEADER_NAMES_PAIRS, chunksize=CHUNK_SIZE, usecols=[score_col], dtype={score_col: PAIRS_DTYPES[score_col]}, on_bad_lines='skip')
    for chunk in tqdm(reader, desc="Finding range", leave=False):
        valid_chunk = chunk[chunk[score_col] > 0][score_col]
        if not valid_chunk.empty: min_val, max_val = min(min_val, valid_chunk.min()), max(max_val, valid_chunk.max())
    if np.isinf(min_val):
        print(f"---> SKIPPING prefix '{column_prefix}': No valid data found."); return
    bins = np.linspace(np.log10(min_val + 1), np.log10(max_val + 1), HISTOGRAM_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Step 2: Build 2D histogram
    print(f"[2/7] Building 2D histogram...")
    max_level_val = 0
    reader_for_level = pd.read_csv(filepath, sep=' ', header=None, names=HEADER_NAMES_PAIRS, chunksize=CHUNK_SIZE, usecols=[term_level_col], dtype={term_level_col: PAIRS_DTYPES[term_level_col]}, on_bad_lines='skip')
    for chunk in tqdm(reader_for_level, desc="Finding max level", leave=False):
        if not chunk.empty: max_level_val = max(max_level_val, chunk[term_level_col].max())
    level_score_hist = np.zeros((int(max_level_val) + 1, HISTOGRAM_BINS), dtype=np.int64)
    reader = pd.read_csv(filepath, sep=' ', header=None, names=HEADER_NAMES_PAIRS, chunksize=CHUNK_SIZE, usecols=[score_col, term_level_col], dtype=PAIRS_DTYPES, on_bad_lines='skip')
    for chunk in tqdm(reader, desc="Building 2D hist", leave=False):
        valid_chunk = chunk[chunk[score_col] > 0].copy()
        if not valid_chunk.empty:
            score_bins = np.clip(np.digitize(np.log10(valid_chunk[score_col] + 1), bins) - 1, 0, HISTOGRAM_BINS - 1)
            np.add.at(level_score_hist, (valid_chunk[term_level_col].astype(int), score_bins), 1)
    H = level_score_hist.sum(axis=0)

    # Step 3: Detect Peaks
    print("[3/7] Detecting peaks...")
    peaks, H_detrended_log, baseline_log = find_peaks_with_detrending(H)
    print(f"   -> Found {len(peaks)} peaks.")

    # Step 4: Analyze Peak Structure and save report
    print("[4/7] Analyzing peak structure...")
    peak_analysis_df = analyze_peak_structure(peaks, H, bin_centers, column_prefix, analysis_output_dir, corpus_name, corpus_tag)

    # Step 5: Create Scatter Plot Sample
    print("[5/7] Creating scatter plot sample...")
    df_plot_sample = create_scatter_sample_safely(filepath, SCATTER_SAMPLE_SIZE, HEADER_NAMES_PAIRS, PAIRS_DTYPES)
    if not df_plot_sample.empty:
        df_plot_sample = df_plot_sample[df_plot_sample[score_col] != -1]

    # Step 6: Generate All Plots
    print("[6/7] Generating all standard plots...")
    plot_all_graphs(H, bins, bin_centers, peaks, H_detrended_log, baseline_log, peak_analysis_df, df_plot_sample, analysis_name, column_prefix, analysis_output_dir, corpus_name, corpus_tag)
    
    # Step 7: Analyze and Plot Termination Levels
    print("[7/7] Performing in-depth termination level analysis...")
    plot_termination_levels(level_score_hist, bins, analysis_name, column_prefix, analysis_output_dir, corpus_name, corpus_tag)

def plot_all_graphs(H, bins, bin_centers, peaks, H_detrended_log, baseline_log, peak_analysis_df, df_plot_sample, analysis_name, column_prefix, analysis_output_dir, corpus_name, corpus_tag):
    """
    A dedicated function to generate and save all standard plots.
    """
    score_col, term_level_col = f'{column_prefix}_score', f'{column_prefix}_term_level'
    peak_locations_log = bin_centers[peaks] if len(peaks) > 0 else []

    # --- Prepare data for scatter plots ---
    plot_df = pd.DataFrame()
    if not df_plot_sample.empty:
        df1 = df_plot_sample[['idx1', score_col, term_level_col]].rename(columns={'idx1': 'index'})
        df2 = df_plot_sample[['idx2', score_col, term_level_col]].rename(columns={'idx2': 'index'})
        plot_df = pd.concat([df1, df2], ignore_index=True)

    # --- PLOT 1: Scatter Plots (Original and Color-coded) ---
    if not plot_df.empty:
        # Original Scatter
        plt.figure(figsize=(18, 10))
        plt.plot(plot_df['index'], plot_df[score_col], 'o', markersize=2, alpha=0.1, color='tab:blue')
        plt.yscale('log'); plt.ylabel('Score (Log Scale)'); plt.xlabel('Word Index')
        plt.title(f'Score vs. Index ({analysis_name} - {column_prefix.upper()})\n({corpus_name})', fontsize=16)
        plt.savefig(os.path.join(analysis_output_dir, f"plot_scatter_original_{corpus_tag}_{column_prefix}.png"), dpi=plot_dpi)
        plt.show(block=False)

        # RESTORED: Color-coded Scatter
        plt.figure(figsize=(18, 10))
        scatter = plt.scatter(plot_df['index'], plot_df[score_col], c=plot_df[term_level_col], s=5, alpha=0.3, cmap='viridis')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Termination Level', fontsize=14)
        plt.yscale('log'); plt.ylabel('Score (Log Scale)'); plt.xlabel('Word Index')
        plt.title(f'Score vs. Index (Color by Term. Level) ({analysis_name} - {column_prefix.upper()})\n({corpus_name})', fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout()
        plt.savefig(os.path.join(analysis_output_dir, f"plot_scatter_color_{corpus_tag}_{column_prefix}.png"), dpi=plot_dpi)
        plt.show(block=False)

    # --- PLOT 2: Diagnostic Plot ---
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(bin_centers, np.log1p(H), color='gray', alpha=0.5, label='Log(1+Count) [Original]')
    ax.plot(bin_centers, baseline_log, color='blue', linestyle='--', label='Calculated Baseline')
    ax.set_ylabel('Log(1+Count)', color='blue'); ax.tick_params(axis='y', labelcolor='blue')
    ax2 = ax.twinx()
    ax2.plot(bin_centers, H_detrended_log, color='green', label='Detrended Signal')
    if len(peaks) > 0: ax2.plot(bin_centers[peaks], H_detrended_log[peaks], 'x', color='red', markersize=10, mew=2, label='Detected Peaks')
    ax2.set_ylabel('Detrended Log Signal', color='green'); ax2.tick_params(axis='y', labelcolor='green')
    ax.set_xlabel('Log10(Score + 1)'); plt.title(f'Peak Detection Diagnostic ({analysis_name} - {column_prefix.upper()})\n({corpus_name})', fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95), bbox_transform=ax.transAxes); plt.grid(False); plt.tight_layout()
    plt.savefig(os.path.join(analysis_output_dir, f"plot_diagnostic_{corpus_tag}_{column_prefix}.png"), dpi=plot_dpi)
    plt.show(block=False)

    # --- PLOT 3: Density Histogram ---
    plt.figure(figsize=(18, 10))
    bar_width = np.diff(bins)[0] if len(bins) > 1 else 1
    plt.bar(bin_centers, H, width=bar_width, edgecolor='black', alpha=0.7, label='Full Data Histogram')
    if len(peaks) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PeakPropertyWarning)
            _, width_heights, left_ips, right_ips = peak_widths(H, peaks, rel_height=0.5)
        plt.plot(bin_centers[peaks], H[peaks], "x", color='red', markersize=10, mew=2, label=f'{len(peaks)} Detected Peaks')
        plt.hlines(y=width_heights, xmin=bin_centers[left_ips.astype(int)], xmax=bin_centers[right_ips.astype(int)], color="orange", linewidth=3, label='Band Widths')
    plt.yscale('log'); plt.xlabel('Log10(Score + 1)'); plt.title(f'Score Density ({analysis_name} - {column_prefix.upper()})\n({corpus_name})', fontsize=16)
    plt.legend(); plt.grid(True, which='both', linestyle='--'); plt.tight_layout()
    plt.savefig(os.path.join(analysis_output_dir, f"plot_density_{corpus_tag}_{column_prefix}.png"), dpi=plot_dpi)
    plt.show(block=False)
    
    # --- PLOT 4: Peak Separation Analysis ---
    if peak_analysis_df is not None and not peak_analysis_df.empty and 'Separation to Next' in peak_analysis_df.columns:
        valid_separations = peak_analysis_df['Separation to Next'].dropna()
        if not valid_separations.empty:
            plt.figure(figsize=(12, 7))
            peak_indices = valid_separations.index
            plt.bar(peak_indices, valid_separations, color='cornflowerblue', edgecolor='black', label='Separation to Next Peak')
            mean_sep = valid_separations.mean(); plt.axhline(y=mean_sep, color='r', linestyle='--', label=f'Mean Separation: {mean_sep:.4f}')
            plt.xlabel('Peak Index (P#)'); plt.ylabel('Separation in Log10(Score) Space')
            plt.title(f'Peak Separation Analysis ({analysis_name} - {column_prefix.upper()})\n({corpus_name})', fontsize=16)
            plt.xticks(peak_indices, [f'P{i}-P{i+1}' for i in peak_indices], rotation=45, ha='right')
            plt.legend(); plt.grid(axis='y', linestyle='--'); plt.tight_layout()
            plt.savefig(os.path.join(analysis_output_dir, f"plot_separation_{corpus_tag}_{column_prefix}.png"), dpi=plot_dpi)
            plt.show(block=False)

def plot_termination_levels(level_score_hist, bins, analysis_name, column_prefix, analysis_output_dir, corpus_name, corpus_tag):
    """
    A dedicated function to generate and save termination level plots and reports.
    """
    bin_centers = (bins[:-1] + bins[1:]) / 2
    total_counts_per_level = level_score_hist.sum(axis=1)
    valid_levels = np.where(total_counts_per_level > 0)[0]
    if len(valid_levels) == 0:
        print("   -> No valid termination level data found. Skipping plotting."); return

    df_level_freq = pd.DataFrame({'Termination Level': valid_levels, 'Count': total_counts_per_level[valid_levels].astype(np.int64)})
    df_level_freq['Percentage (%)'] = (df_level_freq['Count'] / df_level_freq['Count'].sum()) * 100
    df_level_freq = df_level_freq.sort_values(by='Count', ascending=False).reset_index(drop=True)

    # Save report to TXT file
    report_path = os.path.join(analysis_output_dir, f"report_termination_levels_{corpus_tag}_{column_prefix}.txt")
    with open(report_path, 'w') as f:
        f.write(f"--- Termination Level Frequencies (Overall) - {corpus_name} - {column_prefix.upper()} ---\n\n")
        f.write(df_level_freq.to_string(index=False, float_format="%.2f"))
    print(f"   -> Termination level report saved.")

    # Plot Overall Termination Level Histogram
    plt.figure(figsize=(14, 8))
    mode_level = df_level_freq.iloc[0]['Termination Level']
    plt.bar(df_level_freq['Termination Level'], df_level_freq['Count'], color='darkcyan', edgecolor='black', alpha=0.8)
    plt.axvline(x=mode_level, color='red', linestyle='--', linewidth=2, label=f'Most Common Level = {mode_level}')
    plt.xlabel('Termination Level', fontsize=14); plt.ylabel('Number of Pairs (Log Scale)', fontsize=14)
    plt.title(f'Overall Distribution of Termination Levels ({analysis_name} - {column_prefix.upper()})\n({corpus_name})', fontsize=16)
    plt.xticks(np.arange(0, df_level_freq['Termination Level'].max() + 2, 2)); plt.yscale('log')
    plt.legend(); plt.grid(True, which='both', linestyle=':'); plt.tight_layout()
    plt.savefig(os.path.join(analysis_output_dir, f"plot_term_level_dist_{corpus_tag}_{column_prefix}.png"), dpi=plot_dpi)
    plt.show(block=False)
    
    # Plot Stacked Histogram
    plt.figure(figsize=(20, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_levels)))
    level_to_color_map = dict(zip(valid_levels, colors))
    bottom_tracker = np.zeros(level_score_hist.shape[1])
    bar_width = np.diff(bins)[0] if len(bins) > 1 else 1
    for level in df_level_freq['Termination Level']:
        counts = level_score_hist[level, :]
        plt.bar(bin_centers, counts, bottom=bottom_tracker, width=bar_width, label=f'Level {level}', color=level_to_color_map.get(level))
        bottom_tracker += counts
    plt.yscale('log'); plt.xlabel('Log10(Score + 1)')
    plt.title(f'Score Density by Termination Level ({analysis_name} - {column_prefix.upper()})\n({corpus_name})', fontsize=16)
    handles, labels = plt.gca().get_legend_handles_labels()
    top_n_for_legend = 15
    plt.legend(handles[:top_n_for_legend], labels[:top_n_for_legend], title='Top Term. Levels', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(analysis_output_dir, f"plot_stacked_hist_{corpus_tag}_{column_prefix}.png"), dpi=plot_dpi)
    plt.show(block=False)

# ==============================================================================
#                      MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    # Loop through the list of corpora defined at the top of the script
    for corpus_name in CORPORA_TO_ANALYZE:
        corpus_tag = corpus_name.lower().replace(" ", "_")
        corpus_dir = os.path.join(BASE_RESULTS_DIR, corpus_name)
        analysis_output_dir = os.path.join(corpus_dir, OUTPUT_SUBFOLDER_NAME)

        print("\n" + "#"*80)
        print(f"###   STARTING ANALYSIS FOR CORPUS: {corpus_name.upper()}   ###")
        print("#"*80)

        os.makedirs(analysis_output_dir, exist_ok=True)
        print(f"Output will be saved to: {analysis_output_dir}")
        
        # --- Full "All-vs-All" Analysis ---
        all_vs_all_filename = f"pairs_results_{corpus_tag}.txt"
        all_vs_all_path = os.path.join(corpus_dir, all_vs_all_filename)
        if not os.path.exists(all_vs_all_path):
            all_vs_all_path = os.path.join(corpus_dir, "pairs_results.txt") # Fallback
            if not os.path.exists(all_vs_all_path):
                print(f"\n---> SKIPPING [All-vs-All]: Results file not found.")
                continue 
        
        for prefix in ['sod', 'wod']:
            analyze_and_plot_all(all_vs_all_path, 'All-vs-All', prefix, analysis_output_dir, corpus_name, corpus_tag)
        
        plt.close('all')

    print("\n\n################################################")
    print("###   ALL ANALYSES COMPLETED SUCCESSFULLY   ###")
    print("################################################")

if __name__ == '__main__':
    main()