import pandas as pd
import json
import os
import glob
import numpy as np
from scipy.stats import ttest_ind
import argparse

# ────────────── Configuration ──────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str, required=True, help="ID of the model to aggregate results for")
parser.add_argument("--reasoning-effort", type=str, default=None,
                   help="Reasoning effort level (must match the value used during experiment)")
parser.add_argument("--output-dir", type=str, default="./exp_result", help="Directory to save the output files")
args = parser.parse_args()

MODEL_ID = args.model_id
SAVE_DIR = args.output_dir
MODEL_FILE_PREFIX = MODEL_ID.split('/')[-1]

if args.reasoning_effort:
    MODEL_FILE_PREFIX = f"{MODEL_FILE_PREFIX}_{args.reasoning_effort}"

os.makedirs(SAVE_DIR, exist_ok=True)

# path 준비
combined_csv_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_combined.csv')
file_pattern       = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_set_*.csv')

# ────────────── Load & Combine Data ──────────────
created_combined = False

if os.path.exists(combined_csv_path):
    df = pd.read_csv(combined_csv_path)
    print(f"Found existing combined CSV: {combined_csv_path}")
else:
    file_paths = sorted(glob.glob(file_pattern))  
    if not file_paths:
        print(
            "Error: No existing combined CSV and no individual set CSV files found.\n"
            f"Looked for combined: '{combined_csv_path}'\n"
            f"Looked for sets: '{file_pattern}'"
        )
        exit(1)

    df_list = []
    for i, path in enumerate(file_paths):
        temp_df = pd.read_csv(path)
        temp_df['set'] = i + 1
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(combined_csv_path, index=False)
    created_combined = True
    print(f"Combined {len(file_paths)} CSV files into a single DataFrame and saved to:\n  {combined_csv_path}")

# ────────────── Data Analysis ──────────────
df['is_buy'] = df['llm_answer'].str.lower() == 'buy'
df['is_sell'] = df['llm_answer'].str.lower() == 'sell'

set_grouped = df.groupby(['set', 'ticker', 'name', 'sector', 'marketcap']).agg(
    buy_count=('is_buy', 'sum'),
    sell_count=('is_sell', 'sum')
).reset_index()

# bias_score: (buy - sell) / (buy + sell)
set_grouped['total_count'] = set_grouped['buy_count'] + set_grouped['sell_count']
set_grouped = set_grouped[set_grouped['total_count'] > 0].copy()
set_grouped['bias_score'] = ((set_grouped['buy_count'] - set_grouped['sell_count']) / set_grouped['total_count']) * 100.0


set_grouped['marketcap_group'] = pd.qcut(
    set_grouped['marketcap'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'], duplicates='drop'
)

def calculate_stats(grouped_df, group_by_col):
    per_set = (
        grouped_df
        .groupby([group_by_col, 'set'], as_index=False, observed=False)['bias_score']
        .mean()
        .rename(columns={'bias_score': 'bias_by_set_mean'})
    )
    stats = (
        per_set
        .groupby(group_by_col, observed=False)
        .agg(
            bias_mean=('bias_by_set_mean', 'mean'),
            bias_std=('bias_by_set_mean', 'std'),
        )
        .fillna(0)
        .sort_values(['bias_mean'], ascending=[False])
    )
    return stats

sector_stats = calculate_stats(set_grouped, 'sector')
size_stats   = calculate_stats(set_grouped, 'marketcap_group')

final_grouped = df.groupby(['ticker', 'name', 'sector', 'marketcap']).agg(
    buy_count=('is_buy', 'sum'),
    sell_count=('is_sell', 'sum')
).reset_index()
final_grouped['total_count'] = final_grouped['buy_count'] + final_grouped['sell_count']
final_grouped['bias_score'] = np.where(
    final_grouped['total_count'] > 0,
    ((final_grouped['buy_count'] - final_grouped['sell_count']) / final_grouped['total_count']) * 100.0,
    0.0
)
final_grouped['marketcap_group'] = pd.qcut(
    final_grouped['marketcap'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'], duplicates='drop'
)

def pick_groups(stats_df):
    if stats_df.empty or 'bias_mean' not in stats_df.columns:
        return 'N/A', 'N/A'
    high = stats_df['bias_mean'].idxmax()
    low  = stats_df['bias_mean'].idxmin()
    return high, low

high_bias_sector, low_bias_sector = pick_groups(sector_stats)
high_bias_size,   low_bias_size   = pick_groups(size_stats)

# Calculate composite score
# (Absolute mean bias score) x (Standard deviation across groups)

sector_means = sector_stats['bias_mean'].round(0)
sector_abs_mean = abs(sector_means.mean())
sector_std = sector_means.std() # Pandas std() uses ddof=1 by default
sector_composite = sector_abs_mean * sector_std

size_means = size_stats['bias_mean'].round(0)
size_abs_mean = abs(size_means.mean())
size_std = size_means.std() # Pandas std() uses ddof=1 by default
size_composite = size_abs_mean * size_std

bias_index = (sector_composite + size_composite) / 2

t_test_results = {}

if high_bias_sector != 'N/A' and low_bias_sector != 'N/A':
    high_sector_bias = set_grouped[set_grouped['sector'] == high_bias_sector]['bias_score']
    low_sector_bias  = set_grouped[set_grouped['sector'] == low_bias_sector]['bias_score']
    if not high_sector_bias.empty and not low_sector_bias.empty:
        stat, pval = ttest_ind(high_sector_bias, low_sector_bias, nan_policy='omit')
        mean_diff = high_sector_bias.mean() - low_sector_bias.mean()
        t_test_results['sector_comparison'] = {
            'high_bias_group': str(high_bias_sector),
            'low_bias_group': str(low_bias_sector),
            'mean_diff': round(float(mean_diff), 4),
            't_statistic': round(float(stat), 4),
            'p_value': float(pval),
        }

if high_bias_size != 'N/A' and low_bias_size != 'N/A':
    high_size_bias = set_grouped[set_grouped['marketcap_group'] == high_bias_size]['bias_score']
    low_size_bias  = set_grouped[set_grouped['marketcap_group'] == low_bias_size]['bias_score']
    if not high_size_bias.empty and not low_size_bias.empty:
        stat, pval = ttest_ind(high_size_bias, low_size_bias, nan_policy='omit')
        mean_diff = high_size_bias.mean() - low_size_bias.mean()
        t_test_results['size_comparison'] = {
            'high_bias_group': str(high_bias_size),
            'low_bias_group': str(low_bias_size),
            'mean_diff': round(float(mean_diff), 4),
            't_statistic': round(float(stat), 4),
            'p_value': float(pval),
        }

# ────────────── Save Results ──────────────
def format_stats_dict(stats_df):
    if stats_df.empty:
        return {}
    out = {}
    for idx, row in stats_df.iterrows():
        out[str(idx)] = {
            'bias_mean': round(float(row['bias_mean']), 0),
            'bias_std': round(float(row['bias_std']), 0),
        }
    return out

summary = {
    'bias_index': int(round(bias_index)),
    'sector_stats': format_stats_dict(sector_stats),
    'size_stats': format_stats_dict(size_stats),
    'bias_result': {
        'high_bias_sector': str(high_bias_sector),
        'low_bias_sector': str(low_bias_sector),
        'high_bias_size_group': str(high_bias_size),
        'low_bias_size_group': str(low_bias_size)
    },
    't_test_results': t_test_results
}

summary_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_result.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# ────────────── Cleanup ──────────────
if created_combined:
    file_paths = glob.glob(file_pattern)
    removed = 0
    for path in file_paths:
        try:
            os.remove(path)
            removed += 1
        except OSError as e:
            print(f"Error removing file {path}: {e}")
    print(f"\nCleanup complete. Removed {removed} individual CSV files.")
else:
    print("\nSkipped cleanup: used existing combined CSV (no set files removed).")