import pandas as pd
import json
import os
import glob
import numpy as np
from scipy.stats import chisquare
import argparse


# ────────────── Configuration ──────────────
parser = argparse.ArgumentParser(description="Aggregate intensity experiment results for a given model.")
parser.add_argument("--model-id", type=str, required=True, help="ID of the model to aggregate results for")
parser.add_argument("--reasoning-effort", type=str, default=None,
                   choices=["low", "medium", "high"],
                   help="Reasoning effort level (must match the value used during experiment)")
parser.add_argument("--output-dir", type=str, default="./result", help="Directory where the result CSVs and output summary are stored")
args = parser.parse_args()

MODEL_ID = args.model_id
SAVE_DIR = args.output_dir
MODEL_FILE_PREFIX = MODEL_ID.split('/')[-1]

if args.reasoning_effort:
    MODEL_FILE_PREFIX = f"{MODEL_FILE_PREFIX}_{args.reasoning_effort}"

os.makedirs(SAVE_DIR, exist_ok=True)

# ────────────── Load & Combine Data ──────────────
file_pattern = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_str_set_*.csv')
file_paths = glob.glob(file_pattern)

if not file_paths:
    print(f"Error: No CSV files found for model '{MODEL_ID}' with pattern '{file_pattern}'.")
    exit()

df_list = []
# Enumerate to assign a set number to each file
for i, path in enumerate(file_paths, 1):
    temp_df = pd.read_csv(path)
    temp_df['set'] = i  # Add set number column
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
combined_csv_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_str_combined.csv')
df.to_csv(combined_csv_path, index=False)
print(f"Combined {len(file_paths)} CSV files into a single DataFrame with {len(df)} rows.")

# ────────────── Data Analysis ──────────────

# 1. Filter for valid 'buy' or 'sell' answers
valid_answers_df = df[df['llm_answer'].str.lower().isin(['buy', 'sell'])].copy()
if len(valid_answers_df) != len(df):
    print(f"Warning: Dropped {len(df) - len(valid_answers_df)} rows with invalid answers.")

# 2. Create the 'selected_view' column
valid_answers_df['selected_view'] = np.where(
    valid_answers_df['llm_answer'].str.lower() == 'buy',
    valid_answers_df['buy'],
    valid_answers_df['sell']
)

# 3. Calculate overall win rates for 'momentum' vs 'contrarian'
view_counts = valid_answers_df['selected_view'].value_counts()
total_valid_selections = view_counts.sum()

overall_win_rates = (view_counts / total_valid_selections).to_dict() if total_valid_selections > 0 else {}

# 4. Calculate per-set win rates and standard deviation
per_set_stats = {}
momentum_win_rates_list = []

if not valid_answers_df.empty:
    # Group by set and calculate value counts for each
    set_grouped = valid_answers_df.groupby('set')['selected_view'].value_counts().unstack(fill_value=0)
    
    # Ensure both momentum and contrarian columns exist
    for view in ['momentum', 'contrarian']:
        if view not in set_grouped:
            set_grouped[view] = 0
            
    set_grouped['total'] = set_grouped['momentum'] + set_grouped['contrarian']
    set_grouped['momentum_win_rate'] = set_grouped['momentum'] / set_grouped['total']
    
    # Store the results in a dictionary
    per_set_stats = {
        f"set_{idx}": {
            "momentum_win_rate": row['momentum_win_rate'],
            "contrarian_win_rate": 1 - row['momentum_win_rate'],
            "momentum_count": int(row['momentum']),
            "contrarian_count": int(row['contrarian'])
        }
        for idx, row in set_grouped.iterrows()
    }
    momentum_win_rates_list = set_grouped['momentum_win_rate'].tolist()

# Calculate standard deviation of the momentum win rates
win_rate_std_dev = np.std(momentum_win_rates_list) if momentum_win_rates_list else 0.0

# 5. Perform overall Chi-squared goodness of fit test
observed_frequencies = [view_counts.get('momentum', 0), view_counts.get('contrarian', 0)]
chi_squared_result = {}
if sum(observed_frequencies) > 0:
    stat, pval = chisquare(f_obs=observed_frequencies)
    chi_squared_result = {
        'statistic': round(float(stat), 4),
        'p_value': float(pval),
        'observed_counts': {
            'momentum': int(observed_frequencies[0]),
            'contrarian': int(observed_frequencies[1])
        }
    }
else:
    print("Skipping Chi-squared test due to no valid data.")

# ────────────── Save Results ──────────────
preference_result = {
    'momentum_win_rate': round(overall_win_rates.get('momentum', 0.0), 4),
    'contrarian_win_rate': round(overall_win_rates.get('contrarian', 0.0), 4),
    'win_rate_std': round(float(win_rate_std_dev), 4)
}

summary = {
    'model_id': MODEL_ID,
    'total_selections': int(total_valid_selections),
    'preference_result': preference_result,  
    'per_set_stats': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k,v in per_set_stats.items()},
    'chi_squared_test': chi_squared_result
}

summary_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_str_result.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)


# ────────────── Cleanup ──────────────
for path in file_paths:
    try:
        os.remove(path)
    except OSError as e:
        print(f"Error removing file {path}: {e}")

print(f"\nCleanup complete. Removed {len(file_paths)} individual CSV files.")