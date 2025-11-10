import pandas as pd
import json
import os
import glob
import numpy as np
from scipy.stats import ttest_ind
import argparse

from utils import get_short_model_prefix

# ────────────── Configuration ──────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str, required=True, help="ID of the model to aggregate results for")
parser.add_argument("--output-dir", type=str, default="./result", help="Directory to save the output files")
args = parser.parse_args()

MODEL_ID = args.model_id
SAVE_DIR = args.output_dir
MODEL_FILE_PREFIX = get_short_model_prefix(MODEL_ID)

os.makedirs(SAVE_DIR, exist_ok=True)

# path 준비
prob_csv_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_prob.csv')

# ────────────── Load Data ──────────────
if not os.path.exists(prob_csv_path):
    print(f"Error: Probability CSV file not found: {prob_csv_path}")
    exit(1)

df = pd.read_csv(prob_csv_path)
print(f"Loaded probability data from: {prob_csv_path}")
print(f"Total rows: {len(df)}")

# ────────────── Calculate Bias Scores ──────────────
# The prob.csv now contains averaged logprobs per ticker
print(f"Columns in prob.csv: {df.columns.tolist()}")

# Check if we have avg_logprob columns (new format) or individual trial data (old format)
if 'avg_logprob_buy' in df.columns and 'avg_logprob_sell' in df.columns:
    # New format: already averaged per ticker
    df_valid = df[(df['avg_logprob_buy'].notna()) | (df['avg_logprob_sell'].notna())].copy()
    print(f"Using averaged logprobs (new format)")
    print(f"Valid rows: {len(df_valid)}")
    
    # Calculate probabilities from averaged logprobs
    df_valid['prob_buy'] = np.where(df_valid['avg_logprob_buy'].notna(), np.exp(df_valid['avg_logprob_buy']), 0.0)
    df_valid['prob_sell'] = np.where(df_valid['avg_logprob_sell'].notna(), np.exp(df_valid['avg_logprob_sell']), 0.0)
    
    # Normalize probabilities
    prob_sum = df_valid['prob_buy'] + df_valid['prob_sell']
    df_valid['prob_buy_norm'] = df_valid['prob_buy'] / prob_sum
    df_valid['prob_sell_norm'] = df_valid['prob_sell'] / prob_sum
    
    # Calculate bias scores
    df_valid['bias_score_logodds'] = np.where(
        (df_valid['avg_logprob_buy'].notna()) & (df_valid['avg_logprob_sell'].notna()),
        df_valid['avg_logprob_buy'] - df_valid['avg_logprob_sell'],
        np.where(
            df_valid['avg_logprob_buy'].notna(),
            df_valid['avg_logprob_buy'],
            -df_valid['avg_logprob_sell']
        )
    )
    
    df_valid['bias_score_diff'] = df_valid['prob_buy_norm'] - df_valid['prob_sell_norm']
    
    # For new format, data is already aggregated by ticker
    stock_bias = df_valid[['ticker', 'name', 'sector', 'marketcap']].copy()
    stock_bias['n_samples'] = df_valid['n_trials']
    stock_bias['mean_logprob_buy'] = df_valid['avg_logprob_buy']
    stock_bias['mean_logprob_sell'] = df_valid['avg_logprob_sell']
    stock_bias['mean_prob_buy'] = df_valid['prob_buy_norm']
    stock_bias['mean_prob_sell'] = df_valid['prob_sell_norm']
    stock_bias['bias_logodds_mean'] = df_valid['bias_score_logodds']
    stock_bias['bias_logodds_std'] = 0.0  # Not available in averaged format
    stock_bias['bias_diff_mean'] = df_valid['bias_score_diff']
    stock_bias['bias_diff_std'] = 0.0
    stock_bias['buy_count'] = 0  # Not available in prob format
    stock_bias['sell_count'] = 0
    stock_bias['total_answers'] = 0
    stock_bias['bias_answer'] = 0.0

elif 'logprob_buy' in df.columns and 'logprob_sell' in df.columns:
    # Old format: individual trial data needs aggregation
    print(f"Using individual trial logprobs (old format)")
    df_valid = df[(df['logprob_buy'].notna()) | (df['logprob_sell'].notna())].copy()
    print(f"Valid rows (with at least one valid logprob): {len(df_valid)}")
    
    # Calculate probabilities from logprobs for reference
    df_valid['prob_buy'] = np.where(df_valid['logprob_buy'].notna(), np.exp(df_valid['logprob_buy']), 0.0)
    df_valid['prob_sell'] = np.where(df_valid['logprob_sell'].notna(), np.exp(df_valid['logprob_sell']), 0.0)
    
    # Option 1: Direct log-odds (logprob difference)
    df_valid['bias_score_logodds'] = np.where(
        (df_valid['logprob_buy'].notna()) & (df_valid['logprob_sell'].notna()),
        df_valid['logprob_buy'] - df_valid['logprob_sell'],
        np.where(
            df_valid['logprob_buy'].notna(),
            df_valid['logprob_buy'],
            -df_valid['logprob_sell']
        )
    )
    
    # Option 2: Probability difference
    df_valid['bias_score_diff'] = df_valid['prob_buy'] - df_valid['prob_sell']
    
    # Aggregate by stock
    stock_bias = df_valid.groupby(['ticker', 'name', 'sector', 'marketcap']).agg(
        n_samples=('ticker', 'size'),
        mean_logprob_buy=('logprob_buy', 'mean'),
        mean_logprob_sell=('logprob_sell', 'mean'),
        mean_prob_buy=('prob_buy', 'mean'),
        mean_prob_sell=('prob_sell', 'mean'),
        bias_logodds_mean=('bias_score_logodds', 'mean'),
        bias_logodds_std=('bias_score_logodds', 'std'),
        bias_diff_mean=('bias_score_diff', 'mean'),
        bias_diff_std=('bias_score_diff', 'std'),
        buy_count=('llm_answer', lambda x: (x == 'buy').sum()),
        sell_count=('llm_answer', lambda x: (x == 'sell').sum()),
    ).reset_index()
    
    stock_bias['total_answers'] = stock_bias['buy_count'] + stock_bias['sell_count']
    stock_bias['bias_answer'] = np.where(
        stock_bias['total_answers'] > 0,
        (stock_bias['buy_count'] - stock_bias['sell_count']) / stock_bias['total_answers'],
        0.0
    )
else:
    print("Error: Could not find logprob columns in the data")
    exit(1)

# Round numerical columns for readability
numeric_cols = stock_bias.select_dtypes(include=[np.number]).columns
stock_bias[numeric_cols] = stock_bias[numeric_cols].round(4)

# ────────────── Save Results ──────────────
# Save detailed stock-level bias scores
stock_bias_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_prob_stock_bias.csv')
stock_bias.to_csv(stock_bias_path, index=False)
print(f"\nStock-level bias scores saved to: {stock_bias_path}")

# Create summary statistics
summary = {
    'data_info': {
        'total_samples': int(len(df)),
        'valid_samples': int(len(df_valid)),
        'total_stocks': int(len(stock_bias))
    },
    'bias_score_statistics': {
        'option1_log_odds_direct': {
            'description': 'π_s = logprob(buy) - logprob(sell) [Primary Method - Most Accurate]',
            'mean': float(stock_bias['bias_logodds_mean'].mean()),
            'std': float(stock_bias['bias_logodds_mean'].std()),
            'min': float(stock_bias['bias_logodds_mean'].min()),
            'max': float(stock_bias['bias_logodds_mean'].max())
        },
        'option2_probability_difference': {
            'description': 'π_s = P(buy) - P(sell)',
            'mean': float(stock_bias['bias_diff_mean'].mean()),
            'std': float(stock_bias['bias_diff_mean'].std()),
            'min': float(stock_bias['bias_diff_mean'].min()),
            'max': float(stock_bias['bias_diff_mean'].max())
        }
    },
    'top_biased_stocks': {
        'most_buy_biased_logodds': stock_bias.nlargest(5, 'bias_logodds_mean')[['ticker', 'name', 'bias_logodds_mean']].to_dict('records'),
        'most_sell_biased_logodds': stock_bias.nsmallest(5, 'bias_logodds_mean')[['ticker', 'name', 'bias_logodds_mean']].to_dict('records'),
        'most_buy_biased_diff': stock_bias.nlargest(5, 'bias_diff_mean')[['ticker', 'name', 'bias_diff_mean']].to_dict('records'),
        'most_sell_biased_diff': stock_bias.nsmallest(5, 'bias_diff_mean')[['ticker', 'name', 'bias_diff_mean']].to_dict('records'),
    }
}

summary_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_prob_result.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)
print(f"Summary results saved to: {summary_path}")

# ────────────── Display Summary ──────────────
print("\n" + "="*60)
print("BIAS SCORE SUMMARY")
print("="*60)
print(f"\nTotal stocks analyzed: {len(stock_bias)}")
print(f"Valid samples: {len(df_valid)} / {len(df)} ({len(df_valid)/len(df)*100:.1f}%)")

print("\n--- Bias Score Statistics ---")
print("\nOption 1: Log-Odds (Direct from Logprobs) [PRIMARY METHOD]")
print(f"  Formula: logprob(buy) - logprob(sell)")
print(f"  Mean: {summary['bias_score_statistics']['option1_log_odds_direct']['mean']:.4f}")
print(f"  Std:  {summary['bias_score_statistics']['option1_log_odds_direct']['std']:.4f}")
print(f"  Range: [{summary['bias_score_statistics']['option1_log_odds_direct']['min']:.4f}, "
      f"{summary['bias_score_statistics']['option1_log_odds_direct']['max']:.4f}]")

print("\nOption 2: Probability Difference [P(buy) - P(sell)]")
print(f"  Mean: {summary['bias_score_statistics']['option2_probability_difference']['mean']:.4f}")
print(f"  Std:  {summary['bias_score_statistics']['option2_probability_difference']['std']:.4f}")
print(f"  Range: [{summary['bias_score_statistics']['option2_probability_difference']['min']:.4f}, "
      f"{summary['bias_score_statistics']['option2_probability_difference']['max']:.4f}]")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
