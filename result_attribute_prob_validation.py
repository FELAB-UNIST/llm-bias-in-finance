import pandas as pd
import json
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_rel, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from utils import get_short_model_prefix

# ────────────── Configuration ──────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str, required=True, help="ID of the model to validate")
parser.add_argument("--output-dir", type=str, default="./result", help="Directory containing the result files")
args = parser.parse_args()

MODEL_ID = args.model_id
SAVE_DIR = args.output_dir
MODEL_FILE_PREFIX = get_short_model_prefix(MODEL_ID)

os.makedirs(SAVE_DIR, exist_ok=True)

# ────────────── Load Data ──────────────
# Load combined CSV (multiple trials)
combined_csv_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_combined.csv')
if not os.path.exists(combined_csv_path):
    print(f"Error: Combined CSV file not found: {combined_csv_path}")
    exit(1)

# Load probability-based results
prob_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_prob.csv')
if not os.path.exists(prob_path):
    print(f"Error: Probability-based CSV file not found: {prob_path}")
    exit(1)

# Load combined data and calculate bias scores
df_combined = pd.read_csv(combined_csv_path)
print(f"Loaded combined data: {len(df_combined)} rows")

# Check if 'set' column exists
if 'set' not in df_combined.columns:
    print("Error: 'set' column not found in combined CSV")
    exit(1)

print(f"Available sets: {sorted(df_combined['set'].unique())}")

# Filter only set 1 for comparison with prob
df_set1 = df_combined[df_combined['set'] == 1].copy()
print(f"Set 1 data: {len(df_set1)} rows")

if len(df_set1) == 0:
    print("Error: No data found for set 1")
    exit(1)

df_set1['is_buy'] = df_set1['llm_answer'].str.lower() == 'buy'
df_set1['is_sell'] = df_set1['llm_answer'].str.lower() == 'sell'

# Calculate answer-based bias score for set 1
set1_bias = df_set1.groupby(['ticker', 'name', 'sector', 'marketcap']).agg(
    buy_count=('is_buy', 'sum'),
    sell_count=('is_sell', 'sum')
).reset_index()
set1_bias['total_count'] = set1_bias['buy_count'] + set1_bias['sell_count']
set1_bias['bias_answer_set1'] = np.where(
    set1_bias['total_count'] > 0,
    (set1_bias['buy_count'] - set1_bias['sell_count']) / set1_bias['total_count'],
    0.0
)

# Load probability-based bias scores
prob_bias = pd.read_csv(prob_path)
print(f"Loaded probability-based data: {len(prob_bias)} stocks")

# Calculate bias scores from avg_logprob
prob_bias['mean_prob_buy'] = np.exp(prob_bias['avg_logprob_buy'])
prob_bias['mean_prob_sell'] = np.exp(prob_bias['avg_logprob_sell'])

# Normalize probabilities
prob_sum = prob_bias['mean_prob_buy'] + prob_bias['mean_prob_sell']
prob_bias['mean_prob_buy'] = prob_bias['mean_prob_buy'] / prob_sum
prob_bias['mean_prob_sell'] = prob_bias['mean_prob_sell'] / prob_sum

# Calculate bias metrics
prob_bias['bias_diff_mean'] = prob_bias['mean_prob_buy'] - prob_bias['mean_prob_sell']
prob_bias['bias_logodds_mean'] = np.log(prob_bias['mean_prob_buy'] / prob_bias['mean_prob_sell'])
prob_bias['n_samples'] = prob_bias['n_trials']

# ────────────── Merge Data ──────────────
# Merge on ticker to compare set 1 bias scores with probability-based bias scores
merged = pd.merge(
    set1_bias[['ticker', 'name', 'sector', 'marketcap', 'bias_answer_set1', 'buy_count', 'sell_count']],
    prob_bias[['ticker', 'bias_diff_mean', 'bias_logodds_mean', 
               'mean_prob_buy', 'mean_prob_sell', 'n_samples']],
    on='ticker',
    how='inner'
)

print(f"\nMerged data: {len(merged)} stocks")
print(f"Stocks in set 1 only: {len(set1_bias) - len(merged)}")
print(f"Stocks in prob only: {len(prob_bias) - len(merged)}")

# ────────────── Statistical Validation ──────────────
validation_results = {
    'data_info': {
        'n_stocks': int(len(merged)),
        'comparison_set': 'set_1',
        'n_set1_samples': int(set1_bias['total_count'].sum()),
        'n_prob_samples': int(prob_bias['n_samples'].sum())
    },
    'correlations': {},
    'agreement_analysis': {},
    'statistical_tests': {}
}

# 1. Correlation Analysis
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

# Compare each probability-based bias score with set 1 answer-based bias
for bias_type, bias_col in [
    ('Direct Difference', 'bias_diff_mean'),
    ('Log-Odds Ratio', 'bias_logodds_mean')
]:
    # Remove NaN and Inf values
    valid_data = merged[[bias_col, 'bias_answer_set1']].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_data) > 2:
        pearson_r, pearson_p = pearsonr(valid_data[bias_col], valid_data['bias_answer_set1'])
        spearman_r, spearman_p = spearmanr(valid_data[bias_col], valid_data['bias_answer_set1'])
        
        validation_results['correlations'][bias_type] = {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'n_valid_stocks': int(len(valid_data))
        }
        
        print(f"\n{bias_type} vs Set 1 Answer-based Bias:")
        print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})")
        print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4e})")
        print(f"  Valid stocks: {len(valid_data)}")

# 2. Agreement Analysis (Direction of bias)
print("\n" + "="*60)
print("DIRECTION AGREEMENT ANALYSIS")
print("="*60)

for bias_type, bias_col in [
    ('Direct Difference', 'bias_diff_mean'),
    ('Log-Odds Ratio', 'bias_logodds_mean')
]:
    valid_data = merged[[bias_col, 'bias_answer_set1']].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_data) > 0:
        # Classify direction: buy bias (>0), sell bias (<0), neutral (=0)
        prob_direction = np.sign(valid_data[bias_col])
        answer_direction = np.sign(valid_data['bias_answer_set1'])
        
        # Agreement: same sign
        agreement = (prob_direction == answer_direction).sum()
        agreement_rate = agreement / len(valid_data) * 100
        
        # Detailed breakdown
        both_buy = ((prob_direction > 0) & (answer_direction > 0)).sum()
        both_sell = ((prob_direction < 0) & (answer_direction < 0)).sum()
        both_neutral = ((prob_direction == 0) & (answer_direction == 0)).sum()
        disagreement = len(valid_data) - agreement
        
        validation_results['agreement_analysis'][bias_type] = {
            'agreement_rate': float(agreement_rate),
            'total_stocks': int(len(valid_data)),
            'both_buy_bias': int(both_buy),
            'both_sell_bias': int(both_sell),
            'both_neutral': int(both_neutral),
            'disagreement': int(disagreement)
        }
        
        print(f"\n{bias_type}:")
        print(f"  Agreement rate: {agreement_rate:.2f}%")
        print(f"  Both buy-biased: {both_buy}")
        print(f"  Both sell-biased: {both_sell}")
        print(f"  Both neutral: {both_neutral}")
        print(f"  Disagreement: {disagreement}")

# 3. Mean Difference Tests
print("\n" + "="*60)
print("MEAN DIFFERENCE TESTS")
print("="*60)

# Test if mean bias scores are significantly different
for bias_type, bias_col in [
    ('Direct Difference', 'bias_diff_mean')  # Most comparable to answer-based
]:
    valid_data = merged[[bias_col, 'bias_answer_set1']].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_data) > 2:
        # Paired t-test (same stocks)
        t_stat, p_val = ttest_rel(valid_data[bias_col], valid_data['bias_answer_set1'])
        
        mean_prob = valid_data[bias_col].mean()
        mean_answer = valid_data['bias_answer_set1'].mean()
        mean_diff = mean_prob - mean_answer
        
        validation_results['statistical_tests'][bias_type] = {
            'paired_t_statistic': float(t_stat),
            'paired_t_p_value': float(p_val),
            'mean_prob_bias': float(mean_prob),
            'mean_set1_bias': float(mean_answer),
            'mean_difference': float(mean_diff)
        }
        
        print(f"\n{bias_type} vs Set 1 Answer-based (Paired t-test):")
        print(f"  Mean prob-based bias: {mean_prob:.4f}")
        print(f"  Mean set 1 answer-based bias: {mean_answer:.4f}")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4e}")

# 4. Rank Correlation (Top/Bottom stocks)
print("\n" + "="*60)
print("TOP/BOTTOM STOCKS ANALYSIS")
print("="*60)

n_top = min(20, len(merged) // 5)  # Top 20 or 20% of stocks

for bias_type, bias_col in [
    ('Direct Difference', 'bias_diff_mean')
]:
    valid_data = merged[[bias_col, 'bias_answer_set1', 'ticker', 'name']].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_data) >= n_top:
        # Top buy-biased stocks by each method
        top_prob = set(valid_data.nlargest(n_top, bias_col)['ticker'])
        top_answer = set(valid_data.nlargest(n_top, 'bias_answer_set1')['ticker'])
        overlap_buy = len(top_prob & top_answer)
        
        # Top sell-biased stocks by each method
        bottom_prob = set(valid_data.nsmallest(n_top, bias_col)['ticker'])
        bottom_answer = set(valid_data.nsmallest(n_top, 'bias_answer_set1')['ticker'])
        overlap_sell = len(bottom_prob & bottom_answer)
        
        validation_results['agreement_analysis'][f'{bias_type}_top_stocks'] = {
            'n_top': int(n_top),
            'top_buy_overlap': int(overlap_buy),
            'top_buy_overlap_rate': float(overlap_buy / n_top * 100),
            'top_sell_overlap': int(overlap_sell),
            'top_sell_overlap_rate': float(overlap_sell / n_top * 100)
        }
        
        print(f"\n{bias_type} - Top {n_top} stocks:")
        print(f"  Buy-biased overlap: {overlap_buy}/{n_top} ({overlap_buy/n_top*100:.1f}%)")
        print(f"  Sell-biased overlap: {overlap_sell}/{n_top} ({overlap_sell/n_top*100:.1f}%)")

# ────────────── Visualization (Optional) ──────────────
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (bias_type, bias_col) in enumerate([
        ('Direct Diff', 'bias_diff_mean'),
        ('Log-Odds', 'bias_logodds_mean')
    ]):
        valid_data = merged[[bias_col, 'bias_answer_set1']].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(valid_data) > 0:
            ax = axes[idx]
            ax.scatter(valid_data[bias_col], valid_data['bias_answer_set1'], alpha=0.5)
            
            # Add diagonal line (perfect correlation)
            min_val = min(valid_data[bias_col].min(), valid_data['bias_answer_set1'].min())
            max_val = max(valid_data[bias_col].max(), valid_data['bias_answer_set1'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect correlation')
            
            # Add regression line
            z = np.polyfit(valid_data[bias_col], valid_data['bias_answer_set1'], 1)
            p = np.poly1d(z)
            ax.plot(valid_data[bias_col].sort_values(), 
                   p(valid_data[bias_col].sort_values()), 
                   "b-", alpha=0.5, label='Regression line')
            
            ax.set_xlabel(f'Prob-based Bias ({bias_type})')
            ax.set_ylabel('Set 1 Answer-based Bias')
            ax.set_title(f'{bias_type}\nr={validation_results["correlations"][bias_type]["pearson_r"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_prob_validation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nValidation plot saved to: {plot_path}")
    plt.close()
except Exception as e:
    print(f"\nWarning: Could not create visualization: {e}")

# ────────────── Save Results ──────────────
output_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_prob_validation.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(validation_results, f, indent=4, ensure_ascii=False)
print(f"\nValidation results saved to: {output_path}")

# ────────────── Summary ──────────────
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)

best_correlation = max(
    validation_results['correlations'].items(),
    key=lambda x: x[1]['pearson_r']
)
best_agreement = max(
    [(k, v) for k, v in validation_results['agreement_analysis'].items() if 'agreement_rate' in v],
    key=lambda x: x[1]['agreement_rate']
)

print(f"\nBest correlation: {best_correlation[0]}")
print(f"  Pearson r = {best_correlation[1]['pearson_r']:.4f}")
print(f"  p-value = {best_correlation[1]['pearson_p']:.4e}")

print(f"\nBest direction agreement: {best_agreement[0]}")
print(f"  Agreement rate = {best_agreement[1]['agreement_rate']:.2f}%")

# Recommendation
if best_correlation[1]['pearson_r'] > 0.7 and best_correlation[1]['pearson_p'] < 0.01:
    print("\n✅ RECOMMENDATION: Strong correlation found!")
    print("   Probability-based bias scores can serve as a good proxy for answer-based bias.")
elif best_correlation[1]['pearson_r'] > 0.5 and best_correlation[1]['pearson_p'] < 0.05:
    print("\n⚠️  RECOMMENDATION: Moderate correlation found.")
    print("   Probability-based bias scores show some agreement but may not fully replace multiple trials.")
else:
    print("\n❌ RECOMMENDATION: Weak correlation found.")
    print("   Probability-based bias scores may not be a reliable proxy for answer-based bias.")

print("\n" + "="*60)
print("Validation complete!")
print("="*60)
