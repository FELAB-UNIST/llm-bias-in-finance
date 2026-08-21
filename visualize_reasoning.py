"""Reasoning vs Non-Reasoning bias visualizations."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

OUT_DIR = 'mixed_result/viz'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Model pairs ──────────────────────────────────────────────────────
PAIRS = {
    'DeepSeek v3.2': {
        'reasoning': 'reasoning_result/deepseek-v3.2_medium_att_combined.csv',
        'non_reasoning': 'mixed_result/deepseek-v3.2_att_combined.csv',
        'origin': 'CN',
    },
    'GLM-4.7': {
        'reasoning': 'reasoning_result/glm-4.7_medium_att_combined.csv',
        'non_reasoning': 'mixed_result/glm-4.7_att_combined.csv',
        'origin': 'CN',
    },
    'GPT-5.2': {
        'reasoning': 'reasoning_result/gpt-5.2_medium_att_combined.csv',
        'non_reasoning': 'mixed_result/gpt-5.2_none_att_combined.csv',
        'origin': 'US',
    },
    'Grok-4.1': {
        'reasoning': 'reasoning_result/grok-4.1-fast_medium_att_combined.csv',
        'non_reasoning': 'mixed_result/grok-4.1-fast_att_combined.csv',
        'origin': 'US',
    },
    'Kimi-K2': {
        'reasoning': 'reasoning_result/kimi-k2-thinking_medium_att_combined.csv',
        'non_reasoning': 'mixed_result/kimi-k2-0905_att_combined.csv',
        'origin': 'CN',
    },
}

US_COLOR = '#4472C4'
CN_COLOR = '#E05252'

def get_color(name):
    return CN_COLOR if PAIRS[name]['origin'] == 'CN' else US_COLOR

# ── Load data ────────────────────────────────────────────────────────
def compute_bias(path):
    df = pd.read_csv(path)
    bias = df.groupby('ticker')['llm_answer'].apply(
        lambda x: ((x == 'buy').sum() / x.notna().sum() - 0.5) * 200 if x.notna().sum() > 0 else np.nan
    )
    return bias, df

data = {}
ref_df = None
for name, cfg in PAIRS.items():
    r_bias, r_df = compute_bias(cfg['reasoning'])
    n_bias, n_df = compute_bias(cfg['non_reasoning'])
    if ref_df is None:
        ref_df = n_df.drop_duplicates('ticker')[['ticker', 'name', 'sector', 'marketcap']].set_index('ticker')
    data[name] = {'r': r_bias, 'n': n_bias}

MODEL_NAMES = list(PAIRS.keys())
print("Data loaded.\n")


# ════════════════════════════════════════════════════════════════════
# R1. Bias Shift Scatter (non-reasoning vs reasoning per ticker)
# ════════════════════════════════════════════════════════════════════
def viz_r1():
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True, sharex=True)
    for ax, name in zip(axes, MODEL_NAMES):
        n_bias = data[name]['n']
        r_bias = data[name]['r']
        common = n_bias.index.intersection(r_bias.index)
        color = get_color(name)

        ax.scatter(n_bias[common], r_bias[common], s=10, alpha=0.4, color=color, edgecolors='none')
        ax.plot([-100, 100], [-100, 100], 'k--', lw=0.8, alpha=0.3)
        ax.axhline(0, color='grey', ls=':', lw=0.5)
        ax.axvline(0, color='grey', ls=':', lw=0.5)
        ax.set_xlim(-105, 105)
        ax.set_ylim(-105, 105)
        ax.set_aspect('equal')
        ax.set_title(name, fontweight='bold', color=color)
        ax.set_xlabel('Non-Reasoning Bias')

        # Annotate shift direction
        above = (r_bias[common] > n_bias[common]).sum()
        below = len(common) - above
        pct_above = above / len(common) * 100
        ax.text(0.05, 0.95, f'{pct_above:.0f}% more buy\n{100-pct_above:.0f}% more sell',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0].set_ylabel('Reasoning Bias')
    fig.suptitle('Fig R1. Non-Reasoning vs Reasoning Bias per Ticker', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig_r1_scatter.png')
    plt.close(fig)
    print("✓ Fig R1 saved")

viz_r1()


# ════════════════════════════════════════════════════════════════════
# R2. Top/Bottom 10% Dumbbell Chart
# ════════════════════════════════════════════════════════════════════
def viz_r2():
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

    for ax, name in zip(axes, MODEL_NAMES):
        n_bias = data[name]['n']
        r_bias = data[name]['r']
        common = n_bias.index.intersection(r_bias.index)
        sorted_n = n_bias[common].sort_values()
        n10 = max(1, int(len(common) * 0.1))
        color = get_color(name)

        groups = {
            f'Bottom 10%\n(most sell)': sorted_n.head(n10).index,
            f'Q2 (10-30%)': sorted_n.iloc[n10:int(len(common)*0.3)].index,
            f'Middle\n(30-70%)': sorted_n.iloc[int(len(common)*0.3):int(len(common)*0.7)].index,
            f'Q4 (70-90%)': sorted_n.iloc[int(len(common)*0.7):int(len(common)*0.9)].index,
            f'Top 10%\n(most buy)': sorted_n.tail(n10).index,
        }

        y_pos = list(range(len(groups)))
        for i, (label, tickers) in enumerate(groups.items()):
            n_mean = n_bias[tickers].mean()
            r_mean = r_bias[tickers].mean()
            ax.plot([n_mean, r_mean], [i, i], color=color, lw=2, alpha=0.6)
            ax.scatter(n_mean, i, color='white', edgecolors=color, s=60, zorder=3, linewidth=1.5, label='Non-R' if i == 0 else None)
            ax.scatter(r_mean, i, color=color, s=60, zorder=3, edgecolors='white', linewidth=0.5, label='Reasoning' if i == 0 else None)

            # Arrow
            dx = r_mean - n_mean
            ax.annotate('', xy=(r_mean, i), xytext=(n_mean, i),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.6))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(groups.keys()), fontsize=8)
        ax.axvline(0, color='grey', ls='--', lw=0.8)
        ax.set_xlabel('Bias Score')
        ax.set_title(name, fontweight='bold', color=color)
        ax.legend(fontsize=7, loc='lower right')

    fig.suptitle('Fig R2. Bias Shift by Quintile (Non-Reasoning → Reasoning)', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig_r2_dumbbell.png')
    plt.close(fig)
    print("✓ Fig R2 saved")

viz_r2()


# ════════════════════════════════════════════════════════════════════
# R3. |Mean Bias| Paired Bar Chart
# ════════════════════════════════════════════════════════════════════
def viz_r3():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(MODEL_NAMES))
    width = 0.35

    n_abs = [abs(data[name]['n'].mean()) for name in MODEL_NAMES]
    r_abs = [abs(data[name]['r'].mean()) for name in MODEL_NAMES]
    colors = [get_color(name) for name in MODEL_NAMES]

    bars1 = ax.bar(x - width/2, n_abs, width, label='Non-Reasoning',
                   color=[c + '60' for c in colors], edgecolor=colors, linewidth=1.5)
    bars2 = ax.bar(x + width/2, r_abs, width, label='Reasoning',
                   color=colors, edgecolor=colors, linewidth=1.5)

    # Value labels
    for bar, val in zip(list(bars1) + list(bars2), n_abs + r_abs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Change arrows
    for i, name in enumerate(MODEL_NAMES):
        diff = r_abs[i] - n_abs[i]
        sign = '+' if diff > 0 else ''
        ax.text(x[i], max(n_abs[i], r_abs[i]) + 6, f'{sign}{diff:.1f}',
                ha='center', fontsize=9, color='red' if diff > 0 else 'green', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, fontsize=10)
    ax.set_ylabel('|Mean Bias Score|')
    ax.set_title('Fig R3. Absolute Bias: Non-Reasoning vs Reasoning', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(n_abs), max(r_abs)) + 15)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig_r3_abs_bias.png')
    plt.close(fig)
    print("✓ Fig R3 saved")

viz_r3()


# ════════════════════════════════════════════════════════════════════
# R4. Decile Bias Change (Line Plot)
# ════════════════════════════════════════════════════════════════════
def viz_r4():
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in MODEL_NAMES:
        n_bias = data[name]['n']
        r_bias = data[name]['r']
        common = n_bias.index.intersection(r_bias.index)
        sorted_n = n_bias[common].sort_values()

        n_deciles = 10
        decile_n = []
        decile_r = []
        for d in range(n_deciles):
            start = int(len(common) * d / n_deciles)
            end = int(len(common) * (d + 1) / n_deciles)
            tickers = sorted_n.iloc[start:end].index
            decile_n.append(n_bias[tickers].mean())
            decile_r.append(r_bias[tickers].mean())

        color = get_color(name)
        x = np.arange(1, n_deciles + 1)
        ax.plot(x, decile_n, 'o--', color=color, alpha=0.3, lw=1, markersize=4)
        ax.plot(x, decile_r, 'o-', color=color, alpha=0.9, lw=2, markersize=6, label=name)

    # Reference: perfect neutral
    ax.axhline(0, color='grey', ls='--', lw=0.8)

    # Annotation
    ax.text(0.02, 0.98, 'Dashed = Non-Reasoning\nSolid = Reasoning',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Decile (1=most sell-biased, 10=most buy-biased in Non-Reasoning)')
    ax.set_ylabel('Mean Bias Score')
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f'D{i}' for i in range(1, 11)])
    ax.set_title('Fig R4. Bias by Decile: Non-Reasoning vs Reasoning', fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig_r4_decile.png')
    plt.close(fig)
    print("✓ Fig R4 saved")

viz_r4()


# ════════════════════════════════════════════════════════════════════
# R5. Direction Flip Rate (Stacked Bar)
# ════════════════════════════════════════════════════════════════════
def viz_r5():
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = ['Buy→Sell', 'Sell→Buy', 'Buy→Buy', 'Sell→Sell']
    cat_colors = ['#E05252', '#4472C4', '#A8D5A2', '#F5C77E']

    x = np.arange(len(MODEL_NAMES))
    bottoms = np.zeros(len(MODEL_NAMES))

    cat_vals = {c: [] for c in categories}
    for name in MODEL_NAMES:
        n_bias = data[name]['n']
        r_bias = data[name]['r']
        common = n_bias.index.intersection(r_bias.index)
        n_dir = (n_bias[common] > 0)  # True = buy-biased
        r_dir = (r_bias[common] > 0)

        total = len(common)
        cat_vals['Buy→Sell'].append(((n_dir) & (~r_dir)).sum() / total * 100)
        cat_vals['Sell→Buy'].append(((~n_dir) & (r_dir)).sum() / total * 100)
        cat_vals['Buy→Buy'].append(((n_dir) & (r_dir)).sum() / total * 100)
        cat_vals['Sell→Sell'].append(((~n_dir) & (~r_dir)).sum() / total * 100)

    for cat, color in zip(categories, cat_colors):
        vals = cat_vals[cat]
        ax.bar(x, vals, bottom=bottoms, color=color, label=cat, edgecolor='white', linewidth=0.5)
        # Value labels
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 5:
                ax.text(x[i], b + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=8, fontweight='bold')
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, fontsize=10)
    ax.set_ylabel('% of Tickers')
    ax.set_title('Fig R5. Direction Change: Non-Reasoning → Reasoning', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig_r5_flip_rate.png')
    plt.close(fig)
    print("✓ Fig R5 saved")

viz_r5()


# ════════════════════════════════════════════════════════════════════
# R6. Sector-level Reasoning Impact Heatmap
# ════════════════════════════════════════════════════════════════════
def viz_r6():
    sectors = sorted(ref_df['sector'].unique())
    rows = []
    for name in MODEL_NAMES:
        n_bias = data[name]['n']
        r_bias = data[name]['r']
        for sector in sectors:
            tickers = ref_df[ref_df['sector'] == sector].index
            common = n_bias.index.intersection(r_bias.index).intersection(tickers)
            if len(common) > 0:
                diff = r_bias[common].mean() - n_bias[common].mean()
            else:
                diff = 0
            rows.append({'model': name, 'sector': sector, 'diff': diff})

    pivot = pd.DataFrame(rows).pivot(index='model', columns='sector', values='diff')
    # Reorder by model
    pivot = pivot.loc[MODEL_NAMES]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:+.0f}', ha='center', va='center', fontsize=9, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Bias Shift (Reasoning − Non-Reasoning)')
    ax.set_title('Fig R6. Reasoning Impact by Sector', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig_r6_sector_impact.png')
    plt.close(fig)
    print("✓ Fig R6 saved")

viz_r6()


# ════════════════════════════════════════════════════════════════════
# R7. Bias Variance Change (Std Dev comparison)
# ════════════════════════════════════════════════════════════════════
def viz_r7():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(MODEL_NAMES))
    width = 0.35

    n_std = [data[name]['n'].std() for name in MODEL_NAMES]
    r_std = [data[name]['r'].std() for name in MODEL_NAMES]
    colors = [get_color(name) for name in MODEL_NAMES]

    bars1 = ax.bar(x - width/2, n_std, width, label='Non-Reasoning',
                   color=[c + '60' for c in colors], edgecolor=colors, linewidth=1.5)
    bars2 = ax.bar(x + width/2, r_std, width, label='Reasoning',
                   color=colors, edgecolor=colors, linewidth=1.5)

    for bar, val in zip(list(bars1) + list(bars2), n_std + r_std):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for i in range(len(MODEL_NAMES)):
        diff = r_std[i] - n_std[i]
        sign = '+' if diff > 0 else ''
        ax.text(x[i], max(n_std[i], r_std[i]) + 3, f'{sign}{diff:.1f}',
                ha='center', fontsize=9, color='red' if diff > 0 else 'green', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, fontsize=10)
    ax.set_ylabel('Std Dev of Bias Score (across tickers)')
    ax.set_title('Fig R7. Bias Variance: Non-Reasoning vs Reasoning', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(n_std), max(r_std)) + 10)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig_r7_variance.png')
    plt.close(fig)
    print("✓ Fig R7 saved")

viz_r7()


# ════════════════════════════════════════════════════════════════════
# R8. Most Reasoning-Sensitive Tickers
# ════════════════════════════════════════════════════════════════════
def viz_r8():
    # Compute average |shift| across all 5 models per ticker
    all_shifts = pd.DataFrame()
    for name in MODEL_NAMES:
        n_bias = data[name]['n']
        r_bias = data[name]['r']
        common = n_bias.index.intersection(r_bias.index)
        shift = (r_bias[common] - n_bias[common]).abs()
        all_shifts[name] = shift

    mean_shift = all_shifts.mean(axis=1).sort_values(ascending=False)
    top20 = mean_shift.head(20)

    fig, ax = plt.subplots(figsize=(12, 7))

    y_pos = np.arange(len(top20))
    # Stacked horizontal bars showing each model's contribution
    left = np.zeros(len(top20))
    model_colors = [get_color(name) for name in MODEL_NAMES]
    hatches = ['', '//', '..', 'xx', '\\\\']

    for idx, name in enumerate(MODEL_NAMES):
        n_bias = data[name]['n']
        r_bias = data[name]['r']
        vals = []
        for ticker in top20.index:
            if ticker in n_bias.index and ticker in r_bias.index:
                vals.append(abs(r_bias[ticker] - n_bias[ticker]))
            else:
                vals.append(0)
        ax.barh(y_pos, vals, left=left, height=0.7,
                color=model_colors[idx], alpha=0.7, edgecolor='white', linewidth=0.5,
                label=name, hatch=hatches[idx])
        left += np.array(vals)

    labels = [f"{t} ({ref_df.loc[t, 'name'][:22]})" if t in ref_df.index else t for t in top20.index]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Sum of |Bias Shift| across models')
    ax.set_title('Fig R8. Most Reasoning-Sensitive Tickers (Top 20)', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig_r8_sensitive_tickers.png')
    plt.close(fig)
    print("✓ Fig R8 saved")

viz_r8()

print(f"\n✅ All 8 reasoning figures saved in {OUT_DIR}/")
