"""Generate all 10 bias visualizations."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from scipy import stats

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

OUT_DIR = 'mixed_result/viz'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & prepare data ──────────────────────────────────────────────
MODEL_FILES = sorted([f for f in os.listdir('mixed_result') if f.endswith('_att_combined.csv')])
MODELS = [f.replace('_att_combined.csv', '') for f in MODEL_FILES]

US_MODELS = ['gpt-4.1', 'gpt-5.2_none', 'gemini-3-flash-preview', 'grok-4.1-fast', 'llama-4-maverick', 'mistral-large-2512']
CN_MODELS = ['deepseek-chat-v3-0324', 'deepseek-v3.2', 'glm-4.7', 'kimi-k2-0905', 'qwen3-235b-a22b-2507']

# Short display names
SHORT_NAMES = {
    'deepseek-chat-v3-0324': 'DeepSeek-v3\n(0324)',
    'deepseek-v3.2': 'DeepSeek\nv3.2',
    'gemini-3-flash-preview': 'Gemini-3\nFlash',
    'glm-4.7': 'GLM-4.7',
    'gpt-4.1': 'GPT-4.1',
    'gpt-5.2_none': 'GPT-5.2',
    'grok-4.1-fast': 'Grok-4.1',
    'kimi-k2-0905': 'Kimi-K2',
    'llama-4-maverick': 'Llama-4\nMaverick',
    'mistral-large-2512': 'Mistral\nLarge',
    'qwen3-235b-a22b-2507': 'Qwen3\n235B',
}

# Approximate release dates (YYYY-MM)
RELEASE_DATES = {
    'deepseek-chat-v3-0324': '2025-03',
    'gpt-4.1': '2025-04',
    'llama-4-maverick': '2025-04',
    'deepseek-v3.2': '2025-06',
    'glm-4.7': '2025-06',
    'qwen3-235b-a22b-2507': '2025-07',
    'kimi-k2-0905': '2025-09',
    'grok-4.1-fast': '2025-10',
    'mistral-large-2512': '2025-12',
    'gemini-3-flash-preview': '2026-01',
    'gpt-5.2_none': '2026-02',
}

RELEASE_COMPANY = {
    'deepseek-chat-v3-0324': 'DeepSeek',
    'deepseek-v3.2': 'DeepSeek',
    'glm-4.7': 'Zhipu',
    'kimi-k2-0905': 'Moonshot',
    'qwen3-235b-a22b-2507': 'Alibaba',
    'gpt-4.1': 'OpenAI',
    'gpt-5.2_none': 'OpenAI',
    'gemini-3-flash-preview': 'Google',
    'grok-4.1-fast': 'xAI',
    'llama-4-maverick': 'Meta',
    'mistral-large-2512': 'Mistral',
}

US_COLOR = '#4472C4'
CN_COLOR = '#E05252'

# Load all model data → per-ticker bias scores
all_data = {}  # model -> DataFrame
ticker_bias = {}  # model -> Series(ticker -> bias_score in [-100,100])
ticker_buy_rate = {}  # model -> Series(ticker -> buy_rate 0~1)

# Also keep set-level data for viz #10
set_level_bias = {}  # model -> DataFrame(ticker, set, bias_score)

ref_df = None
for model in MODELS:
    df = pd.read_csv(f'mixed_result/{model}_att_combined.csv')
    all_data[model] = df
    if ref_df is None:
        ref_df = df.drop_duplicates('ticker')[['ticker', 'name', 'sector', 'marketcap']].copy()

    # Buy rate per ticker (across all sets & trials)
    buy_rate = df.groupby('ticker')['llm_answer'].apply(
        lambda x: (x == 'buy').sum() / x.notna().sum() if x.notna().sum() > 0 else np.nan
    )
    ticker_buy_rate[model] = buy_rate
    ticker_bias[model] = (buy_rate - 0.5) * 200  # scale to [-100, 100]

    # Set-level bias
    set_buy = df.groupby(['ticker', 'set'])['llm_answer'].apply(
        lambda x: ((x == 'buy').sum() / x.notna().sum() - 0.5) * 200 if x.notna().sum() > 0 else np.nan
    ).reset_index(name='bias_score')
    set_level_bias[model] = set_buy

bias_df = pd.DataFrame(ticker_bias)  # rows=tickers, cols=models
buy_rate_df = pd.DataFrame(ticker_buy_rate)
ref_df = ref_df.set_index('ticker')

# Overall bias per model
model_avg_bias = bias_df.mean()

print("Data loaded. Starting visualizations...\n")


# ════════════════════════════════════════════════════════════════════
# 1. US vs China – Strip + Box Plot
# ════════════════════════════════════════════════════════════════════
def viz1():
    fig, ax = plt.subplots(figsize=(8, 5))

    us_biases = [model_avg_bias[m] for m in US_MODELS]
    cn_biases = [model_avg_bias[m] for m in CN_MODELS]

    # Box plots
    bp = ax.boxplot([us_biases, cn_biases], positions=[0, 1], widths=0.5,
                    patch_artist=True, showfliers=False, zorder=1)
    bp['boxes'][0].set_facecolor(US_COLOR + '30')
    bp['boxes'][0].set_edgecolor(US_COLOR)
    bp['boxes'][1].set_facecolor(CN_COLOR + '30')
    bp['boxes'][1].set_edgecolor(CN_COLOR)
    for element in ['whiskers', 'caps']:
        for i, item in enumerate(bp[element]):
            item.set_color(US_COLOR if i < 2 else CN_COLOR)
    for m in bp['medians']:
        m.set_color('black')

    # Strip (jittered points)
    np.random.seed(42)
    for i, (models, color) in enumerate([(US_MODELS, US_COLOR), (CN_MODELS, CN_COLOR)]):
        jitter = np.random.uniform(-0.15, 0.15, len(models))
        vals = [model_avg_bias[m] for m in models]
        ax.scatter(i + jitter, vals, c=color, s=80, zorder=3, edgecolors='white', linewidth=0.8)
        for j, m in enumerate(models):
            short = SHORT_NAMES[m].replace('\n', ' ')
            ax.annotate(short, (i + jitter[j], vals[j]),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=7, color=color, va='center')

    ax.axhline(0, color='grey', ls='--', lw=0.8, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['US Models', 'China Models'], fontsize=12)
    ax.set_ylabel('Average Bias Score')
    ax.set_title('Fig 1. US vs China Models – Average Bias Score', fontweight='bold')
    ax.set_xlim(-0.6, 1.8)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig01_us_vs_china_box.png')
    plt.close(fig)
    print("✓ Fig 1 saved")

viz1()


# ════════════════════════════════════════════════════════════════════
# 2. US vs China × Sector Heatmap
# ════════════════════════════════════════════════════════════════════
def viz2():
    sectors = ref_df['sector'].unique()
    rows = []
    for group_name, models in [('US Models', US_MODELS), ('China Models', CN_MODELS)]:
        for sector in sorted(sectors):
            tickers_in_sector = ref_df[ref_df['sector'] == sector].index
            vals = []
            for m in models:
                vals.extend(bias_df.loc[bias_df.index.isin(tickers_in_sector), m].dropna().tolist())
            rows.append({'group': group_name, 'sector': sector, 'bias': np.mean(vals) if vals else 0})

    pivot = pd.DataFrame(rows).pivot(index='group', columns='sector', values='bias')
    # Sort columns by absolute difference
    diff = abs(pivot.loc['US Models'] - pivot.loc['China Models'])
    pivot = pivot[diff.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-80, vmax=80)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha='right', fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(pivot.index, fontsize=11)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f'{pivot.values[i, j]:.0f}', ha='center', va='center', fontsize=10, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Bias Score')
    ax.set_title('Fig 2. US vs China Models – Bias by Sector', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig02_us_china_sector.png')
    plt.close(fig)
    print("✓ Fig 2 saved")

viz2()


# ════════════════════════════════════════════════════════════════════
# 3. Ticker-level Agreement / Disagreement Scatter
# ════════════════════════════════════════════════════════════════════
def viz3():
    mean_bias = bias_df.mean(axis=1)
    std_bias = bias_df.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(mean_bias, std_bias, c=mean_bias, cmap='RdYlGn',
                         s=15, alpha=0.6, edgecolors='grey', linewidth=0.3)
    fig.colorbar(scatter, ax=ax, label='Mean Bias Score', shrink=0.8)

    # Label notable tickers
    highlight_tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'GOOG', 'META', 'JPM', 'XOM', 'JNJ']
    for t in highlight_tickers:
        if t in mean_bias.index:
            ax.annotate(t, (mean_bias[t], std_bias[t]),
                        fontsize=8, fontweight='bold',
                        textcoords='offset points', xytext=(5, 5))

    # Also label extremes
    top_disagree = std_bias.nlargest(5).index
    top_agree = std_bias.nsmallest(5).index
    for t in list(top_disagree) + list(top_agree):
        if t not in highlight_tickers:
            ax.annotate(t, (mean_bias[t], std_bias[t]),
                        fontsize=7, color='grey',
                        textcoords='offset points', xytext=(5, 3))

    ax.set_xlabel('Mean Bias Score (across all models)')
    ax.set_ylabel('Std Dev of Bias Score (model disagreement)')
    ax.set_title('Fig 3. Model Agreement per Ticker', fontweight='bold')
    ax.axvline(0, color='grey', ls='--', lw=0.8)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig03_ticker_agreement.png')
    plt.close(fig)
    print("✓ Fig 3 saved")

viz3()


# ════════════════════════════════════════════════════════════════════
# 4. Largest US–China Model Divergence per Ticker
# ════════════════════════════════════════════════════════════════════
def viz4():
    us_avg = bias_df[US_MODELS].mean(axis=1)
    cn_avg = bias_df[CN_MODELS].mean(axis=1)
    diff = cn_avg - us_avg  # positive = China more buy-biased

    # Top/bottom 15
    top_cn = diff.nlargest(15)
    top_us = diff.nsmallest(15)
    show = pd.concat([top_cn, top_us]).sort_values()

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [CN_COLOR if v > 0 else US_COLOR for v in show.values]
    bars = ax.barh(range(len(show)), show.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(show)))
    labels = [f"{t} ({ref_df.loc[t, 'name'][:25]})" if t in ref_df.index else t for t in show.index]
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Bias Difference (China avg − US avg)')
    ax.set_title('Fig 4. Tickers with Largest US–China Model Divergence', fontweight='bold')

    # Legend
    us_patch = mpatches.Patch(color=US_COLOR, label='US models more Buy-biased')
    cn_patch = mpatches.Patch(color=CN_COLOR, label='China models more Buy-biased')
    ax.legend(handles=[us_patch, cn_patch], loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig04_us_china_divergence.png')
    plt.close(fig)
    print("✓ Fig 4 saved")

viz4()


# ════════════════════════════════════════════════════════════════════
# 5. Bias Score Distribution – Ridge Plot
# ════════════════════════════════════════════════════════════════════
def viz5():
    from scipy.stats import gaussian_kde

    # Sort models by mean bias
    order = model_avg_bias.sort_values().index.tolist()

    fig, axes = plt.subplots(len(order), 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=-0.3)

    x_grid = np.linspace(-100, 100, 300)

    for i, model in enumerate(order):
        ax = axes[i]
        vals = bias_df[model].dropna().values
        if len(vals) < 5:
            continue
        kde = gaussian_kde(vals, bw_method=0.3)
        density = kde(x_grid)

        color = CN_COLOR if model in CN_MODELS else US_COLOR
        ax.fill_between(x_grid, density, alpha=0.5, color=color)
        ax.plot(x_grid, density, color=color, lw=1.2)
        ax.axvline(0, color='grey', ls='--', lw=0.5)
        ax.axvline(np.mean(vals), color=color, ls='-', lw=1.5, alpha=0.7)

        ax.set_yticks([])
        ax.set_ylabel(SHORT_NAMES[model].replace('\n', ' '), fontsize=9, rotation=0,
                       labelpad=80, va='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if i < len(order) - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom=False)

    axes[-1].set_xlabel('Bias Score')
    axes[0].set_title('Fig 5. Bias Score Distribution per Model (Ridge Plot)', fontweight='bold', pad=15)

    us_patch = mpatches.Patch(color=US_COLOR, alpha=0.5, label='US Model')
    cn_patch = mpatches.Patch(color=CN_COLOR, alpha=0.5, label='China Model')
    axes[0].legend(handles=[us_patch, cn_patch], loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig05_ridge_plot.png')
    plt.close(fig)
    print("✓ Fig 5 saved")

viz5()


# ════════════════════════════════════════════════════════════════════
# 6. Release Date Timeline
# ════════════════════════════════════════════════════════════════════
def viz6():
    dates = pd.to_datetime([RELEASE_DATES[m] for m in MODELS])
    abs_bias = [abs(model_avg_bias[m]) for m in MODELS]
    companies = [RELEASE_COMPANY[m] for m in MODELS]

    fig, ax = plt.subplots(figsize=(12, 6))

    for m, d, b in zip(MODELS, dates, abs_bias):
        color = CN_COLOR if m in CN_MODELS else US_COLOR
        ax.scatter(d, b, c=color, s=120, zorder=3, edgecolors='white', linewidth=1)
        short = SHORT_NAMES[m].replace('\n', ' ')
        ax.annotate(short, (d, b), textcoords='offset points', xytext=(0, 10),
                    fontsize=8, ha='center', fontweight='bold', color=color)

    # Connect same-company models
    company_models = {}
    for m in MODELS:
        c = RELEASE_COMPANY[m]
        company_models.setdefault(c, []).append(m)
    for company, ms in company_models.items():
        if len(ms) > 1:
            ms_sorted = sorted(ms, key=lambda m: RELEASE_DATES[m])
            for a, b_m in zip(ms_sorted[:-1], ms_sorted[1:]):
                color = CN_COLOR if a in CN_MODELS else US_COLOR
                ax.plot([pd.to_datetime(RELEASE_DATES[a]), pd.to_datetime(RELEASE_DATES[b_m])],
                        [abs(model_avg_bias[a]), abs(model_avg_bias[b_m])],
                        color=color, ls='--', lw=1, alpha=0.5)

    # Trend line
    x_num = np.array([(d - dates.min()).days for d in dates], dtype=float)
    slope, intercept, r, p, se = stats.linregress(x_num, abs_bias)
    x_line = np.linspace(x_num.min(), x_num.max(), 100)
    ax.plot([dates.min() + pd.Timedelta(days=d) for d in x_line],
            intercept + slope * x_line, 'k--', alpha=0.3, lw=1.5,
            label=f'Trend (r={r:.2f}, p={p:.2f})')

    ax.set_xlabel('Release Date (approx.)')
    ax.set_ylabel('|Bias Index| (absolute)')
    ax.set_title('Fig 6. Model Bias over Release Timeline', fontweight='bold')
    ax.legend(fontsize=9)

    us_patch = mpatches.Patch(color=US_COLOR, label='US Model')
    cn_patch = mpatches.Patch(color=CN_COLOR, label='China Model')
    ax.legend(handles=[us_patch, cn_patch, plt.Line2D([0],[0], color='k', ls='--', alpha=0.3, label=f'Trend (r={r:.2f}, p={p:.2f})')],
              fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig06_timeline.png')
    plt.close(fig)
    print("✓ Fig 6 saved")

viz6()


# ════════════════════════════════════════════════════════════════════
# 7. Bias Direction Flip Matrix (Alluvial-style pairwise)
# ════════════════════════════════════════════════════════════════════
def viz7():
    # For each pair of models, compute % of tickers that AGREE on direction
    decisions = (bias_df > 0).astype(int)  # 1=buy-biased, 0=sell-biased
    n = len(MODELS)
    agree_vals = np.zeros((n, n))
    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            v1 = decisions[m1].dropna()
            v2 = decisions[m2].dropna()
            common = v1.index.intersection(v2.index)
            if len(common) > 0:
                agree_vals[i, j] = (v1[common].values == v2[common].values).mean() * 100
    agree_matrix = pd.DataFrame(agree_vals, index=MODELS, columns=MODELS)

    # Sort by model origin
    order = [m for m in US_MODELS if m in MODELS] + [m for m in CN_MODELS if m in MODELS]
    agree_matrix = agree_matrix.loc[order, order]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(agree_matrix.values, cmap='YlGn', vmin=40, vmax=100)
    short_labels = [SHORT_NAMES[m].replace('\n', ' ') for m in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(short_labels, fontsize=9)

    for i in range(len(order)):
        for j in range(len(order)):
            val = agree_matrix.values[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=8, color=color)

    # Draw border between US and CN groups
    n_us = len([m for m in US_MODELS if m in MODELS])
    ax.axhline(n_us - 0.5, color='black', lw=2)
    ax.axvline(n_us - 0.5, color='black', lw=2)

    # Labels for quadrants
    ax.text(n_us / 2 - 0.5, -1.5, 'US', ha='center', fontsize=11, fontweight='bold', color=US_COLOR)
    ax.text(n_us + (len(order) - n_us) / 2 - 0.5, -1.5, 'China', ha='center', fontsize=11, fontweight='bold', color=CN_COLOR)

    fig.colorbar(im, ax=ax, shrink=0.7, label='Agreement Rate (%)')
    ax.set_title('Fig 7. Decision Agreement Rate Between Model Pairs', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig07_agreement_matrix.png')
    plt.close(fig)
    print("✓ Fig 7 saved")

viz7()


# ════════════════════════════════════════════════════════════════════
# 8. Radar / Spider Chart – Sector Profile
# ════════════════════════════════════════════════════════════════════
def viz8():
    sectors = sorted(ref_df['sector'].unique())
    n_sectors = len(sectors)
    angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False).tolist()
    angles += angles[:1]  # close

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))

    for ax, (group_name, models, color) in zip(axes,
            [('US Models', US_MODELS, US_COLOR), ('China Models', CN_MODELS, CN_COLOR)]):
        for m in models:
            vals = []
            for s in sectors:
                tickers_in = ref_df[ref_df['sector'] == s].index
                v = bias_df.loc[bias_df.index.isin(tickers_in), m].mean()
                vals.append(v)
            vals += vals[:1]
            short = SHORT_NAMES[m].replace('\n', ' ')
            ax.plot(angles, vals, lw=1.5, label=short, alpha=0.7)
            ax.fill(angles, vals, alpha=0.05)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(sectors, fontsize=7)
        ax.set_title(group_name, fontweight='bold', pad=20, color=color)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)

    fig.suptitle('Fig 8. Sector Bias Profile per Model (Radar)', fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig08_radar.png', bbox_inches='tight')
    plt.close(fig)
    print("✓ Fig 8 saved")

viz8()


# ════════════════════════════════════════════════════════════════════
# 9. Market Cap vs Bias Scatter (with regression per model)
# ════════════════════════════════════════════════════════════════════
def viz9():
    log_mcap = np.log10(ref_df['marketcap'])

    fig, ax = plt.subplots(figsize=(11, 7))

    slopes = {}
    for m in MODELS:
        color = CN_COLOR if m in CN_MODELS else US_COLOR
        valid_tickers = bias_df[m].dropna().index.intersection(log_mcap.index)
        x = log_mcap[valid_tickers].values
        y = bias_df.loc[valid_tickers, m].values

        # Regression line only
        slope, intercept, r, p, se = stats.linregress(x, y)
        slopes[m] = slope
        x_line = np.linspace(x.min(), x.max(), 50)
        short = SHORT_NAMES[m].replace('\n', ' ')
        ax.plot(x_line, intercept + slope * x_line, color=color, lw=1.5, alpha=0.6,
                label=f'{short} (slope={slope:.1f})')

    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.set_xlabel('Market Cap (log₁₀ USD)')
    ax.set_ylabel('Bias Score')
    ax.set_title('Fig 9. Market Cap vs Bias Score (Regression Lines)', fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='upper left')

    # Add secondary x-axis labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    mcap_labels = [1e9, 1e10, 1e11, 1e12, 3e12]
    mcap_ticks = [np.log10(v) for v in mcap_labels]
    mcap_strs = ['$1B', '$10B', '$100B', '$1T', '$3T']
    ax2.set_xticks(mcap_ticks)
    ax2.set_xticklabels(mcap_strs, fontsize=9)
    ax2.set_xlabel('Market Cap')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig09_mcap_bias.png')
    plt.close(fig)
    print("✓ Fig 9 saved")

viz9()


# ════════════════════════════════════════════════════════════════════
# 10. Set-level Consistency (Test-Retest Reliability)
# ════════════════════════════════════════════════════════════════════
def viz10():
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        sdf = set_level_bias[model]
        pivot = sdf.pivot_table(index='ticker', columns='set', values='bias_score')

        if 1 in pivot.columns and 2 in pivot.columns:
            valid = pivot[[1, 2]].dropna()
            r12, _ = stats.pearsonr(valid[1], valid[2])
            color = CN_COLOR if model in CN_MODELS else US_COLOR

            ax.scatter(valid[1], valid[2], s=8, alpha=0.4, color=color, edgecolors='none')
            ax.plot([-100, 100], [-100, 100], 'k--', lw=0.8, alpha=0.3)
            ax.set_xlim(-105, 105)
            ax.set_ylim(-105, 105)

            short = SHORT_NAMES[model].replace('\n', ' ')
            ax.set_title(f'{short}\nr = {r12:.3f}', fontsize=10, color=color)
            ax.set_aspect('equal')
            if idx >= 8:
                ax.set_xlabel('Set 1 Bias')
            if idx % 4 == 0:
                ax.set_ylabel('Set 2 Bias')

    # Hide unused subplot
    if len(MODELS) < len(axes):
        for i in range(len(MODELS), len(axes)):
            axes[i].set_visible(False)

    fig.suptitle('Fig 10. Test-Retest Reliability (Set 1 vs Set 2)', fontweight='bold', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig10_test_retest.png', bbox_inches='tight')
    plt.close(fig)
    print("✓ Fig 10 saved")

viz10()


# ════════════════════════════════════════════════════════════════════
# 11. Market Cap × Model Correlation Analysis
# ════════════════════════════════════════════════════════════════════
def viz11():
    log_mcap = np.log10(ref_df['marketcap'])

    # ── Compute correlations per model ──
    rows = []
    for m in MODELS:
        valid_tickers = bias_df[m].dropna().index.intersection(log_mcap.index)
        x = log_mcap[valid_tickers].values
        y = bias_df.loc[valid_tickers, m].values

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        short = SHORT_NAMES[m].replace('\n', ' ')
        rows.append({
            'model': m,
            'display': short,
            'n': len(x),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'slope': slope,
        })

    corr_df = pd.DataFrame(rows).sort_values('pearson_r')

    # ── Also compute overall (all models pooled) ──
    all_x, all_y = [], []
    for m in MODELS:
        valid_tickers = bias_df[m].dropna().index.intersection(log_mcap.index)
        all_x.extend(log_mcap[valid_tickers].values)
        all_y.extend(bias_df.loc[valid_tickers, m].values)
    overall_pearson_r, overall_pearson_p = stats.pearsonr(all_x, all_y)
    overall_spearman_r, overall_spearman_p = stats.spearmanr(all_x, all_y)

    # ── Save CSV ──
    csv_path = f'{OUT_DIR}/fig11_mcap_correlation.csv'
    corr_df.to_csv(csv_path, index=False)
    print(f"  Correlation table saved to {csv_path}")

    # ── Plot A: Coefficient bar chart (Pearson + Spearman side-by-side) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: bar chart (no US/China grouping – uniform color per model)
    ax = axes[0]
    y_pos = np.arange(len(corr_df))
    bar_h = 0.35
    cmap_models = plt.cm.tab20(np.linspace(0, 1, len(corr_df)))

    bars1 = ax.barh(y_pos - bar_h/2, corr_df['pearson_r'], bar_h,
                     color=cmap_models, alpha=0.85, edgecolor='white', label='Pearson r')
    bars2 = ax.barh(y_pos + bar_h/2, corr_df['spearman_r'], bar_h,
                     color=cmap_models, alpha=0.45, edgecolor='white', hatch='//', label='Spearman ρ')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(corr_df['display'], fontsize=10, fontweight='bold')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('Market Cap – Bias Score Correlation per Model', fontweight='bold')

    # Add significance markers
    for i, row in enumerate(corr_df.itertuples()):
        for val, p_val, offset in [(row.pearson_r, row.pearson_p, -bar_h/2),
                                    (row.spearman_r, row.spearman_p, bar_h/2)]:
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            if sig:
                x_pos = val + (0.01 if val >= 0 else -0.01)
                ha = 'left' if val >= 0 else 'right'
                ax.text(x_pos, i + offset, sig, fontsize=9, va='center', ha=ha, fontweight='bold')

    ax.legend(handles=[mpatches.Patch(facecolor='grey', alpha=0.85, label='Pearson r'),
                        mpatches.Patch(facecolor='grey', alpha=0.45, hatch='//', label='Spearman ρ')],
              fontsize=9, loc='lower right')

    # Right panel: per-ticker scatter + regression line per model
    ax2 = axes[1]
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))

    for idx, m in enumerate(MODELS):
        valid_tickers = bias_df[m].dropna().index.intersection(log_mcap.index)
        x = log_mcap[valid_tickers].values
        y = bias_df.loc[valid_tickers, m].values
        short = SHORT_NAMES[m].replace('\n', ' ')

        # Scatter (small, semi-transparent)
        ax2.scatter(x, y, s=5, alpha=0.15, color=model_colors[idx], rasterized=True)

        # Regression line
        slope, intercept, _, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax2.plot(x_line, intercept + slope * x_line,
                 color=model_colors[idx], lw=2, alpha=0.85, label=short)

    ax2.axhline(0, color='grey', ls='--', lw=0.8)
    ax2.set_xlabel('Market Cap (log₁₀ USD)')
    ax2.set_ylabel('Bias Score')
    ax2.set_title(f'Pooled: Pearson r={overall_pearson_r:.3f} (p={overall_pearson_p:.1e})\n'
                  f'Spearman ρ={overall_spearman_r:.3f} (p={overall_spearman_p:.1e})',
                  fontweight='bold', fontsize=11)
    ax2.legend(fontsize=7, loc='upper left', ncol=2)

    # Secondary x-axis
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    mcap_labels = [1e9, 1e10, 1e11, 1e12, 3e12]
    mcap_ticks = [np.log10(v) for v in mcap_labels]
    mcap_strs = ['$1B', '$10B', '$100B', '$1T', '$3T']
    ax2_top.set_xticks(mcap_ticks)
    ax2_top.set_xticklabels(mcap_strs, fontsize=9)

    fig.suptitle('Fig 11. Market Cap × Bias Score – Correlation Analysis', fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig11_mcap_correlation.png', bbox_inches='tight')
    plt.close(fig)

    # ── Print summary ──
    print("\n  ── Market Cap × Bias Correlation Summary ──")
    print(f"  {'Model':<22} {'Pearson r':>10} {'p-value':>10} {'Spearman ρ':>11} {'p-value':>10}")
    print(f"  {'─'*65}")
    for _, row in corr_df.iterrows():
        sig_p = '***' if row['pearson_p'] < 0.001 else '**' if row['pearson_p'] < 0.01 else '*' if row['pearson_p'] < 0.05 else ''
        sig_s = '***' if row['spearman_p'] < 0.001 else '**' if row['spearman_p'] < 0.01 else '*' if row['spearman_p'] < 0.05 else ''
        print(f"  {row['display']:<22} {row['pearson_r']:>8.4f}{sig_p:<2} {row['pearson_p']:>10.2e} {row['spearman_r']:>9.4f}{sig_s:<2} {row['spearman_p']:>10.2e}")
    print(f"  {'─'*65}")
    print(f"  {'Overall (pooled)':<22} {overall_pearson_r:>8.4f}   {overall_pearson_p:>10.2e} {overall_spearman_r:>9.4f}   {overall_spearman_p:>10.2e}")
    print(f"\n  * p<0.05  ** p<0.01  *** p<0.001")
    print("✓ Fig 11 saved")

viz11()

print(f"\n✅ All 11 figures saved in {OUT_DIR}/")
