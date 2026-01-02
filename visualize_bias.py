import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Directory containing the result files
result_dir = "./mix_v2"

# Find all *att_result.json files
file_pattern = os.path.join(result_dir, "*_att_result.json")
files = glob.glob(file_pattern)

sector_means = {}
sector_stds = {}
size_means = {}
size_stds = {}
bias_scores = []

# Load data
for file_path in files:
    filename = os.path.basename(file_path)
    # Extract model name: remove '_att_result.json'
    model_name = filename.replace("_att_result.json", "")
    
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}")
            continue

        if 'bias_index' in data:
            bias_scores.append({
                'model': model_name,
                'bias_index': data['bias_index']
            })
        
        # Process Sector Stats
        if 'sector_stats' in data:
            sector_stats = data['sector_stats']
            for sector, stats in sector_stats.items():
                if sector not in sector_means:
                    sector_means[sector] = {}
                if sector not in sector_stds:
                    sector_stds[sector] = {}
                
                sector_means[sector][model_name] = stats.get('bias_mean', 0)
                sector_stds[sector][model_name] = stats.get('bias_std', 0)

        # Process Size Stats
        if 'size_stats' in data:
            size_stats = data['size_stats']
            for size_group, stats in size_stats.items():
                if size_group not in size_means:
                    size_means[size_group] = {}
                if size_group not in size_stds:
                    size_stds[size_group] = {}
                
                size_means[size_group][model_name] = stats.get('bias_mean', 0)
                size_stds[size_group][model_name] = stats.get('bias_std', 0)

# Create DataFrames
df_mean = pd.DataFrame(sector_means).T  # Rows: Sectors, Cols: Models
df_std = pd.DataFrame(sector_stds).T    # Rows: Sectors, Cols: Models

df_size_mean = pd.DataFrame(size_means).T
df_size_std = pd.DataFrame(size_stds).T

# Transpose to have Models as Rows
df_mean = df_mean.transpose()
df_std = df_std.transpose()
df_size_mean = df_size_mean.transpose()
df_size_std = df_size_std.transpose()

# Define Sector Order
sector_order = [
    "Technology",
    "Energy",
    "Healthcare",
    "Communication Services",
    "Industrials",
    "Utilities",
    "Real Estate",
    "Basic Materials",
    "Consumer Cyclical",
    "Financial Services",
    "Consumer Defensive"
]

# Filter and reorder columns for Sector
existing_cols = [col for col in sector_order if col in df_mean.columns]
remaining_cols = [col for col in df_mean.columns if col not in existing_cols]
final_cols = existing_cols + remaining_cols

df_mean = df_mean[final_cols]
df_std = df_std[final_cols]

# Define Size Order
size_order = ["Q1", "Q2", "Q3", "Q4"]
existing_size_cols = [col for col in size_order if col in df_size_mean.columns]
remaining_size_cols = [col for col in df_size_mean.columns if col not in existing_size_cols]
final_size_cols = existing_size_cols + remaining_size_cols

df_size_mean = df_size_mean[final_size_cols]
df_size_std = df_size_std[final_size_cols]

# Generate Rank CSV (optional) and determine model order
if bias_scores:
    df_bias = pd.DataFrame(bias_scores)
    df_bias = df_bias.sort_values(by='bias_index', ascending=True)
    df_bias['rank'] = range(1, len(df_bias) + 1)
    df_bias = df_bias[['rank', 'model', 'bias_index']]

    rank_csv_path = os.path.join(result_dir, "model_bias_ranks.csv")
    df_bias.to_csv(rank_csv_path, index=False)
    print(f"Model bias ranks saved to {rank_csv_path}")

# Sort models by name (case-insensitive)
sorted_models = sorted(df_mean.index.tolist(), key=lambda s: str(s).lower())

df_mean = df_mean.loc[sorted_models]
df_std = df_std.loc[sorted_models]
# Ensure size dataframe has same models (handle missing if any, though unlikely)
df_size_mean = df_size_mean.reindex(sorted_models)
df_size_std = df_size_std.reindex(sorted_models)

# Rename columns to wrap text for better visualization (Sector only)
df_mean.columns = [col.replace(' ', '\n') for col in df_mean.columns]
df_std.columns = [col.replace(' ', '\n') for col in df_std.columns]

# Create annotation matrix for Sector
annot_matrix = df_mean.copy().astype(object)
for model in df_mean.index:
    for sector in df_mean.columns:
        mean_val = df_mean.loc[model, sector]
        std_val = df_std.loc[model, sector]
        # annot_matrix.loc[model, sector] = f"{mean_val:.2f}\n(±{std_val:.2f})"
        annot_matrix.loc[model, sector] = f"{mean_val:.0f}"

# Create annotation matrix for Size
annot_matrix_size = df_size_mean.copy().astype(object)
for model in df_size_mean.index:
    for size_group in df_size_mean.columns:
        mean_val = df_size_mean.loc[model, size_group]
        std_val = df_size_std.loc[model, size_group]
        # annot_matrix_size.loc[model, size_group] = f"{mean_val:.2f}\n(±{std_val:.2f})"
        annot_matrix_size.loc[model, size_group] = f"{mean_val:.0f}"

# Define Custom Colormap
# Increased saturation (s=100) and adjusted luminance (l=40) for better color distinction
cmap = sns.diverging_palette(10, 130, s=100, l=40, sep=5, as_cmap=True) 

# --- Plotting Sector Heatmap ---
plt.figure(figsize=(20, 8))
ax = sns.heatmap(
    df_mean,
    annot=annot_matrix.values,
    fmt="",
    cmap=cmap,
    center=0,
    vmin=-100,
    vmax=100,
    linewidths=0.5,
    annot_kws={"size": 18},
    cbar_kws={"label": "Bias Score"}
)
# Customize Colorbar
cbar = ax.collections[0].colorbar
cbar.set_label("Bias Score", size=14, weight='bold')
cbar.set_ticks([-100, -50, 0, 50, 100])
cbar.set_ticklabels(['-100\n(Sell)', '-50', '0', '50', '100\n(Buy)'])

plt.xlabel("", fontsize=14)
plt.ylabel("", fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(rotation=0, fontsize=12, weight='bold')

plt.tight_layout()
output_path_sector = os.path.join(result_dir, "bias_heatmap_sector.png")
plt.savefig(output_path_sector, dpi=300, bbox_inches='tight')
print(f"Sector Heatmap saved to {output_path_sector}")
plt.close()

# --- Plotting Size Heatmap ---
plt.figure(figsize=(10, 8)) # Adjusted width for size heatmap
ax = sns.heatmap(
    df_size_mean,
    annot=annot_matrix_size.values,
    fmt="",
    cmap=cmap,
    center=0,
    vmin=-100,
    vmax=100,
    linewidths=0.5,
    annot_kws={"size": 18},
    cbar_kws={"label": "Bias Score"}
)
# Customize Colorbar
cbar = ax.collections[0].colorbar
cbar.set_label("Bias Score", size=14, weight='bold')
cbar.set_ticks([-100, -50, 0, 50, 100])
cbar.set_ticklabels(['-100\n(Sell)', '-50', '0', '50', '100\n(Buy)'])

plt.xlabel("", fontsize=14)
plt.ylabel("", fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(rotation=0, fontsize=12, weight='bold')

plt.tight_layout()
output_path_size = os.path.join(result_dir, "bias_heatmap_size.png")
plt.savefig(output_path_size, dpi=300, bbox_inches='tight')
print(f"Size Heatmap saved to {output_path_size}")
plt.close()
