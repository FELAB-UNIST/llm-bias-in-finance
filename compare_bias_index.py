import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np

# Directories
reasoning_dir = "./reasoning_result"
mixed_dir = "./mixed_result"

def load_bias_data(directory):
    """Load bias_index from all *_att_result.json files in a directory."""
    file_pattern = os.path.join(directory, "*_att_result.json")
    files = glob.glob(file_pattern)

    data = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        model_name = filename.replace("_att_result.json", "")

        with open(file_path, 'r') as f:
            try:
                result = json.load(f)
                if 'bias_index' in result:
                    data[model_name] = result['bias_index']
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}")
                continue

    return data

# Load data from both directories
reasoning_data = load_bias_data(reasoning_dir)
mixed_data = load_bias_data(mixed_dir)

print("Reasoning Result Models:", list(reasoning_data.keys()))
print("Mixed Result Models:", list(mixed_data.keys()))

# Model name mapping (reasoning -> mixed)
model_mapping = {
    'deepseek-v3.2_medium': 'deepseek-v3.2',
    'glm-4.7_medium': 'glm-4.7',
    'gpt-5.2_medium': 'gpt-5.2_none',
    'grok-4.1-fast_medium': 'grok-4.1-fast',
    'kimi-k2-thinking_medium': 'kimi-k2-0905',
}

# Build comparison dataframe
comparison_data = []
for reasoning_model, mixed_model in model_mapping.items():
    if reasoning_model in reasoning_data and mixed_model in mixed_data:
        # Extract base model name for display
        display_name = reasoning_model.replace('_medium', '')
        comparison_data.append({
            'Model': display_name,
            'Reasoning (medium)': reasoning_data[reasoning_model],
            'Non-Reasoning': mixed_data[mixed_model]
        })

df = pd.DataFrame(comparison_data)
print("\nComparison Data:")
print(df)

# Calculate difference
df['Difference'] = df['Reasoning (medium)'] - df['Non-Reasoning']

# Sort by difference
df = df.sort_values('Difference', ascending=False)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Grouped Bar Chart ---
ax1 = axes[0]
x = np.arange(len(df))
width = 0.35

bars1 = ax1.bar(x - width/2, df['Reasoning (medium)'], width, label='Reasoning (medium)', color='#e74c3c', edgecolor='black')
bars2 = ax1.bar(x + width/2, df['Non-Reasoning'], width, label='Non-Reasoning', color='#3498db', edgecolor='black')

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Bias Index', fontsize=12, fontweight='bold')
ax1.set_title('Bias Index: Reasoning vs Non-Reasoning Models', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'], rotation=15, ha='right', fontsize=10)
ax1.legend(fontsize=10)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylim(0, max(df['Reasoning (medium)'].max(), df['Non-Reasoning'].max()) * 1.15)
ax1.grid(axis='y', alpha=0.3)

# --- Plot 2: Difference Chart ---
ax2 = axes[1]
colors = ['#27ae60' if d > 0 else '#c0392b' for d in df['Difference']]
bars3 = ax2.barh(df['Model'], df['Difference'], color=colors, edgecolor='black')

ax2.set_xlabel('Bias Index Difference\n(Reasoning - Non-Reasoning)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Model', fontsize=12, fontweight='bold')
ax2.set_title('Bias Index Difference\n(Positive = Reasoning has Higher Bias)', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars3, df['Difference']):
    width = bar.get_width()
    x_pos = width + 5 if width >= 0 else width - 5
    ha = 'left' if width >= 0 else 'right'
    ax2.annotate(f'{int(val):+d}',
                xy=(width, bar.get_y() + bar.get_height()/2),
                xytext=(3 if width >= 0 else -3, 0),
                textcoords="offset points",
                ha=ha, va='center', fontsize=10, fontweight='bold')

plt.tight_layout()

# Save figure
output_path = "./compare_bias_index.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to {output_path}")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nAverage Bias Index:")
print(f"  - Reasoning (medium): {df['Reasoning (medium)'].mean():.1f}")
print(f"  - Non-Reasoning:      {df['Non-Reasoning'].mean():.1f}")
print(f"  - Average Difference: {df['Difference'].mean():.1f}")

higher_count = (df['Difference'] > 0).sum()
lower_count = (df['Difference'] < 0).sum()
print(f"\n{higher_count}/{len(df)} models show HIGHER bias with reasoning")
print(f"{lower_count}/{len(df)} models show LOWER bias with reasoning")

if lower_count > 0:
    lower_models = df[df['Difference'] < 0]['Model'].tolist()
    print(f"  Exception(s): {', '.join(lower_models)}")
