import argparse
import glob
import os
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = ["llm_answer", "original_llm_answer", "added_evidence_count"]


def load_bias_index(csv_path: str) -> dict[str, int]:
    """Load model bias_index from CSV file."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["model"], df["bias_index"]))


@dataclass(frozen=True)
class FlipStats:
    model: str
    added_evidence_count: int
    flip_rate: float
    n: int


def _normalize_answer(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()


def _extract_model_name(filename: str) -> str:
    # Expected: <model>_weight_evidence_<...>.csv
    # Example: gpt-5.2_none_weight_evidence_bias_score_bottom_10_abs.csv
    marker = "_weight_evidence_"
    if marker in filename:
        return filename.split(marker, 1)[0]
    return os.path.splitext(filename)[0]


def _extract_group_name(filename: str) -> str:
    # Expected suffix contains either top_10_abs or bottom_10_abs for bias_score verification.
    if "top_10_abs" in filename:
        return "top_10_abs"
    if "bottom_10_abs" in filename:
        return "bottom_10_abs"
    return "unknown"


def load_and_concat_by_model_and_group(csv_files: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    by_key: dict[tuple[str, str], list[pd.DataFrame]] = {}

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        model = _extract_model_name(filename)
        group = _extract_group_name(filename)

        df = pd.read_csv(file_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {filename}")

        df = df[REQUIRED_COLUMNS].copy()
        df = df.dropna(subset=REQUIRED_COLUMNS)
        df["llm_answer"] = _normalize_answer(df["llm_answer"])
        df["original_llm_answer"] = _normalize_answer(df["original_llm_answer"])

        # Ensure evidence count is numeric (sometimes read as str)
        df["added_evidence_count"] = pd.to_numeric(df["added_evidence_count"], errors="coerce")
        df = df.dropna(subset=["added_evidence_count"])
        df["added_evidence_count"] = df["added_evidence_count"].astype(int)

        by_key.setdefault((model, group), []).append(df)

    return {key: pd.concat(dfs, ignore_index=True) for key, dfs in by_key.items()}


def calculate_flip_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_flipped"] = df["llm_answer"] != df["original_llm_answer"]

    stats = (
        df.groupby("added_evidence_count")["is_flipped"]
        .agg([("flip_rate", "mean"), ("n", "count")])
        .reset_index()
        .sort_values("added_evidence_count")
    )
    return stats


# Pretty color palette for models
MODEL_COLORS = [
    "#4E79A7",  # steel blue
    "#F28E2B",  # orange
    "#E15759",  # coral red
    "#76B7B2",  # teal
    "#59A14F",  # green
    "#EDC948",  # gold
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
    "#86BCB6",  # light teal
    "#D37295",  # magenta
]


def _plot_grouped_by_model(
    ax: plt.Axes,
    stats_df: pd.DataFrame,
    title: str,
    bias_index: dict[str, int] | None = None,
    model_color_map: dict[str, str] | None = None,
) -> None:
    """Plot flip rate with models on x-axis, evidence 1 & 2 as grouped bars."""
    if stats_df.empty:
        ax.set_title(f"{title} (no data)")
        ax.set_axis_off()
        return

    # Sort models by bias_index descending (highest first)
    if bias_index:
        models = sorted(
            stats_df["model"].unique(),
            key=lambda m: (-bias_index.get(m, 0), m.lower()),
        )
    else:
        models = sorted(stats_df["model"].unique(), key=lambda s: s.lower())

    evidence_counts = [1, 2]
    x_positions = range(len(models))
    bar_width = 0.35

    # Create lookup for flip rates
    rate_lookup = {}
    for _, row in stats_df.iterrows():
        rate_lookup[(row["model"], row["added_evidence_count"])] = row["flip_rate"]

    for i, ev_count in enumerate(evidence_counts):
        offsets = [x + (i - 0.5) * bar_width for x in x_positions]
        y_vals = [rate_lookup.get((m, ev_count), 0.0) for m in models]
        colors = [model_color_map.get(m, "#888888") for m in models] if model_color_map else None

        bars = ax.bar(
            offsets,
            y_vals,
            width=bar_width,
            label=f"evidence={ev_count}",
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9 if ev_count == 1 else 0.6,
        )

        # Add bias_index above bars (only for evidence=2 to avoid clutter)
        if ev_count == 2 and bias_index:
            for bar, model in zip(bars, models):
                if model in bias_index:
                    height = bar.get_height()
                    ax.annotate(
                        str(bias_index[model]),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("flip rate")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def plot_top_vs_bottom(
    group_to_stats: dict[str, pd.DataFrame],
    output_path: str,
    bias_index: dict[str, int] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Build model -> color mapping (consistent across both plots)
    all_models = set()
    for stats_df in group_to_stats.values():
        if not stats_df.empty:
            all_models.update(stats_df["model"].unique())

    # Sort models by bias_index descending for consistent coloring
    if bias_index:
        sorted_models = sorted(all_models, key=lambda m: (-bias_index.get(m, 0), m.lower()))
    else:
        sorted_models = sorted(all_models, key=lambda s: s.lower())

    model_color_map = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(sorted_models)}

    # Left plot: top_10_abs
    _plot_grouped_by_model(
        axes[0],
        group_to_stats.get("top_10_abs", pd.DataFrame()),
        title="Top 10% bias_score tickers",
        bias_index=bias_index,
        model_color_map=model_color_map,
    )

    # Right plot: bottom_10_abs
    _plot_grouped_by_model(
        axes[1],
        group_to_stats.get("bottom_10_abs", pd.DataFrame()),
        title="Bottom 10% bias_score tickers",
        bias_index=bias_index,
        model_color_map=model_color_map,
    )

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute flip rate (llm_answer != original_llm_answer) grouped by added_evidence_count, "
            "aggregate by model across all CSVs, and plot all models in one figure."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="bias_verification",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_weight_evidence_*.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("bias_verification", "verification_flip_rate_top_vs_bottom.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--bias_index_csv",
        type=str,
        default=os.path.join("mixed_result", "model_bias_ranks.csv"),
        help="Path to CSV with model bias_index values",
    )
    args = parser.parse_args()

    # Load bias_index for sorting and labeling
    bias_index: dict[str, int] | None = None
    if os.path.exists(args.bias_index_csv):
        bias_index = load_bias_index(args.bias_index_csv)
        print(f"Loaded bias_index from: {args.bias_index_csv}")
    else:
        print(f"Warning: bias_index CSV not found at {args.bias_index_csv}, skipping bias_index features")

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not csv_files:
        raise SystemExit(
            f"No CSV files found in {args.input_dir} matching pattern '{args.pattern}'."
        )

    key_to_df = load_and_concat_by_model_and_group(csv_files)

    # Print overall flip rates per (group, model)
    overall_rows = []
    for (model, group), df in sorted(key_to_df.items(), key=lambda x: (x[0][1], x[0][0].lower())):
        is_flipped = df["llm_answer"] != df["original_llm_answer"]
        overall_rows.append(
            {
                "group": group,
                "model": model,
                "overall_flip_rate": float(is_flipped.mean()),
                "n": int(is_flipped.shape[0]),
            }
        )
    overall_df = pd.DataFrame(overall_rows)
    print("\nOverall flip rate by group/model:")
    if not overall_df.empty:
        print(overall_df.sort_values(["group", "model"]).to_string(index=False))

    # Stats for plotting: group -> dataframe with columns [model, added_evidence_count, flip_rate, n]
    group_to_stats: dict[str, pd.DataFrame] = {}
    for (model, group), df in key_to_df.items():
        stats = calculate_flip_stats(df)
        stats.insert(0, "model", model)
        group_to_stats[group] = (
            pd.concat([group_to_stats.get(group, pd.DataFrame()), stats], ignore_index=True)
            if group in group_to_stats
            else stats
        )

    # Print grouped stats for sanity-check
    print("\nFlip rate by added_evidence_count (group -> model):")
    for group, stats_df in sorted(group_to_stats.items()):
        print(f"\n[{group}]")
        if stats_df.empty:
            print("(empty)")
        else:
            print(stats_df.sort_values(["model", "added_evidence_count"]).to_string(index=False))

    plot_top_vs_bottom(group_to_stats, args.output, bias_index=bias_index)
    print(f"\nSaved figure to: {args.output}")


if __name__ == "__main__":
    main()
