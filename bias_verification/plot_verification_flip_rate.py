import argparse
import glob
import os
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = ["llm_answer", "original_llm_answer", "added_evidence_count"]


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


def _plot_grouped_bar(ax: plt.Axes, stats_df: pd.DataFrame, title: str) -> None:
    if stats_df.empty:
        ax.set_title(f"{title} (no data)")
        ax.set_axis_off()
        return

    models = sorted(stats_df["model"].unique(), key=lambda s: s.lower())
    x_vals = sorted(stats_df["added_evidence_count"].unique())

    x_positions = list(range(len(x_vals)))
    bar_width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        model_df = stats_df[stats_df["model"] == model].set_index("added_evidence_count")
        y = [float(model_df.loc[x, "flip_rate"]) if x in model_df.index else 0.0 for x in x_vals]
        offsets = [p + (i - (len(models) - 1) / 2) * bar_width for p in x_positions]
        ax.bar(offsets, y, width=bar_width, label=model)

    ax.set_title(title)
    ax.set_xlabel("added_evidence_count")
    ax.set_ylabel("flip rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(x) for x in x_vals])
    ax.grid(True, axis="y", alpha=0.3)


def plot_top_vs_bottom(group_to_stats: dict[str, pd.DataFrame], output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    _plot_grouped_bar(
        axes[0],
        group_to_stats.get("top_10_abs", pd.DataFrame()),
        "Top 10% bias_score tickers",
    )
    _plot_grouped_bar(
        axes[1],
        group_to_stats.get("bottom_10_abs", pd.DataFrame()),
        "Bottom 10% bias_score tickers",
    )

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="model", loc="upper center", ncol=min(len(labels), 4))

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200)


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
        default="mix_v2_verification",
        help="Folder containing *_weight_evidence_*.csv files (default: mix_v2_verification)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_weight_evidence_*.csv",
        help="Glob pattern for CSV files (default: *_weight_evidence_*.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("mix_v2_verification", "verification_flip_rate_top_vs_bottom.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()

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

    plot_top_vs_bottom(group_to_stats, args.output)
    print(f"\nSaved figure to: {args.output}")


if __name__ == "__main__":
    main()
