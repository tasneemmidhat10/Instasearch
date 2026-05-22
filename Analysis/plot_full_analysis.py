from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".codex_pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ANALYSIS_DIR = Path(__file__).resolve().parent
FULL_ANALYSIS = ANALYSIS_DIR / "full_analysis.csv"
OOD_FAILURES = ANALYSIS_DIR / "ood_retrieval_failures_denovo_results.csv"
PLOT_DIR = ANALYSIS_DIR / "seaborn_plots"

BUCKET_ORDER = ["missing", "top1", "top5", "top10", "top50", "top100"]
MOVEMENT_ORDER = ["improved", "unchanged", "worsened", "still_missing"]
SOURCE_COLUMNS = [
    "sample_idx",
    "ground_truth",
    "stage1_rank",
    "rescored_rank",
    "rank_delta",
    "movement",
    "stage1_bucket",
    "rescored_bucket",
    "top1_candidate",
    "top1_score",
    "top1_correct",
    "top10_candidates",
]
PALETTE = {
    "Stage 1": "#4C78A8",
    "Rescored": "#F58518",
    "improved": "#54A24B",
    "unchanged": "#9D9DA3",
    "worsened": "#E45756",
    "still_missing": "#B279A2",
}


def base_sequence(seq: str) -> str:
    return re.sub(r"\([^)]+\)", "", str(seq))


def peptide_length(seq: str) -> int:
    return len(base_sequence(seq))


def expected_bucket(rank: pd.Series) -> np.ndarray:
    return np.select(
        [
            rank.eq(101),
            rank.eq(1),
            rank.le(5),
            rank.le(10),
            rank.le(50),
            rank.le(100),
        ],
        BUCKET_ORDER,
        default="UNKNOWN",
    )


def parse_top10_margin(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"(\d+):(.*) \[([-+0-9.eE]+)\]$")
    margins: list[float] = []
    gt_positions: list[float] = []
    parse_bad = 0
    count_bad = 0
    order_bad = 0
    first_matches = True

    for gt, top1, candidate_text in zip(
        df["ground_truth"], df["top1_candidate"], df["top10_candidates"]
    ):
        parts = str(candidate_text).split(" || ")
        if len(parts) != 10:
            count_bad += 1

        scores: list[float] = []
        candidates: list[str] = []
        gt_pos = math.nan
        previous_score = None

        for expected_idx, part in enumerate(parts, start=1):
            match = pattern.match(part)
            if not match:
                parse_bad += 1
                continue
            idx = int(match.group(1))
            candidate = match.group(2)
            score = float(match.group(3))
            if idx != expected_idx:
                parse_bad += 1
            if previous_score is not None and score > previous_score + 1e-9:
                order_bad += 1
            previous_score = score
            scores.append(score)
            candidates.append(candidate)
            if candidate == gt and math.isnan(gt_pos):
                gt_pos = expected_idx

        if candidates and candidates[0] != top1:
            first_matches = False
        margins.append(scores[0] - scores[1] if len(scores) >= 2 else math.nan)
        gt_positions.append(gt_pos)

    df = df.copy()
    df["top1_margin"] = margins
    df["gt_position_in_top10"] = gt_positions
    df.attrs["top10_validation"] = {
        "top10_parse_bad_items_or_indices": int(parse_bad),
        "top10_not_10_items": int(count_bad),
        "top10_score_order_bad_item_pairs": int(order_bad),
        "top10_first_matches_top1_candidate": bool(first_matches),
    }
    return df


def validate(df: pd.DataFrame) -> dict[str, object]:
    expected_delta = (df["stage1_rank"] - df["rescored_rank"]).astype(float)
    missing_mask = df["stage1_rank"].eq(101) & df["rescored_rank"].eq(101)
    expected_delta[missing_mask] = np.nan
    expected_movement = np.where(
        pd.isna(expected_delta),
        "still_missing",
        np.where(expected_delta > 0, "improved", np.where(expected_delta < 0, "worsened", "unchanged")),
    )
    expected_gt_pos = df["gt_position_in_top10"]
    top10_position_mismatch = (
        ((df["rescored_rank"] <= 10) & (expected_gt_pos != df["rescored_rank"]))
        | ((df["rescored_rank"] > 10) & expected_gt_pos.notna())
    )

    validation = {
        "rows": int(len(df)),
        "columns": int(len(SOURCE_COLUMNS)),
        "sample_idx_unique_sequential": bool(
            df["sample_idx"].is_unique
            and df["sample_idx"].min() == 0
            and df["sample_idx"].max() == len(df) - 1
        ),
        "nulls_except_rank_delta": int(df[SOURCE_COLUMNS].drop(columns=["rank_delta"]).isna().sum().sum()),
        "rank_delta_null_rows": int(df["rank_delta"].isna().sum()),
        "rank_delta_null_all_missing_101_to_101": bool((df["rank_delta"].isna() == missing_mask).all()),
        "rank_delta_mismatch_nonmissing": int(
            (~pd.isna(expected_delta) & ~np.isclose(df["rank_delta"], expected_delta)).sum()
        ),
        "movement_mismatch": int((df["movement"] != expected_movement).sum()),
        "stage1_bucket_mismatch": int((df["stage1_bucket"] != expected_bucket(df["stage1_rank"])).sum()),
        "rescored_bucket_mismatch": int((df["rescored_bucket"] != expected_bucket(df["rescored_rank"])).sum()),
        "top1_correct_mismatch_rescored_rank_eq_1": int(
            (df["top1_correct"].astype(bool) != df["rescored_rank"].eq(1)).sum()
        ),
        "top1_candidate_mismatch_ground_truth_when_correct": int(
            (df["top1_correct"].astype(bool) & df["top1_candidate"].ne(df["ground_truth"])).sum()
        ),
        "top10_gt_position_mismatch_rescored_rank": int(top10_position_mismatch.sum()),
    }
    validation.update(df.attrs.get("top10_validation", {}))
    return validation


def savefig(name: str) -> None:
    plt.tight_layout()
    path = PLOT_DIR / name
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def annotate_bars(ax, suffix: str = "%", decimals: int = 1) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height) or height <= 0:
            continue
        ax.annotate(
            f"{height:.{decimals}f}{suffix}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_performance(df: pd.DataFrame) -> None:
    rows = []
    for label, threshold in [("Top 1", 1), ("Top 5", 5), ("Top 10", 10), ("Top 50", 50), ("Top 100", 100)]:
        rows.append({"Metric": label, "Stage": "Stage 1", "Percent": (df["stage1_rank"].le(threshold)).mean() * 100})
        rows.append({"Metric": label, "Stage": "Rescored", "Percent": (df["rescored_rank"].le(threshold)).mean() * 100})
    perf = pd.DataFrame(rows)

    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=perf, x="Metric", y="Percent", hue="Stage", palette=PALETTE)
    ax.set_title("Top-k Retrieval Accuracy: Stage 1 vs Rescored")
    ax.set_xlabel("")
    ax.set_ylabel("Rows correct within k (%)")
    ax.set_ylim(0, 105)
    ax.legend(title="Stage", loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    annotate_bars(ax)
    sns.despine()
    savefig("01_topk_performance.png")


def plot_movement_and_transition(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [0.8, 1.2]})

    movement = (
        df["movement"]
        .value_counts()
        .reindex(MOVEMENT_ORDER)
        .rename_axis("Movement")
        .reset_index(name="Rows")
    )
    movement["Percent"] = movement["Rows"] / len(df) * 100
    ax = sns.barplot(data=movement, x="Movement", y="Percent", hue="Movement", palette=PALETTE, legend=False, ax=axes[0])
    ax.set_title("Outcome After Rescoring")
    ax.set_xlabel("")
    ax.set_ylabel("Rows (%)")
    ax.tick_params(axis="x", rotation=20)
    annotate_bars(ax)
    sns.despine(ax=ax)

    counts = pd.crosstab(df["stage1_bucket"], df["rescored_bucket"]).reindex(index=BUCKET_ORDER, columns=BUCKET_ORDER).fillna(0)
    row_pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    sns.heatmap(
        row_pct,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={"label": "Row percentage"},
        linewidths=0.5,
        linecolor="white",
        ax=axes[1],
    )
    axes[1].set_title("Rank Bucket Transition Matrix")
    axes[1].set_xlabel("Rescored bucket")
    axes[1].set_ylabel("Stage 1 bucket")
    savefig("02_movement_and_bucket_transition.png")


def plot_length_and_mods(df: pd.DataFrame) -> None:
    df = df.copy()
    df["gt_length"] = df["ground_truth"].map(peptide_length)
    df["n_mods_raw"] = df["ground_truth"].str.count(r"\(")
    df["n_mods"] = df["n_mods_raw"].clip(upper=4).astype(int).astype(str)
    df.loc[df["n_mods_raw"] >= 4, "n_mods"] = "4+"
    df["length_bin"] = pd.cut(
        df["gt_length"],
        bins=[0, 7, 9, 12, 15, 20, 10_000],
        labels=["<=7", "8-9", "10-12", "13-15", "16-20", ">20"],
        include_lowest=True,
    )

    length_summary = df.groupby("length_bin", observed=False).agg(
        Rows=("sample_idx", "size"),
        Stage_1=("stage1_rank", lambda s: s.le(1).mean() * 100),
        Rescored=("rescored_rank", lambda s: s.le(1).mean() * 100),
    )
    length_long = length_summary.reset_index().melt(
        id_vars=["length_bin", "Rows"],
        value_vars=["Stage_1", "Rescored"],
        var_name="Stage",
        value_name="Top1 Percent",
    )
    length_long["Stage"] = length_long["Stage"].str.replace("_", " ")

    mod_summary = df.groupby("n_mods", observed=False).agg(
        Rows=("sample_idx", "size"),
        Stage_1=("stage1_rank", lambda s: s.le(1).mean() * 100),
        Rescored=("rescored_rank", lambda s: s.le(1).mean() * 100),
    )
    mod_summary = mod_summary.reindex(["0", "1", "2", "3", "4+"])
    mod_long = mod_summary.reset_index().melt(
        id_vars=["n_mods", "Rows"],
        value_vars=["Stage_1", "Rescored"],
        var_name="Stage",
        value_name="Top1 Percent",
    )
    mod_long["Stage"] = mod_long["Stage"].str.replace("_", " ")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), sharey=True)
    sns.lineplot(data=length_long, x="length_bin", y="Top1 Percent", hue="Stage", marker="o", palette=PALETTE, ax=axes[0])
    axes[0].set_title("Top-1 Accuracy by Peptide Length")
    axes[0].set_xlabel("Ground-truth peptide length")
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_ylim(0, 100)
    for _, row in length_summary.reset_index().iterrows():
        axes[0].text(row.name, 3, f"n={int(row['Rows']):,}", ha="center", fontsize=8, rotation=90, color="#555555")

    sns.lineplot(data=mod_long, x="n_mods", y="Top1 Percent", hue="Stage", marker="o", palette=PALETTE, ax=axes[1])
    axes[1].set_title("Top-1 Accuracy by Number of Ground-truth Mods")
    axes[1].set_xlabel("Modification count")
    axes[1].set_ylabel("")
    axes[1].set_ylim(0, 100)
    for idx, row in enumerate(mod_summary.reset_index().itertuples(index=False)):
        if not pd.isna(row.Rows):
            axes[1].text(idx, 3, f"n={int(row.Rows):,}", ha="center", fontsize=8, rotation=90, color="#555555")
    sns.despine()
    savefig("03_length_and_modification_effects.png")


def plot_confidence(df: pd.DataFrame) -> None:
    df = df.copy()
    df["score_decile"] = pd.qcut(df["top1_score"], 10, labels=range(1, 11), duplicates="drop").astype(int)
    df["margin_decile"] = pd.qcut(df["top1_margin"], 10, labels=range(1, 11), duplicates="drop").astype(int)

    score_summary = df.groupby("score_decile", observed=False).agg(
        Correct=("top1_correct", lambda s: s.mean() * 100),
        MedianMargin=("top1_margin", "median"),
    ).reset_index()
    margin_summary = df.groupby("margin_decile", observed=False).agg(
        Correct=("top1_correct", lambda s: s.mean() * 100),
        MedianScore=("top1_score", "median"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    sns.lineplot(data=score_summary, x="score_decile", y="Correct", marker="o", color="#4C78A8", ax=axes[0])
    axes[0].set_title("Correctness by Top-1 Score Decile")
    axes[0].set_xlabel("Top-1 score decile (low to high)")
    axes[0].set_ylabel("Top-1 correct (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_xticks(range(1, 11))

    sns.lineplot(data=margin_summary, x="margin_decile", y="Correct", marker="o", color="#F58518", ax=axes[1])
    axes[1].set_title("Correctness by Top1-vs-Top2 Margin Decile")
    axes[1].set_xlabel("Margin decile (small to large)")
    axes[1].set_ylabel("")
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(range(1, 11))
    sns.despine()
    savefig("04_confidence_calibration.png")


def plot_wrong_top1_profile(df: pd.DataFrame) -> None:
    wrong = df.loc[~df["top1_correct"].astype(bool)].copy()
    wrong["gt_base"] = wrong["ground_truth"].map(base_sequence)
    wrong["top1_base"] = wrong["top1_candidate"].map(base_sequence)
    wrong["same_base_ignore_mods"] = wrong["gt_base"].eq(wrong["top1_base"])
    wrong["len_diff"] = wrong["top1_base"].str.len() - wrong["gt_base"].str.len()

    near_miss = pd.DataFrame(
        {
            "Category": [
                "Same base sequence\n(ignore mods)",
                "Same length",
                "Within 2 residues",
                "Other wrong top-1",
            ],
            "Rows": [
                wrong["same_base_ignore_mods"].sum(),
                wrong["len_diff"].eq(0).sum(),
                wrong["len_diff"].abs().le(2).sum(),
                len(wrong) - wrong["len_diff"].abs().le(2).sum(),
            ],
        }
    )
    near_miss["Percent of wrong top-1"] = near_miss["Rows"] / len(wrong) * 100

    top_candidates = (
        wrong.groupby("top1_candidate")
        .agg(Rows=("sample_idx", "size"), DistinctTruths=("ground_truth", "nunique"))
        .sort_values(["Rows", "DistinctTruths"], ascending=False)
        .head(12)
        .reset_index()
    )
    top_candidates["candidate_short"] = top_candidates["top1_candidate"].str.slice(0, 32)
    top_candidates.loc[top_candidates["top1_candidate"].str.len() > 32, "candidate_short"] += "..."

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [0.85, 1.15]})
    ax = sns.barplot(data=near_miss, x="Percent of wrong top-1", y="Category", color="#72B7B2", ax=axes[0])
    ax.set_title("Wrong Top-1 Near-miss Profile")
    ax.set_xlabel("Wrong top-1 rows (%)")
    ax.set_ylabel("")
    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(f"{width:.1f}%", (width, patch.get_y() + patch.get_height() / 2), va="center", ha="left", fontsize=8, xytext=(4, 0), textcoords="offset points")

    ax = sns.barplot(data=top_candidates, x="Rows", y="candidate_short", color="#E45756", ax=axes[1])
    ax.set_title("Most Recurrent Wrong Top-1 Candidates")
    ax.set_xlabel("Wrong top-1 rows")
    ax.set_ylabel("")
    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(f"{int(width)}", (width, patch.get_y() + patch.get_height() / 2), va="center", ha="left", fontsize=8, xytext=(4, 0), textcoords="offset points")
    sns.despine()
    savefig("05_wrong_top1_profile.png")


def plot_ood_failures() -> None:
    if not OOD_FAILURES.exists():
        return

    ood = pd.read_csv(OOD_FAILURES)
    if "len_bucket" in ood.columns:
        ood["len_bucket"] = ood["len_bucket"].astype(str).replace({"≤7": "<=7", "nan": "unknown", "NaN": "unknown"})

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), gridspec_kw={"width_ratios": [0.95, 0.85, 1.2]})
    failure_order = ood["failure_mode"].value_counts().index.tolist()
    failure_counts = ood["failure_mode"].value_counts().rename_axis("Failure mode").reset_index(name="Rows")
    failure_counts["Percent"] = failure_counts["Rows"] / len(ood) * 100
    ax = sns.barplot(data=failure_counts, x="Percent", y="Failure mode", order=failure_order, color="#72B7B2", ax=axes[0])
    ax.set_title("OOD Not-retrieved Failure Modes")
    ax.set_xlabel("Rows (%)")
    ax.set_ylabel("")
    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(f"{width:.1f}%", (width, patch.get_y() + patch.get_height() / 2), va="center", ha="left", fontsize=8, xytext=(4, 0), textcoords="offset points")

    len_order = ["<=7", "8-10", "11-15", "16-20", "21+", "unknown"]
    len_counts = ood["len_bucket"].value_counts().reindex(len_order).dropna().rename_axis("Length bucket").reset_index(name="Rows")
    sns.barplot(data=len_counts, x="Length bucket", y="Rows", color="#4C78A8", ax=axes[1])
    axes[1].set_title("OOD Failures by Length")
    axes[1].set_xlabel("Peptide length bucket")
    axes[1].set_ylabel("Rows")
    axes[1].tick_params(axis="x", rotation=20)

    sns.boxplot(data=ood, x="failure_mode", y="cos_sim", order=failure_order, color="#F58518", fliersize=2, ax=axes[2])
    axes[2].set_title("Spectrum-Truth Cosine by Failure Mode")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Cosine similarity")
    axes[2].tick_params(axis="x", rotation=25)
    sns.despine()
    savefig("06_ood_failure_profile.png")


def write_metrics(df: pd.DataFrame, validation: dict[str, object]) -> None:
    core_metrics = {
        "rows": len(df),
        "unique_ground_truth": int(df["ground_truth"].nunique()),
        "stage1_top1_pct": round(df["stage1_rank"].eq(1).mean() * 100, 4),
        "rescored_top1_pct": round(df["rescored_rank"].eq(1).mean() * 100, 4),
        "stage1_top5_pct": round(df["stage1_rank"].le(5).mean() * 100, 4),
        "rescored_top5_pct": round(df["rescored_rank"].le(5).mean() * 100, 4),
        "stage1_missing_pct": round(df["stage1_rank"].eq(101).mean() * 100, 4),
        "rescored_missing_pct": round(df["rescored_rank"].eq(101).mean() * 100, 4),
        "improved_count": int(df["movement"].eq("improved").sum()),
        "worsened_count": int(df["movement"].eq("worsened").sum()),
        "unchanged_count": int(df["movement"].eq("unchanged").sum()),
        "still_missing_count": int(df["movement"].eq("still_missing").sum()),
        "rescued_to_top1_count": int((df["stage1_rank"].ne(1) & df["rescored_rank"].eq(1)).sum()),
        "lost_from_top1_count": int((df["stage1_rank"].eq(1) & df["rescored_rank"].ne(1)).sum()),
    }
    with (PLOT_DIR / "plot_validation_and_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"validation": validation, "core_metrics": core_metrics}, f, indent=2)


def main() -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.78)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    df = pd.read_csv(FULL_ANALYSIS)
    df = parse_top10_margin(df)
    validation = validate(df)
    write_metrics(df, validation)

    failed_checks = {
        key: value
        for key, value in validation.items()
        if (
            key.endswith("_mismatch")
            or key.endswith("_mismatch_nonmissing")
            or key.endswith("_bad_items_or_indices")
            or key.endswith("_bad_item_pairs")
            or key.endswith("_not_10_items")
        )
        and value != 0
    }
    if failed_checks:
        raise ValueError(f"Validation failed; refusing to plot from inconsistent data: {failed_checks}")

    plot_performance(df)
    plot_movement_and_transition(df)
    plot_length_and_mods(df)
    plot_confidence(df)
    plot_wrong_top1_profile(df)
    plot_ood_failures()

    print(f"Wrote seaborn plots to {PLOT_DIR}")
    print(json.dumps({"validation": validation}, indent=2))


if __name__ == "__main__":
    main()
