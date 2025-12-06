import os
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResults


# Use a workspace-local matplotlib cache to avoid permission issues.
MPL_CACHE = Path(".matplotlib_cache")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
MPL_CACHE.mkdir(parents=True, exist_ok=True)


def _fit_logit(df: pd.DataFrame) -> Optional[BinaryResults]:
    clean = df[df["winner"].isin([0, 1])].copy()
    if clean.empty:
        return None
    clean["p0_win"] = (clean["winner"] == 0).astype(int)
    X = sm.add_constant(clean["material_diff"])
    model = sm.Logit(clean["p0_win"], X)
    try:
        result = model.fit(disp=False)
    except Exception:
        return None
    return result


def plot_logit_win_prob(df: pd.DataFrame, output: Path) -> None:
    """Plot logistic win probability vs material diff (overall)."""
    result = _fit_logit(df)
    if result is None:
        return
    xs = np.linspace(df["material_diff"].quantile(0.01), df["material_diff"].quantile(0.99), 200)
    preds = result.predict(sm.add_constant(xs))

    # Empirical bins
    bins = np.linspace(df["material_diff"].min(), df["material_diff"].max(), 20)
    df = df.copy()
    df["bin"] = pd.cut(df["material_diff"], bins=bins)
    grouped = df[df["winner"].isin([0, 1])].groupby("bin", observed=True)["winner"]
    empirical = grouped.apply(lambda s: (s == 0).mean())
    centers = [interval.mid for interval in empirical.index]

    plt.figure(figsize=(7, 5))
    sns.lineplot(x=xs, y=preds, label="Logit fit")
    sns.scatterplot(x=centers, y=empirical.values, color="black", label="Empirical")
    plt.xlabel("Material diff (p0 - p1)")
    plt.ylabel("P(p0 win)")
    plt.title("Win probability vs material diff")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_logit_by_deck_bucket(df: pd.DataFrame, output: Path, deck_buckets: Iterable[str]) -> None:
    """Plot logistic win probability curves per deck bucket."""
    plt.figure(figsize=(8, 5))
    for bucket in deck_buckets:
        subset = df[df["deck_bucket"] == bucket]
        result = _fit_logit(subset)
        if result is None:
            continue
        xs = np.linspace(
            subset["material_diff"].quantile(0.05),
            subset["material_diff"].quantile(0.95),
            150,
        )
        preds = result.predict(sm.add_constant(xs))
        sns.lineplot(x=xs, y=preds, label=f"{bucket}")
    plt.xlabel("Material diff (p0 - p1)")
    plt.ylabel("P(p0 win)")
    plt.title("Win probability by deck depth")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_flip_by_diff(df: pd.DataFrame, bins: np.ndarray, output: Path) -> None:
    """Plot probability of future material lead flip vs current diff (overall)."""
    df = df.copy()
    df["diff_bin"] = pd.cut(df["material_diff"], bins=bins, include_lowest=True)
    agg = df.groupby("diff_bin", observed=True)["future_flip"].mean().reset_index()
    centers = [interval.mid for interval in agg["diff_bin"]]
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=centers, y=agg["future_flip"], marker="o")
    plt.xlabel("Material diff (p0 - p1)")
    plt.ylabel("P(future lead flip)")
    plt.title("Lead flip probability vs material diff")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_flip_by_deck_and_diff(
    df: pd.DataFrame,
    bins: np.ndarray,
    output: Path,
    bucket_legend: list[tuple[str, str]],
) -> None:
    """Flip probability vs diff, stratified by deck bucket with legend text."""
    df = df.copy()
    df["diff_bin"] = pd.cut(df["material_diff"], bins=bins, include_lowest=True)
    agg = (
        df.groupby(["deck_bucket", "diff_bin"], observed=True)["future_flip"]
        .mean()
        .reset_index()
    )
    centers = {interval: interval.mid for interval in agg["diff_bin"].unique()}
    plt.figure(figsize=(8, 5))
    for label, desc in bucket_legend:
        subset = agg[agg["deck_bucket"] == label]
        if subset.empty:
            continue
        xs = subset["diff_bin"].map(centers)
        sns.lineplot(
            x=xs,
            y=subset["future_flip"],
            marker="o",
            label=f"{label} ({desc})",
        )
    plt.xlabel("Material diff (p0 - p1)")
    plt.ylabel("P(future lead flip)")
    plt.title("Lead flip probability by deck depth")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.2)
    plt.legend(title="Deck buckets")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_win_rate_by_deck_and_diff(
    df: pd.DataFrame,
    bins: np.ndarray,
    output: Path,
    bucket_legend: list[tuple[str, str]],
) -> None:
    """Empirical win rate vs diff per deck bucket (binned)."""
    df = df.copy()
    df = df[df["winner"].isin([0, 1])]
    df["diff_bin"] = pd.cut(df["material_diff"], bins=bins, include_lowest=True)
    agg = (
        df.groupby(["deck_bucket", "diff_bin"], observed=True)["winner"]
        .apply(lambda s: (s == 0).mean())
        .reset_index(name="p0_win_rate")
    )
    centers = {interval: interval.mid for interval in agg["diff_bin"].unique()}
    plt.figure(figsize=(8, 5))
    for label, desc in bucket_legend:
        subset = agg[agg["deck_bucket"] == label]
        if subset.empty:
            continue
        xs = subset["diff_bin"].map(centers)
        sns.lineplot(
            x=xs,
            y=subset["p0_win_rate"],
            marker="o",
            label=f"{label} ({desc})",
        )
    plt.xlabel("Material diff (p0 - p1)")
    plt.ylabel("P(p0 win)")
    plt.title("Win rate vs material diff by deck depth")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.2)
    plt.legend(title="Deck buckets")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_win_rate_by_diff(df: pd.DataFrame, bins: np.ndarray, output: Path) -> None:
    """Empirical win rate vs material diff (binned, overall)."""
    df = df.copy()
    df = df[df["winner"].isin([0, 1])]
    df["diff_bin"] = pd.cut(df["material_diff"], bins=bins, include_lowest=True)
    agg = (
        df.groupby("diff_bin", observed=True)["winner"]
        .apply(lambda s: (s == 0).mean())
        .reset_index(name="p0_win_rate")
    )
    centers = [interval.mid for interval in agg["diff_bin"]]
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=centers, y=agg["p0_win_rate"], marker="o", color="teal")
    plt.xlabel("Material diff (p0 - p1)")
    plt.ylabel("P(p0 win)")
    plt.title("Win rate vs material diff")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_lead_timing(lead_table: pd.DataFrame, output: Path) -> None:
    """Bar plot for decisive lead hit and stability by margin."""
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x="margin",
        y="hit_rate",
        data=lead_table,
        color="steelblue",
        label="Hit rate",
    )
    sns.barplot(
        x="margin",
        y="stable_hit_rate",
        data=lead_table,
        color="darkorange",
        alpha=0.7,
        label="Stable hit rate",
    )
    plt.xlabel("Material margin (eventual winner)")
    plt.ylabel("Fraction of games")
    plt.title("First decisive lead: hit vs stay")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.2, axis="y")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_move_deltas(move_table: pd.DataFrame, output: Path) -> None:
    """Horizontal bar plot of avg material diff delta by move type."""
    move_table = move_table.sort_values("avg_delta")
    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="avg_delta",
        y="moveType",
        data=move_table,
        palette="coolwarm",
        orient="h",
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Avg material diff delta")
    plt.ylabel("Move type")
    plt.title("Material change by move type")
    plt.grid(True, alpha=0.2, axis="x")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_first_action_win_rates(
    table: pd.DataFrame, title: str, output: Path
) -> None:
    """Bar plot for first-action win rates (first mover, points, kings)."""
    plt.figure(figsize=(5, 4))
    sns.barplot(
        x="first_player",
        y="win_rate_first_player",
        data=table,
        palette="Set2",
    )
    plt.ylim(0, 1)
    plt.xlabel("First player to act (0=p0, 1=p1)")
    plt.ylabel("Win rate for that player")
    plt.title(title)
    plt.grid(True, alpha=0.2, axis="y")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_win_rate_heatmap(
    df: pd.DataFrame,
    diff_bins: np.ndarray,
    deck_bins: np.ndarray,
    output: Path,
    title: str = "Win rate heatmap (p0 win)",
) -> None:
    """2D heatmap of p0 win rate by material diff and deck size bins."""
    df = df.copy()
    df = df[df["winner"].isin([0, 1])]
    df["diff_bin"] = pd.cut(df["material_diff"], bins=diff_bins, include_lowest=True)
    df["deck_bin"] = pd.cut(df["deck_size"], bins=deck_bins, include_lowest=True)
    pivot = (
        df.groupby(["deck_bin", "diff_bin"], observed=True)["winner"]
        .apply(lambda s: (s == 0).mean())
        .unstack()
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=False,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "P(p0 win)"},
    )
    plt.xlabel("Material diff bin")
    plt.ylabel("Deck size bin")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_flip_rate_heatmap(
    df: pd.DataFrame,
    diff_bins: np.ndarray,
    deck_bins: np.ndarray,
    output: Path,
    title: str = "Lead flip probability heatmap",
) -> None:
    """2D heatmap of future flip probability by material diff and deck size bins."""
    df = df.copy()
    df["diff_bin"] = pd.cut(df["material_diff"], bins=diff_bins, include_lowest=True)
    df["deck_bin"] = pd.cut(df["deck_size"], bins=deck_bins, include_lowest=True)
    pivot = (
        df.groupby(["deck_bin", "diff_bin"], observed=True)["future_flip"]
        .mean()
        .unstack()
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=False,
        cmap="magma_r",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "P(future lead flip)"},
    )
    plt.xlabel("Material diff bin")
    plt.ylabel("Deck size bin")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
