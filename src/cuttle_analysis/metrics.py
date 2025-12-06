from typing import Iterable, List

import numpy as np
import pandas as pd
from .material import card_rank


def add_leader_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate leader sign (+1 p0, -1 p1, 0 tie) and winner sign."""
    labeled = df.copy()
    labeled["leader_sign"] = np.sign(labeled["material_diff"]).astype(int)
    labeled["winner_sign"] = labeled["winner"].map({0: 1, 1: -1})
    labeled["leader_matches_winner"] = labeled["winner_sign"].notna() & (
        labeled["leader_sign"] == labeled["winner_sign"]
    )
    return labeled


def add_future_flip_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each state, mark whether the material leader flips later in the game.

    flip = True when a future state has material_diff with opposite sign.
    For diff == 0 (no leader), flip is True if any future state has a non-zero leader.
    """
    frames: List[pd.DataFrame] = []
    for _, group in df.sort_values(["gameId_hashed", "move_number"]).groupby(
        "gameId_hashed", sort=False
    ):
        signs = np.sign(group["material_diff"].to_numpy())
        future_flip = np.zeros_like(signs, dtype=bool)
        future_pos = False
        future_neg = False

        for idx in range(len(group) - 1, -1, -1):
            s = signs[idx]
            if s > 0 and future_neg:
                future_flip[idx] = True
            elif s < 0 and future_pos:
                future_flip[idx] = True
            elif s == 0 and (future_pos or future_neg):
                future_flip[idx] = True
            future_pos = future_pos or s > 0
            future_neg = future_neg or s < 0

        annotated = group.copy()
        annotated["future_flip"] = future_flip
        frames.append(annotated)

    return pd.concat(frames, ignore_index=True)


def win_rate_by_diff(
    df: pd.DataFrame, bins: Iterable[float]
) -> pd.DataFrame:
    """
    Aggregate win/stalemate rates by material_diff bins.
    """
    working = add_leader_labels(df)
    working["diff_bin"] = pd.cut(
        working["material_diff"], bins=bins, include_lowest=True
    )

    def rate(series: pd.Series, target) -> float:
        valid = series.dropna()
        if valid.empty:
            return float("nan")
        return (valid == target).mean()

    grouped = working.groupby("diff_bin", observed=True)
    summary = grouped.apply(
        lambda g: pd.Series(
            {
                "states": len(g),
                "p0_win_rate": rate(g["winner"], 0),
                "p1_win_rate": rate(g["winner"], 1),
                "stalemate_rate": rate(g["winner"], np.nan),
                "leader_win_rate": g["leader_matches_winner"].mean(),
                "avg_moves_to_end": g["moves_to_end"].mean(),
            }
        ),
        include_groups=False,
    ).reset_index()
    return summary


def flip_rate_by_diff(df: pd.DataFrame, bins: Iterable[float]) -> pd.DataFrame:
    """Probability of a future material lead flip, bucketed by current diff."""
    working = add_future_flip_flag(df)
    working["diff_bin"] = pd.cut(
        working["material_diff"], bins=bins, include_lowest=True
    )
    grouped = working.groupby("diff_bin", observed=True)
    summary = grouped["future_flip"].agg(states="count", flip_rate="mean").reset_index()
    return summary


def stratify_by_deck_depth(
    df: pd.DataFrame, quantiles: int = 5
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """
    Add deck depth buckets (qcut on deck_size) and return legend info.

    Returns (labeled_df, legend) where legend is [(bucket_label, "low-high cards left"), ...]
    """
    labeled = df.copy()
    codes, bins = pd.qcut(
        labeled["deck_size"], q=quantiles, duplicates="drop", retbins=True, labels=False
    )
    bucket_count = len(bins) - 1
    labels = [f"Q{i+1}" for i in range(bucket_count)]
    labeled["deck_bucket"] = codes.map(dict(enumerate(labels)))
    # Human-readable legend showing card ranges
    legend: list[tuple[str, str]] = []
    for i, label in enumerate(labels):
        low = int(bins[i])
        high = int(bins[i + 1])
        legend.append((label, f"{low}-{high} cards left"))
    labeled["deck_bucket_label"] = labeled["deck_bucket"].map(
        dict(legend)
    )
    return labeled, legend


def leader_win_rate_by_deck_and_diff(
    df: pd.DataFrame, diff_bins: Iterable[float], deck_quantiles: int = 5
) -> pd.DataFrame:
    """Leader-win and flip rates stratified by deck depth."""
    stratified, _ = stratify_by_deck_depth(
        add_future_flip_flag(add_leader_labels(df)), deck_quantiles
    )
    stratified["diff_bin"] = pd.cut(
        stratified["material_diff"], bins=diff_bins, include_lowest=True
    )
    grouped = stratified.groupby(
        ["deck_bucket", "diff_bin"], observed=True, dropna=False
    )
    summary = grouped.agg(
        states=("gameId_hashed", "count"),
        leader_win_rate=("leader_matches_winner", "mean"),
        flip_rate=("future_flip", "mean"),
        avg_moves_to_end=("moves_to_end", "mean"),
    ).reset_index()
    return summary


def win_rate_by_diff_and_deck(
    df: pd.DataFrame, diff_bins: Iterable[float], deck_quantiles: int = 5
) -> pd.DataFrame:
    """Win/stalemate rates stratified by deck depth and material diff."""
    stratified, _ = stratify_by_deck_depth(add_leader_labels(df), deck_quantiles)
    stratified["diff_bin"] = pd.cut(
        stratified["material_diff"], bins=diff_bins, include_lowest=True
    )
    grouped = stratified.groupby(
        ["deck_bucket", "diff_bin"], observed=True, dropna=False
    )

    def rate(series: pd.Series, target) -> float:
        valid = series.dropna()
        if valid.empty:
            return float("nan")
        return (valid == target).mean()

    summary = grouped.apply(
        lambda g: pd.Series(
            {
                "states": len(g),
                "p0_win_rate": rate(g["winner"], 0),
                "p1_win_rate": rate(g["winner"], 1),
                "stalemate_rate": rate(g["winner"], np.nan),
                "leader_win_rate": g["leader_matches_winner"].mean(),
                "avg_moves_to_end": g["moves_to_end"].mean(),
            }
        ),
        include_groups=False,
    ).reset_index()
    return summary


def decisive_lead_summary(
    df: pd.DataFrame, margins: Iterable[float]
) -> pd.DataFrame:
    """
    For each game, find the first time the eventual winner leads by >= margin.

    Returns per-margin aggregates on timing and stability (whether the lead ever flips).
    """
    records = []
    # Ensure future_flip is present for stability checks
    annotated = add_future_flip_flag(df)
    for margin in margins:
        hits = 0
        stable_hits = 0
        move_sum = 0
        deck_sum = 0
        moves_to_end_sum = 0
        total_games = 0
        for _, group in annotated.sort_values("move_number").groupby("gameId_hashed"):
            total_games += 1
            if group["winner"].iloc[-1] not in (0, 1):
                continue
            winner_sign = 1 if group["winner"].iloc[-1] == 0 else -1
            signed_diff = group["material_diff"] * winner_sign
            meet = signed_diff >= margin
            if not meet.any():
                continue
            first_idx = meet.idxmax()  # index of first True
            row = group.loc[first_idx]
            hits += 1
            move_sum += row["move_number"]
            deck_sum += row["deck_size"]
            moves_to_end_sum += row["moves_to_end"]
            stable = not (group.loc[first_idx:, "future_flip"].any())
            stable_hits += int(stable)
        records.append(
            {
                "margin": margin,
                "games": total_games,
                "hit_rate": hits / total_games if total_games else float("nan"),
                "stable_hit_rate": stable_hits / total_games if total_games else float("nan"),
                "avg_move_number": move_sum / hits if hits else float("nan"),
                "avg_deck_size": deck_sum / hits if hits else float("nan"),
                "avg_moves_to_end": moves_to_end_sum / hits if hits else float("nan"),
            }
        )
    return pd.DataFrame.from_records(records)


def move_type_material_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute change in material_diff per moveType.
    """
    frames = []
    for _, group in df.sort_values(["gameId_hashed", "move_number"]).groupby(
        "gameId_hashed", sort=False
    ):
        diff = group["material_diff"].to_numpy()
        delta = np.diff(diff, prepend=diff[0])
        g = group.copy()
        g["material_diff_delta"] = delta
        frames.append(g)
    enriched = pd.concat(frames, ignore_index=True)
    return (
        enriched.groupby("moveType", dropna=False)
        .agg(
            moves=("gameId_hashed", "count"),
            avg_delta=("material_diff_delta", "mean"),
            median_delta=("material_diff_delta", "median"),
            p90_delta=("material_diff_delta", lambda x: np.nanpercentile(x, 90)),
            p10_delta=("material_diff_delta", lambda x: np.nanpercentile(x, 10)),
        )
        .reset_index()
    )


def _game_groups(df: pd.DataFrame):
    return df.sort_values("move_number").groupby("gameId_hashed", sort=False)


def first_mover_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Win rates by which player acted first in the game (playedBy of first row).
    """
    records = []
    for _, group in _game_groups(df):
        first_player = group.iloc[0]["playedBy"]
        winner = group.iloc[-1]["winner"]
        records.append({"first_player": first_player, "winner": winner})
    frame = pd.DataFrame(records)
    result = []
    for player in sorted(frame["first_player"].unique()):
        subset = frame[frame["first_player"] == player]
        result.append(
            {
                "first_player": player,
                "games": len(subset),
                "win_rate_first_player": (subset["winner"] == player).mean(),
                "p0_win_rate": (subset["winner"] == 0).mean(),
                "p1_win_rate": (subset["winner"] == 1).mean(),
            }
        )
    return pd.DataFrame(result)


def first_to_action_win_rates(df: pd.DataFrame, action_filter) -> pd.DataFrame:
    """
    Generic helper: win rates by who first performs an action (points or king).
    action_filter: function(df_group) -> boolean Series mask
    """
    records = []
    for _, group in _game_groups(df):
        winner = group.iloc[-1]["winner"]
        first_rows = group[action_filter(group)]
        if first_rows.empty:
            continue
        first_player = first_rows.iloc[0]["playedBy"]
        records.append({"first_player": first_player, "winner": winner})
    frame = pd.DataFrame(records)
    result = []
    for player in sorted(frame["first_player"].unique()):
        subset = frame[frame["first_player"] == player]
        result.append(
            {
                "first_player": player,
                "games": len(subset),
                "win_rate_first_player": (subset["winner"] == player).mean(),
                "p0_win_rate": (subset["winner"] == 0).mean(),
                "p1_win_rate": (subset["winner"] == 1).mean(),
            }
        )
    return pd.DataFrame(result)


def first_to_points_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Win rates by who first plays points in a game."""
    return first_to_action_win_rates(df, lambda g: g["moveType"] == "points")


def first_to_kings_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Win rates by who first plays a king as a face card."""
    return first_to_action_win_rates(
        df,
        lambda g: (g["moveType"] == "faceCard")
        & (g["playedCard"].fillna("").apply(card_rank) == "K"),
    )
