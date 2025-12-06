from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.cuttle_analysis.material import card_rank, point_value, select_contributing_points

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
POINT_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T"]
FACE_RANKS = ["J", "Q", "K", "8"]


@dataclass
class PerspectiveSpec:
    include_scrap: bool = False
    include_deck_size: bool = False


def _count_by_rank(cards: List[str], ranks: List[str]) -> Dict[str, int]:
    counts = {r: 0 for r in ranks}
    for card in cards:
        r = card_rank(card)
        if r in counts:
            counts[r] += 1
    return counts


def _apply_jacked(cards: List[str], point_counts: Dict[str, int], face_counts: Dict[str, int]) -> None:
    """
    Fold jacked tokens (e.g., 'JJ9C') into point/face counts.

    Only odd-depth stacks contribute (JJ cancels), so depth_parity = len(J-prefix) % 2.
    """
    for raw in cards:
        if not raw:
            continue
        s = str(raw)
        depth = len(s) - len(s.lstrip("J"))
        if depth % 2 == 0:
            continue
        base = s[depth:]
        r = card_rank(base)
        if r in point_counts:
            point_counts[r] += 1
        elif r in face_counts:
            face_counts[r] += 1


def _cards_needed_to_win(point_counts: Dict[str, int], face_counts: Dict[str, int]) -> int:
    kings = face_counts.get("K", 0)
    target = max(21 - 7 * kings, 0)
    points_total = sum(point_value(r) * count for r, count in point_counts.items())
    remaining = max(target - points_total, 0)
    if remaining == 0:
        return 0
    return int(np.ceil(remaining / 10))


def _board_features(points: List[str], faces: List[str], jacked: List[str]) -> Dict[str, float]:
    # Apply jacked control to point/face counts
    point_counts = _count_by_rank(points, POINT_RANKS)
    face_counts = _count_by_rank(faces, FACE_RANKS)
    _apply_jacked(jacked, point_counts, face_counts)

    # Caps and exclusion
    kings = face_counts.get("K", 0)
    queens_capped = min(face_counts.get("Q", 0), 2)
    glasses_capped = min(face_counts.get("8", 0), 1)
    # ignore jacks as persistent

    contributing_points, cards_needed = select_contributing_points(
        [p for p in points], kings
    )
    contrib_counts = _count_by_rank(contributing_points, POINT_RANKS)
    redundant_points = max(len(points) - len(contributing_points), 0)
    board_material = len(contributing_points) + kings + queens_capped + glasses_capped

    feats: Dict[str, float] = {}
    feats["contrib_points"] = float(len(contributing_points))
    feats["points_total"] = float(sum(point_value(c) for c in contributing_points))
    feats["cards_needed"] = float(cards_needed)
    feats["redundant_points"] = float(redundant_points)
    feats["face_K"] = float(kings)
    feats["face_Q"] = float(queens_capped)
    feats["face_8"] = float(glasses_capped)
    feats["board_material"] = float(board_material)
    return feats


def _self_block(
    hand: List[str],
    points: List[str],
    faces: List[str],
    jacked: List[str],
) -> Dict[str, float]:
    hand_counts = _count_by_rank(hand, RANKS)
    feats: Dict[str, float] = {}
    for r in RANKS:
        feats[f"hand_{r}"] = float(hand_counts.get(r, 0))
    feats.update(_board_features(points, faces, jacked))
    feats["has_glasses"] = 1.0 if feats.get("face_8", 0.0) > 0 else 0.0
    return feats


def _scrap_block(scrap: List[str]) -> Dict[str, float]:
    counts = _count_by_rank(scrap, RANKS)
    return {f"scrap_{r}": float(counts[r]) for r in RANKS}


def perspective_features(row: pd.Series, player: int, spec: PerspectiveSpec) -> Dict[str, float]:
    """
    Features for `player`'s perspective at this state.

    Player 0 or 1. Target label = 1 if player == winner else 0.
    """
    if player == 0:
        hand, points, faces, jacked = row["p0Hand"], row["p0Points"], row["p0FaceCards"], row.get("p0JackedCards", [])
        opp_points, opp_faces, opp_jacked = row["p1Points"], row["p1FaceCards"], row.get("p1JackedCards", [])
    else:
        hand, points, faces, jacked = row["p1Hand"], row["p1Points"], row["p1FaceCards"], row.get("p1JackedCards", [])
        opp_points, opp_faces, opp_jacked = row["p0Points"], row["p0FaceCards"], row.get("p0JackedCards", [])

    feats: Dict[str, float] = {}
    feats.update(_self_block(hand, points, faces, jacked))
    # Opponent public board (aggregate only)
    opp_board = _board_features(opp_points, opp_faces, opp_jacked)
    feats["opp_board_material"] = opp_board["board_material"]
    feats["opp_face_K"] = opp_board["face_K"]
    feats["opp_face_Q"] = opp_board["face_Q"]
    feats["opp_face_8"] = opp_board["face_8"]
    feats["opp_contrib_points"] = opp_board["contrib_points"]
    feats["opp_cards_needed"] = opp_board["cards_needed"]
    feats["opp_redundant_points"] = opp_board["redundant_points"]
    feats["opp_has_glasses"] = 1.0 if opp_board.get("face_8", 0.0) > 0 else 0.0
    feats["is_turn"] = 1.0 if row["turn"] == player else 0.0
    return feats


def build_perspective_matrix(df: pd.DataFrame, spec: PerspectiveSpec) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build per-player samples (two per state) with labels indicating whether that player wins.
    """
    samples: List[np.ndarray] = []
    labels: List[int] = []
    # Infer feature names from the first row
    dummy = perspective_features(df.iloc[0], 0, spec)
    feature_names = list(dummy.keys())
    for _, row in df.iterrows():
        winner = row["winner"]
        if winner not in (0, 1):
            continue
        for player in (0, 1):
            feats = perspective_features(row, player, spec)
            samples.append(np.array([feats[k] for k in feature_names], dtype=float))
            labels.append(1 if player == winner else 0)
    X = np.vstack(samples) if samples else np.empty((0, len(feature_names)))
    y = np.array(labels, dtype=int)
    return X, y, feature_names
