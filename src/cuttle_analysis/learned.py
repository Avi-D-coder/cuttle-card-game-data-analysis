from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .material import card_rank, point_value, select_contributing_points

POINT_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T"]
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


def load_learned_weights(path: str | Path) -> Dict[str, float]:
    data = json.loads(Path(path).read_text())
    weights = {}
    if "weights" in data and isinstance(data["weights"], list):
        for item in data["weights"]:
            weights[item["feature"]] = item["weight"]
    elif "weights" in data and isinstance(data["weights"], dict):
        weights.update(data["weights"])
    if "intercept" in data:
        weights["_intercept"] = data["intercept"]
    if "_intercept" not in weights:
        weights["_intercept"] = 0.0
    return weights


def _count_by_rank(cards: List[str], ranks: List[str]) -> Dict[str, int]:
    counts = {r: 0 for r in ranks}
    for card in cards:
        r = card_rank(card)
        if r in counts:
            counts[r] += 1
    return counts


def _apply_jacked(cards: List[str], point_counts: Dict[str, int], face_counts: Dict[str, int]) -> None:
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


def _board_features(points: List[str], faces: List[str], jacked: List[str]) -> Dict[str, float]:
    point_counts = _count_by_rank(points, POINT_RANKS)
    face_counts = _count_by_rank(faces, ["J", "Q", "K", "8"])
    _apply_jacked(jacked, point_counts, face_counts)
    kings = face_counts.get("K", 0)
    queens_capped = min(face_counts.get("Q", 0), 2)
    glasses_capped = min(face_counts.get("8", 0), 1)
    contributing_points, cards_needed = select_contributing_points(points, kings)
    redundant_points = max(len(points) - len(contributing_points), 0)
    board_material = len(contributing_points) + kings + queens_capped + glasses_capped
    return {
        "contrib_points": float(len(contributing_points)),
        "points_total": float(sum(point_value(c) for c in contributing_points)),
        "cards_needed": float(cards_needed),
        "redundant_points": float(redundant_points),
        "face_K": float(kings),
        "face_Q": float(queens_capped),
        "face_8": float(glasses_capped),
        "board_material": float(board_material),
        "has_glasses": 1.0 if glasses_capped > 0 else 0.0,
    }


def _hand_counts(hand: List[str]) -> Dict[str, float]:
    counts = _count_by_rank(hand, RANKS)
    return {f"hand_{r}": float(counts[r]) for r in RANKS}


def learned_features_for_row(row: pd.Series) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    # Self
    feats.update(_hand_counts(row["p0Hand"]))
    feats.update(_board_features(row["p0Points"], row["p0FaceCards"], row.get("p0JackedCards", [])))
    # Opponent public board
    opp_board = _board_features(row["p1Points"], row["p1FaceCards"], row.get("p1JackedCards", []))
    feats["opp_board_material"] = opp_board["board_material"]
    feats["opp_face_K"] = opp_board["face_K"]
    feats["opp_face_Q"] = opp_board["face_Q"]
    feats["opp_face_8"] = opp_board["face_8"]
    feats["opp_contrib_points"] = opp_board["contrib_points"]
    feats["opp_cards_needed"] = opp_board["cards_needed"]
    feats["opp_redundant_points"] = opp_board["redundant_points"]
    feats["opp_has_glasses"] = opp_board["has_glasses"]
    feats["is_turn"] = 1.0 if row["turn"] == 0 else 0.0
    return feats


def apply_learned_scores(df: pd.DataFrame, weights_path: str | Path) -> pd.DataFrame:
    weights = load_learned_weights(weights_path)
    intercept = float(weights.pop("_intercept", 0.0))
    rows = []
    for _, row in df.iterrows():
        feats = learned_features_for_row(row)
        score = intercept + sum(weights.get(k, 0.0) * feats.get(k, 0.0) for k in weights.keys())
        rows.append(score)
    out = df.copy()
    out["learned_score"] = np.array(rows, dtype=float)
    out["material_diff"] = out["learned_score"]
    return out
