from __future__ import annotations

import ast
from typing import Dict, List

import pandas as pd

from .material import card_rank, point_value, select_contributing_points

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
POINT_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T"]
FACE_RANKS = ["J", "Q", "K", "8"]


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


def build_learned_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        feats: Dict[str, float] = {}
        for player in (0, 1):
            suffix = "" if player == 0 else "_opp"
            hand = row["p0Hand"] if player == 0 else row["p1Hand"]
            points = row["p0Points"] if player == 0 else row["p1Points"]
            faces = row["p0FaceCards"] if player == 0 else row["p1FaceCards"]
            jacked = row.get("p0JackedCards", []) if player == 0 else row.get("p1JackedCards", [])

            point_counts = _count_by_rank(points, POINT_RANKS)
            face_counts = _count_by_rank(faces, FACE_RANKS)
            _apply_jacked(jacked, point_counts, face_counts)

            kings = face_counts.get("K", 0)
            queens_capped = min(face_counts.get("Q", 0), 2)
            glasses_capped = min(face_counts.get("8", 0), 1)
            contributing_points, cards_needed = select_contributing_points(points, kings)
            contrib_counts = _count_by_rank(contributing_points, POINT_RANKS)
            redundant_points = max(len(points) - len(contributing_points), 0)
            board_material = len(contributing_points) + kings + queens_capped + glasses_capped

            for r in RANKS:
                feats[f"{suffix}hand_{r}"] = float(_count_by_rank(hand, RANKS)[r])
            feats[f"{suffix}contrib_points"] = float(len(contributing_points))
            feats[f"{suffix}points_total"] = float(sum(point_value(c) for c in contributing_points))
            feats[f"{suffix}cards_needed"] = float(cards_needed)
            feats[f"{suffix}redundant_points"] = float(redundant_points)
            feats[f"{suffix}face_K"] = float(kings)
            feats[f"{suffix}face_Q"] = float(queens_capped)
            feats[f"{suffix}face_8"] = float(glasses_capped)
            feats[f"{suffix}board_material"] = float(board_material)
            feats[f"{suffix}has_glasses"] = 1.0 if glasses_capped > 0 else 0.0
        feats["is_turn"] = 1.0 if row["turn"] == 0 else 0.0
        feats["winner"] = row["winner"]
        rows.append(feats)
    return pd.DataFrame(rows)
