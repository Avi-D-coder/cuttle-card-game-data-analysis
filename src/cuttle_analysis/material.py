import math
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

POINT_VALUES = {
    "A": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
}


def card_rank(card: str) -> str:
    return (card or "").strip()[:1]


def point_value(card: str) -> int:
    return POINT_VALUES.get(card_rank(card), 0)


def target_points(kings_count: int) -> int:
    """Return point threshold after king reductions."""
    return max(21 - 7 * kings_count, 0)


def cards_needed_to_win(current_points: int, kings_count: int) -> int:
    """Minimum additional point cards (value 10 max each) needed to win."""
    threshold = target_points(kings_count)
    remaining = max(threshold - current_points, 0)
    return math.ceil(remaining / 10)


def select_contributing_points(
    point_cards: Sequence[str], kings_count: int
) -> Tuple[List[str], int]:
    """
    Identify point cards that actually reduce the number of cards needed to win.

    We keep the smallest set of point cards (by descending value) required to
    preserve the current "cards needed to win" count. Points beyond that set
    are treated as non-contributing.
    """
    values = [point_value(c) for c in point_cards]
    threshold = target_points(kings_count)
    total_points = sum(values)
    cards_needed = cards_needed_to_win(total_points, kings_count)
    contributing_indices: List[int] = []

    # Points required to stay at the current card-count threshold.
    required_points = max(threshold - 10 * cards_needed, 0)
    if required_points > 0:
        running = 0
        for idx in sorted(range(len(values)), key=lambda i: values[i], reverse=True):
            if running >= required_points:
                break
            running += values[idx]
            contributing_indices.append(idx)

    return [point_cards[i] for i in contributing_indices], cards_needed


def facecard_board_count(face_cards: Iterable[str]) -> Tuple[int, Dict[str, int]]:
    """Count board-contributing face cards with glasses/queen caps and jack exclusion."""
    kings = queens = glasses = jacks = 0
    others = 0
    for card in face_cards:
        rank = card_rank(card)
        if rank == "K":
            kings += 1
        elif rank == "Q":
            queens += 1
        elif rank == "8":
            glasses += 1
        elif rank == "J":
            jacks += 1
        else:
            others += 1

    board_cards = kings
    board_cards += min(queens, 2)
    board_cards += min(glasses, 1)
    board_cards += others

    counts = {
        "kings": kings,
        "queens": queens,
        "glasses": glasses,
        "jacks": jacks,
        "other_face": others,
    }
    return board_cards, counts


def material_components(
    hand: Sequence[str],
    point_cards: Sequence[str],
    face_cards: Sequence[str],
    is_turn: bool,
) -> Dict[str, float]:
    kings_count = sum(1 for c in face_cards if card_rank(c) == "K")
    contributing_points, cards_needed = select_contributing_points(
        point_cards, kings_count
    )
    board_facecards, face_counts = facecard_board_count(face_cards)
    board_cards = len(contributing_points) + board_facecards

    return {
        "material": float(len(hand) + 2 * board_cards + (0.5 if is_turn else 0.0)),
        "hand_count": len(hand),
        "board_cards": board_cards,
        "board_points_kept": contributing_points,
        "cards_needed_to_win": cards_needed,
        "face_counts": face_counts,
    }


def row_material(row: pd.Series) -> Dict[str, float]:
    """Compute material scores for both players for a single DataFrame row."""
    p0 = material_components(
        row.get("p0Hand", []),
        row.get("p0Points", []),
        row.get("p0FaceCards", []),
        bool(row.get("turn") == 0),
    )
    p1 = material_components(
        row.get("p1Hand", []),
        row.get("p1Points", []),
        row.get("p1FaceCards", []),
        bool(row.get("turn") == 1),
    )
    return {
        "material_p0": p0["material"],
        "material_p1": p1["material"],
        "material_diff": p0["material"] - p1["material"],
        "p0_cards_needed": p0["cards_needed_to_win"],
        "p1_cards_needed": p1["cards_needed_to_win"],
    }


def apply_material_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add material score columns to the DataFrame."""
    material_df = df.apply(row_material, axis=1, result_type="expand")
    return pd.concat([df, material_df], axis=1)


# --- Point-aware variant -----------------------------------------------------
POINT_FACE_VALUES = {
    "K": 2.0,
    "Q": 1.5,
    "8": 1.5,  # glasses
}
DEFAULT_FACE_VALUE = 2.0  # other non-jack faces still count as 2


def point_card_material_value(card: str) -> float:
    """Scale point card by its value: 10 -> 2.0, 5 -> 1.0, Ace -> 0.2."""
    return point_value(card) / 5.0


def facecard_material_value(card: str) -> float:
    """Map face cards to material values in the point-aware model."""
    rank = card_rank(card)
    if rank == "J":
        return 0.0
    return POINT_FACE_VALUES.get(rank, DEFAULT_FACE_VALUE)


def facecard_board_value(face_cards: Iterable[str]) -> float:
    """
    Total board value from face cards with caps on queens (2) and glasses (1).
    """
    kings = queens = glasses = 0
    other_faces: List[str] = []
    for card in face_cards:
        rank = card_rank(card)
        if rank == "K":
            kings += 1
        elif rank == "Q":
            queens += 1
        elif rank == "8":
            glasses += 1
        elif rank == "J":
            continue
        else:
            other_faces.append(card)

    total = 0.0
    total += kings * facecard_material_value("K")
    total += min(queens, 2) * facecard_material_value("Q")
    total += min(glasses, 1) * facecard_material_value("8")
    total += sum(facecard_material_value(card) for card in other_faces)
    return total


def material_components_point_aware(
    hand: Sequence[str],
    point_cards: Sequence[str],
    face_cards: Sequence[str],
    is_turn: bool,
) -> Dict[str, float]:
    kings_count = sum(1 for c in face_cards if card_rank(c) == "K")
    contributing_points, cards_needed = select_contributing_points(
        point_cards, kings_count
    )
    board_points_value = sum(point_card_material_value(c) for c in contributing_points)
    board_face_value = facecard_board_value(face_cards)

    return {
        "material": float(len(hand) + board_points_value + board_face_value + (0.5 if is_turn else 0.0)),
        "hand_count": len(hand),
        "board_cards": len(contributing_points),  # count still useful diagnostically
        "board_points_kept": contributing_points,
        "cards_needed_to_win": cards_needed,
    }


def row_material_point_aware(row: pd.Series) -> Dict[str, float]:
    """Point-aware material scores for both players for a single DataFrame row."""
    p0 = material_components_point_aware(
        row.get("p0Hand", []),
        row.get("p0Points", []),
        row.get("p0FaceCards", []),
        bool(row.get("turn") == 0),
    )
    p1 = material_components_point_aware(
        row.get("p1Hand", []),
        row.get("p1Points", []),
        row.get("p1FaceCards", []),
        bool(row.get("turn") == 1),
    )
    return {
        "material_p0": p0["material"],
        "material_p1": p1["material"],
        "material_diff": p0["material"] - p1["material"],
        "p0_cards_needed": p0["cards_needed_to_win"],
        "p1_cards_needed": p1["cards_needed_to_win"],
    }


def apply_material_scores_point_aware(df: pd.DataFrame) -> pd.DataFrame:
    """Add point-aware material score columns to the DataFrame."""
    material_df = df.apply(row_material_point_aware, axis=1, result_type="expand")
    return pd.concat([df, material_df], axis=1)
