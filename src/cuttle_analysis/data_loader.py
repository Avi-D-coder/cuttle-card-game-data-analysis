import ast
from typing import Iterable, List

import pandas as pd

RAW_LIST_COLUMNS: Iterable[str] = (
    "p0Hand",
    "p0Points",
    "p0FaceCards",
    "p0JackedCards",
    "p1Hand",
    "p1Points",
    "p1FaceCards",
    "p1JackedCards",
    "deck",
    "scrap",
    "discardedCards",
    "twos",
)

WINNER_MAP = {
    "p0_win": 0,
    "p1_win": 1,
    "stalemate": None,
}


def _parse_list(cell: str) -> List[str]:
    if pd.isna(cell) or cell == "":
        return []
    try:
        parsed = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_bool(cell: object) -> bool:
    if isinstance(cell, bool):
        return cell
    if pd.isna(cell):
        return False
    return str(cell).strip().lower() == "true"


def load_gamestates(csv_path: str) -> pd.DataFrame:
    """Load the game state CSV and coerce list/boolean/int columns."""
    df = pd.read_csv(csv_path)

    for col in RAW_LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list)

    bool_cols = ["is_last_row_for_game", "resolved"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(_parse_bool)

    int_cols = ["move_number", "playedBy", "turn", "phase"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df["deck_size"] = df["deck"].apply(len)
    df["game_length"] = (
        df.groupby("gameId_hashed")["move_number"].transform("max")
    )
    df["moves_to_end"] = df["game_length"] - df["move_number"]
    df["winner"] = df["result"].map(WINNER_MAP)

    return df
