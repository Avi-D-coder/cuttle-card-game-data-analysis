"""
Preprocess the raw gamestate CSV to add jacked-card tracking.

We rebuild jack stacks per game:
- On jack plays (moveType == 'jack' or 'sevenJack'), push the jack player onto
  the target card's stack (control flips to the top jack).
- On jack removals (oneOffTargetType == 'jack'), pop the top of the target card's
  stack (control flips to the next stack holder, or none if empty).

Output columns (per row):
- p0JackedCards / p1JackedCards: list of strings like "J9C", "JJTC", "JJJQC"
  indicating stack height via repeated 'J' prefix, assigned to the controlling
  player at that point in time.

This reconstructs jack control using only move order and target annotations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add jacked card tracking and emit processed CSV.")
    parser.add_argument("--input", type=str, default="first_100k_gamestates.csv")
    parser.add_argument("--output", type=str, default="first_100k_gamestates_processed.csv")
    return parser.parse_args()


def base_card(card: str) -> str:
    """Strip jack annotations like '9C(JH-p0)' -> '9C'."""
    if not isinstance(card, str):
        return ""
    m = re.match(r"^([^\\(]+)", card.strip())
    return m.group(1) if m else card.strip()


def process_game(group: pd.DataFrame) -> pd.DataFrame:
    # card -> stack of controllers (list[int])
    stacks: Dict[str, List[int]] = {}
    rows = []
    for _, row in group.sort_values("move_number").iterrows():
        move = str(row.get("moveType", ""))
        target_card = base_card(str(row.get("targetCard", "")))
        oneoff_target = base_card(str(row.get("oneOffTarget", "")))

        if move in {"jack", "sevenJack"} and target_card:
            stack = stacks.setdefault(target_card, [])
            stack.append(int(row.get("playedBy", 0)))

        if str(row.get("oneOffTargetType", "")) == "jack" and oneoff_target:
            stack = stacks.get(oneoff_target)
            if stack:
                stack.pop()
                if not stack:
                    stacks.pop(oneoff_target, None)

        p0_jacked = []
        p1_jacked = []
        for card, stack in stacks.items():
            if not stack:
                continue
            token = f"{'J'*len(stack)}{card}"
            controller = stack[-1]
            if controller == 0:
                p0_jacked.append(token)
            else:
                p1_jacked.append(token)

        new_row = row.copy()
        new_row["p0JackedCards"] = str(p0_jacked)
        new_row["p1JackedCards"] = str(p1_jacked)
        rows.append(new_row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    processed = (
        df.groupby("gameId_hashed", sort=False)
        .apply(process_game)
        .reset_index(drop=True)
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, index=False)
    print(f"Wrote processed CSV with jacked card columns to {args.output}")


if __name__ == "__main__":
    main()
