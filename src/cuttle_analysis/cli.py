import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .data_loader import load_gamestates
from .material import apply_material_scores, apply_material_scores_point_aware
from .metrics import (
    add_future_flip_flag,
    decisive_lead_summary,
    flip_rate_by_diff,
    first_mover_win_rates,
    first_to_kings_win_rates,
    first_to_points_win_rates,
    leader_win_rate_by_deck_and_diff,
    move_type_material_deltas,
    stratify_by_deck_depth,
    win_rate_by_diff,
    win_rate_by_diff_and_deck,
)
from .visuals import (
    plot_first_action_win_rates,
    plot_lead_timing,
    plot_flip_by_deck_and_diff,
    plot_flip_by_diff,
    plot_logit_by_deck_bucket,
    plot_logit_win_prob,
    plot_move_deltas,
    plot_win_rate_heatmap,
    plot_flip_rate_heatmap,
    plot_win_rate_by_deck_and_diff,
    plot_win_rate_by_diff,
)



DEFAULT_DIFF_BINS = [-np.inf, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, np.inf]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cuttle material score predictiveness."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="first_100k_gamestates.csv",
        help="Path to the game state CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write summary CSVs.",
    )
    parser.add_argument(
        "--point-aware-output-dir",
        type=str,
        default=None,
        help="Optional directory for point-aware scoring outputs (e.g., artifacts-point-aware).",
    )
    parser.add_argument(
        "--diff-bins",
        type=str,
        default=None,
        help="Comma-separated material diff bin edges (override default).",
    )
    parser.add_argument(
        "--deck-quantiles",
        type=int,
        default=5,
        help="Number of deck-size quantiles for stratified stats.",
    )
    parser.add_argument(
        "--deck-bin-width",
        type=int,
        default=1,
        help="Deck size bin width for heatmaps (cards remaining).",
    )
    parser.add_argument(
        "--lead-margins",
        type=str,
        default="2,4,6",
        help="Comma-separated lead margins to summarize decisive lead timing.",
    )
    return parser.parse_args()


def parse_bins(bin_str: Optional[str]) -> Iterable[float]:
    if not bin_str:
        return DEFAULT_DIFF_BINS
    return [float(part) for part in bin_str.split(",")]


def ensure_output_dir(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    args = parse_args()
    bins = parse_bins(args.diff_bins)
    out_dir = ensure_output_dir(args.output_dir)
    point_out_dir = ensure_output_dir(args.point_aware_output_dir)
    lead_margins = [float(x) for x in args.lead_margins.split(",")]

    base_df = load_gamestates(args.csv)

    def run_pipeline(
        scoring_name: str,
        scorer,
        output_dir: Optional[Path],
    ) -> None:
        print(f"\n=== Running {scoring_name} scoring ===")
        df = scorer(base_df.copy())
        df, deck_legend = stratify_by_deck_depth(df, args.deck_quantiles)
        df = add_future_flip_flag(df)

        win_table = win_rate_by_diff(df, bins)
        flip_table = flip_rate_by_diff(df, bins)
        deck_table = leader_win_rate_by_deck_and_diff(
            df, bins, deck_quantiles=args.deck_quantiles
        )
        win_deck_table = win_rate_by_diff_and_deck(
            df, bins, deck_quantiles=args.deck_quantiles
        )
        lead_table = decisive_lead_summary(df, lead_margins)
        move_delta_table = move_type_material_deltas(df)
        first_mover_table = first_mover_win_rates(df)
        first_points_table = first_to_points_win_rates(df)
        first_kings_table = first_to_kings_win_rates(df)

        print("\nWin rates by material diff:")
        print(win_table.to_string(index=False))

        print("\nFuture lead flip rates by material diff:")
        print(flip_table.to_string(index=False))

        print("\nLeader win/flip by deck depth and diff:")
        print(deck_table.to_string(index=False))

        print("\nWin rates by deck depth and material diff:")
        print(win_deck_table.to_string(index=False))

        print("\nDecisive lead timing (eventual winner hits margin):")
        print(lead_table.to_string(index=False))

        print("\nMaterial diff delta by move type:")
        print(move_delta_table.to_string(index=False))

        print("\nWin rates by first mover:")
        print(first_mover_table.to_string(index=False))

        print("\nWin rates by who first plays points:")
        print(first_points_table.to_string(index=False))

        print("\nWin rates by who first plays a king:")
        print(first_kings_table.to_string(index=False))

        if output_dir:
            win_table.to_csv(output_dir / "win_rates_by_diff.csv", index=False)
            flip_table.to_csv(output_dir / "flip_rates_by_diff.csv", index=False)
            deck_table.to_csv(output_dir / "deck_depth_stats.csv", index=False)
            win_deck_table.to_csv(output_dir / "win_rates_by_deck_and_diff.csv", index=False)
            lead_table.to_csv(output_dir / "decisive_lead_timing.csv", index=False)
            move_delta_table.to_csv(output_dir / "move_type_material_deltas.csv", index=False)
            first_mover_table.to_csv(output_dir / "first_mover_win_rates.csv", index=False)
            first_points_table.to_csv(output_dir / "first_points_win_rates.csv", index=False)
            first_kings_table.to_csv(output_dir / "first_kings_win_rates.csv", index=False)

            # Plots
            try:
                plot_logit_win_prob(df, output_dir / "win_prob_logit.png")
                deck_buckets = sorted(
                    [b for b in df["deck_bucket"].dropna().unique()], key=str
                )
                plot_win_rate_by_diff(df, np.array(bins, dtype=float), output_dir / "win_rate.png")
                plot_flip_by_diff(df, np.array(bins, dtype=float), output_dir / "flip_rate.png")
                plot_lead_timing(lead_table, output_dir / "lead_timing.png")
                plot_move_deltas(move_delta_table, output_dir / "move_deltas.png")
                plot_first_action_win_rates(
                    first_mover_table, "Win rate: first mover", output_dir / "first_mover.png"
                )
                plot_first_action_win_rates(
                    first_points_table,
                    "Win rate: first to play points",
                    output_dir / "first_points.png",
                )
                plot_first_action_win_rates(
                    first_kings_table,
                    "Win rate: first to play a king",
                    output_dir / "first_kings.png",
                )
                if deck_buckets:
                    plot_logit_by_deck_bucket(
                        df, output_dir / "win_prob_by_deck.png", deck_buckets
                    )
                    plot_win_rate_by_deck_and_diff(
                        df,
                        np.array(bins, dtype=float),
                        output_dir / "win_rate_by_deck.png",
                        deck_legend,
                    )
                    plot_flip_by_deck_and_diff(
                        df,
                        np.array(bins, dtype=float),
                        output_dir / "flip_rate_by_deck.png",
                        deck_legend,
                    )
                # Heatmaps with deck-size bins for more granular view
                deck_bins = np.arange(
                    df["deck_size"].min(),
                    df["deck_size"].max() + args.deck_bin_width,
                    args.deck_bin_width,
                )
                plot_win_rate_heatmap(
                    df,
                    np.array(bins, dtype=float),
                    deck_bins,
                    output_dir / "win_rate_heatmap.png",
                )
                plot_flip_rate_heatmap(
                    df,
                    np.array(bins, dtype=float),
                    deck_bins,
                    output_dir / "flip_rate_heatmap.png",
                )
            except Exception as exc:  # pragma: no cover
                print(f"[warn] plotting failed: {exc}")
            print(f"\nWrote summaries to {output_dir}")

    # Run baseline
    run_pipeline("baseline", apply_material_scores, out_dir)
    # Run point-aware if requested
    if point_out_dir:
        run_pipeline("point-aware", apply_material_scores_point_aware, point_out_dir)


if __name__ == "__main__":
    main()
