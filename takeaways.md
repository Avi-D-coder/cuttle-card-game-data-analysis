# Material Score Predictiveness — Takeaways

Key outputs live in `artifacts/`:
- Tables: `win_rates_by_diff.csv`, `flip_rates_by_diff.csv`, `deck_depth_stats.csv`, `win_rates_by_deck_and_diff.csv`, `decisive_lead_timing.csv`, `move_type_material_deltas.csv`, `first_mover_win_rates.csv`, `first_points_win_rates.csv`, `first_kings_win_rates.csv`
- Plots: `win_prob_logit.png`, `win_prob_by_deck.png`, `flip_rate.png`, `flip_rate_by_deck.png`, `win_rate_by_deck.png`, `win_rate_heatmap.png`, `flip_rate_heatmap.png`, `lead_timing.png`, `move_deltas.png`, `first_mover.png`, `first_points.png`, `first_kings.png`

## Big picture
- Material diff is informative but not decisive until ~±4: leaders win ~82–91% when diff in (2,4] or (4,6]; flip risk drops to ~0.26 at +2 to +4 and ~0.11 at +4 to +6. Small edges (|diff|≤2) flip often (≥0.5, ~0.96–0.99 near 0).
- Deck depth: shallow decks stabilize faster than deep ones; near-zero leads flip frequently across buckets; bigger leads (≥+4) are robust even with many cards left.
- Lead timing: eventual winners hit +2 around move ~16 (77% of games), +4 at move ~34 (39% of games), +6 at move ~49 (9% of games). Staying ahead after the first +2 remains hard (~24% stable in this run).
- Moves that shift material most (move value = change in diff per move): scuttles (~+0.07 on average, median +1) and counters (+0.05, median +1) add material; point plays (-0.07, median -1) and face cards (-0.04, median -0.5) typically spend material; drawing is mildly plus (~+0.04). Seven-to-scuttle stands out (~+0.6 average).
- Win-prob curves: advantage becomes strong around +3–4; flip probabilities peak near zero diff regardless of deck depth.
- Initiative and first actions:
  - First mover (player on move 1) wins ~51.6% overall (slight edge).
  - First to play points: if P1 plays points first, they win ~57%; if P0 plays points first, they win ~46%.
  - First to play a king: if P1 plays a king first, they win ~61%; if P0 plays a king first, they win ~51%.

## Player-facing guidance
- Treat +4 material as a “likely conversion” edge, especially late; below +2, expect volatility and play for swings.
- When behind ≤2, prioritize swingy moves (scuttles, counters, sevens) to flip the lead; when ahead ≥4, prefer low-variance lines and avoid giving counterplay.
- Deck-aware adjustment: with few cards left, smaller leads are safer; with many cards left, assume opponents can still reverse unless you’re >+4.

## Plot/key notes
- Deck legends (Q1–Q5) on `win_prob_by_deck.png`, `flip_rate_by_deck.png`, and `win_rate_by_deck.png` include the card-count ranges for each bucket (Q1 = shallow deck; Q5 = deep deck).
