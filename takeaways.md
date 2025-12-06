# Material Score Predictiveness — Takeaways

Key outputs live in `artifacts/`:
- Tables: `win_rates_by_diff.csv`, `flip_rates_by_diff.csv`, `deck_depth_stats.csv`, `win_rates_by_deck_and_diff.csv`, `decisive_lead_timing.csv`, `move_type_material_deltas.csv`, `first_mover_win_rates.csv`, `first_points_win_rates.csv`, `first_kings_win_rates.csv`
- Plots: `win_prob_logit.png`, `win_prob_by_deck.png`, `flip_rate.png`, `flip_rate_by_deck.png`, `win_rate_by_deck.png`

## Big picture
- Material diff is informative but not decisive until ~±4: leaders win ~82–91% when diff in (2,4] or (4,6], and flip risk falls below ~0.27 beyond +2. Small edges (|diff|≤2) flip often (46–73%).
- Deck depth matters: with a short deck (Q1 in the legends), even +2 is fairly steady (flip ~0.20; leader wins ~56%), while deep-deck states (Q4/Q5) need ~+4 for similar confidence.
- Lead timing: eventual winners hit +2 at move ~16 (77% of games), +4 at move ~34 (39% of games), +6 at move ~49 (9% of games). Only ~42% of games keep the first +2 lead to the end; ~35% keep the first +4.
- Moves that shift material most: scuttles and counters add material; point plays usually spend material (median -1); seven → scuttle/Jack spikes; draws mildly plus; faceCard plays slightly negative on average.
- Win-prob curves: logistic fits cross 60/70/80% p0 win odds around +1.3/+2.3/+3.6 overall; thresholds are lower (smaller diff needed) in low-deck buckets.
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
