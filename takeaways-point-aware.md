# Material Score (Point-Aware Variant) — Takeaways

Outputs live in `artifacts-point-aware/` (mirrors baseline artifact set with point-aware scoring values).

## Big picture
- Point-aware scaling (10 → 2, faces weighted: K=2, Q=1.5, 8=1.5) produces similar shape to baseline but slightly less stable: leaders win ~80% in (2,4] vs ~81.7% baseline; flip risk in (2,4] is ~0.34 vs ~0.26 baseline.
- Small edges remain volatile: at diff in (1,2] leader wins ~66% with ~0.59 flip risk; near-zero diff flip risk ~0.89.
- Strong leads still convert: diff in (4,6] wins ~91% with flip risk ~0.14; (6,8] wins ~80% with ~0.02 flip risk (low sample).
- Lead timing: eventual winners hit +2 around move ~19 (hit rate ~0.70), later than baseline (~16). Stable +2 leads occur in ~31% of games (higher than baseline’s ~24%). +4 hits in ~26.5% of games around move ~38; +6 is rare (~4%).
- Move values shift because of point scaling: point plays are less punishing (avg ~-0.048, median -0.2) and scuttles are smaller bumps (avg ~+0.03, median +0.2) vs baseline (+0.07, median +1).

## Player-facing guidance (point-aware model)
- Treat +4 as a likely conversion edge; +2 is meaningful but still swingy (flip ~0.72).
- Small edges (≤2) are fragile; expect swings and play for volatility. Big edges (≥4) are robust, especially late in the deck.
- The model rewards higher point values directly: banking big point cards (9/10) swings material more than small points; scuttles/counters remain the clearest positive swing moves, but smaller than in baseline.
- Deck depth still matters: shallow decks stabilize faster; deep decks see high flip risk near zero diff and need larger leads for safety.
