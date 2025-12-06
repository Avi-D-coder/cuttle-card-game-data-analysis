from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.cuttle_analysis.data_loader import load_gamestates
from .model import predict_probs
from .perspective_features import PerspectiveSpec, build_perspective_matrix


def load_weights(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    weights = {item["feature"]: item["weight"] for item in data["weights"]}
    weights["_intercept"] = data["intercept"]
    return weights


def quantize_weights(weights: Dict[str, float], max_target: float = 10.0) -> Dict[str, float]:
    # Exclude intercept from scaling
    feats = {k: v for k, v in weights.items() if k != "_intercept"}
    max_abs = max(abs(v) for v in feats.values() if v != 0) or 1.0
    scale = max_target / max_abs
    quantized = {k: round(v * scale) for k, v in feats.items()}
    quantized["_intercept"] = round(weights["_intercept"] * scale)
    quantized["_scale"] = scale
    return quantized


def evaluate_quantized(
    weights: Dict[str, float],
    feature_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    # Reconstruct weight vector with same ordering as feature_names
    coef = np.array([weights.get(f, 0.0) for f in feature_names], dtype=float)
    intercept = weights.get("_intercept", 0.0)
    probs = 1 / (1 + np.exp(-(X @ coef + intercept)))
    return {
        "auc": roc_auc_score(y, probs),
        "brier": brier_score_loss(y, probs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize perspective weights to whole-number cheat sheet.")
    parser.add_argument("--csv", type=str, default="first_100k_gamestates_processed.csv")
    parser.add_argument(
        "--weights-json",
        type=str,
        default="artifacts-ml-perspective/weights_nocardsneeded.json",
        help="Trained weights (cards-needed folded) to quantize.",
    )
    parser.add_argument("--out-json", type=str, default="artifacts-ml-perspective/weights_quantized.json")
    parser.add_argument("--out-csv", type=str, default="artifacts-ml-perspective/weights_quantized.csv")
    args = parser.parse_args()

    df = load_gamestates(args.csv)
    spec = PerspectiveSpec(include_scrap=True, include_deck_size=True)
    X, y, feature_names = build_perspective_matrix(df, spec)

    raw_weights = load_weights(Path(args.weights_json))
    quant = quantize_weights(raw_weights, max_target=10.0)

    # Evaluate raw vs quantized (in the scaled space)
    raw_metrics = evaluate_quantized(raw_weights, feature_names, X, y)
    quant_metrics = evaluate_quantized(
        {k: v / quant["_scale"] if k not in ("_intercept", "_scale") else raw_weights.get(k, 0.0) for k, v in quant.items()},
        feature_names,
        X,
        y,
    )

    Path(args.out_json).write_text(json.dumps({"weights": quant, "feature_order": feature_names, "metrics_raw": raw_metrics, "metrics_quant": quant_metrics}, indent=2))
    pd.DataFrame(
        [{"feature": k, "weight": v} for k, v in quant.items() if k not in ("_intercept", "_scale")]
        + [{"feature": "_intercept", "weight": quant["_intercept"]}]
    ).to_csv(args.out_csv, index=False)
    print(f"Wrote quantized weights to {args.out_json} and {args.out_csv}")
    print(f"Raw metrics: {raw_metrics}")
    print(f"Quantized metrics (scaled): {quant_metrics}")


if __name__ == "__main__":
    main()
