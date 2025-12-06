from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cuttle_analysis.data_loader import load_gamestates
from .perspective_features import PerspectiveSpec, build_perspective_matrix
from .model import coef_as_records, train_symmetric_logit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-player perspective model.")
    parser.add_argument("--csv", type=str, default="first_100k_gamestates_processed.csv")
    parser.add_argument("--out-dir", type=str, default="artifacts-ml-perspective")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--C", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_gamestates(args.csv)
    spec = PerspectiveSpec(include_scrap=False, include_deck_size=False)
    X, y, feature_names = build_perspective_matrix(df, spec)
    feature_means = dict(zip(feature_names, X.mean(axis=0)))
    result = train_symmetric_logit(
        X, y, feature_names, test_size=args.test_size, C=args.C
    )

    # Save weights
    weights_path = out_dir / "weights.json"
    with weights_path.open("w") as f:
        json.dump(
            {
                "intercept": float(result.intercept),
                "weights": coef_as_records(result),
                "feature_order": feature_names,
            },
            f,
            indent=2,
        )

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "auc": result.auc,
                "brier": result.brier,
                "n_train": result.n_train,
                "n_val": result.n_val,
                "test_size": args.test_size,
                "C": args.C,
            },
            f,
            indent=2,
        )

    pd.DataFrame(coef_as_records(result)).to_csv(out_dir / "weights.csv", index=False)

    logits = X @ result.coef + result.intercept
    probs = 1 / (1 + np.exp(-logits))
    df_eval = pd.DataFrame({"prob": probs, "winner": y})
    df_eval["prob_bin"] = pd.cut(df_eval["prob"], bins=np.linspace(0, 1, 11))
    calib = (
        df_eval.groupby("prob_bin")
        .agg(count=("winner", "count"), win_rate=("winner", "mean"), avg_prob=("prob", "mean"))
        .reset_index()
    )
    calib.to_csv(out_dir / "calibration.csv", index=False)

    df_eval["logit"] = logits
    df_eval["logit_bin"] = pd.cut(df_eval["logit"], bins=np.linspace(-8, 8, 17))
    diff_table = (
        df_eval.groupby("logit_bin")
        .agg(count=("winner", "count"), win_rate=("winner", "mean"), avg_logit=("logit", "mean"))
        .reset_index()
    )
    diff_table.to_csv(out_dir / "logit_win_curve.csv", index=False)

    print(f"Trained perspective model. AUC={result.auc:.4f}, Brier={result.brier:.4f}")
    print(f"Wrote weights to {weights_path}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
