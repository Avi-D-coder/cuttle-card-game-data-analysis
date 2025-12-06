from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class TrainResult:
    coef: np.ndarray
    intercept: float
    feature_names: List[str]
    auc: float
    brier: float
    n_train: int
    n_val: int


def train_symmetric_logit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
) -> TrainResult:
    """
    Train a logistic regression on symmetric features.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = LogisticRegression(max_iter=200, C=C)
    clf.fit(X_train, y_train)
    val_probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    brier = brier_score_loss(y_val, val_probs)
    return TrainResult(
        coef=clf.coef_[0],
        intercept=clf.intercept_[0],
        feature_names=feature_names,
        auc=auc,
        brier=brier,
        n_train=len(y_train),
        n_val=len(y_val),
    )


def predict_probs(result: TrainResult, X: np.ndarray) -> np.ndarray:
    logits = X @ result.coef + result.intercept
    return 1 / (1 + np.exp(-logits))


def coef_as_records(result: TrainResult) -> List[dict]:
    return [
        {"feature": name, "weight": float(w)}
        for name, w in zip(result.feature_names, result.coef)
    ]
