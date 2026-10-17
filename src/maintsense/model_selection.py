"""Validation-based candidate selection and one-time final test evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score

from maintsense.baseline import DataSplits, build_models, validation_metrics
from maintsense.threshold_analysis import THRESHOLDS, evaluate_threshold


def select_validation_configuration(splits: DataSplits) -> dict[str, object]:
    """Compare every existing candidate over the fixed validation threshold grid.

    The selection rule maximizes validation F1, then recall, then minimizes
    false positives, then prefers a threshold nearer 0.50. Stable model/grid
    order resolves any remaining tie.
    """
    rows: list[dict[str, object]] = []
    for model_name, estimator in build_models().items():
        estimator.fit(splits.X_train, splits.y_train)
        probabilities = estimator.predict_proba(splits.X_validation)[:, 1]
        for threshold in THRESHOLDS:
            row = evaluate_threshold(
                splits.y_validation,
                probabilities,
                threshold,
            )
            row["model"] = model_name
            row["accuracy"] = float(
                accuracy_score(
                    splits.y_validation,
                    (probabilities >= threshold).astype(int),
                )
            )
            row["average_precision"] = float(
                average_precision_score(splits.y_validation, probabilities)
            )
            rows.append(row)

    selected = max(
        rows,
        key=lambda row: (
            row["f1"],
            row["recall"],
            -row["false_positives"],
            -abs(row["threshold"] - 0.50),
        ),
    )
    return {"selected": selected, "validation_results": rows}


def evaluate_selected_test(
    splits: DataSplits,
    *,
    model_name: str,
    threshold: float,
) -> dict[str, float | int | list[list[int]] | str]:
    """Fit the frozen candidate on train and score the reserved test once.

    This function is intentionally separate from validation selection. Call
    it only after the model and threshold have been recorded as frozen.
    """
    models = build_models()
    if model_name not in models:
        raise ValueError(f"unknown model candidate: {model_name}")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    estimator = models[model_name]
    estimator.fit(splits.X_train, splits.y_train)
    probabilities = np.asarray(estimator.predict_proba(splits.X_test)[:, 1])
    predictions = (probabilities >= threshold).astype(int)
    metrics = validation_metrics(splits.y_test, predictions, probabilities)
    matrix = metrics["confusion_matrix"]
    return {
        "model": model_name,
        "threshold": float(threshold),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "average_precision": metrics["average_precision"],
        "accuracy": float(accuracy_score(splits.y_test, predictions)),
        "confusion_matrix": matrix,
        "false_positives": int(matrix[0][1]),
        "false_negatives": int(matrix[1][0]),
    }
