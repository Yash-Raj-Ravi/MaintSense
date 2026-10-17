"""Validation-only threshold analysis for tree-based MaintSense candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from maintsense.baseline import DataSplits, build_models, load_dataset, split_dataset

THRESHOLDS = tuple(round(step / 20, 2) for step in range(1, 20))
SELECTED_THRESHOLDS = (0.25, 0.50, 0.75)
THRESHOLD_MODELS = ("random_forest", "gradient_boosting")


def evaluate_threshold(
    y_true: pd.Series,
    probabilities,
    threshold: float,
) -> dict[str, float | int | list[list[int]]]:
    """Score one probability cutoff, classifying scores >= cutoff as failures."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    probabilities = np.asarray(probabilities)
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have equal lengths")

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_threshold_grid(
    y_true: pd.Series,
    probabilities,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> list[dict[str, float | int | list[list[int]]]]:
    """Evaluate fixed thresholds without selecting or tuning a cutoff."""
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return [evaluate_threshold(y_true, probabilities, value) for value in thresholds]


def validation_probabilities(splits: DataSplits) -> dict[str, np.ndarray]:
    """Fit Random Forest and Gradient Boosting on train, predict validation only."""
    models = build_models()
    probabilities = {}
    for name in THRESHOLD_MODELS:
        estimator = models[name]
        estimator.fit(splits.X_train, splits.y_train)
        probabilities[name] = estimator.predict_proba(splits.X_validation)[:, 1]
    return probabilities


def validation_precision_recall_data(
    y_true: pd.Series,
    probabilities,
) -> dict[str, object]:
    """Return a validation precision-recall curve and average precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }


def save_precision_recall_plot(
    y_true: pd.Series,
    probabilities_by_model: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save validation precision-recall curves without accessing test data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, probabilities in probabilities_by_model.items():
        curve = validation_precision_recall_data(y_true, probabilities)
        ax.plot(
            curve["recall"],
            curve["precision"],
            label=f"{model_name.replace('_', ' ').title()} (AP={curve['average_precision']:.3f})",
        )
    prevalence = float((y_true == 1).mean())
    ax.axhline(prevalence, color="gray", linestyle="--", label=f"Prevalence={prevalence:.3f}")
    ax.set(
        title="Validation precision-recall curves",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run validation-only threshold analysis and save its PR curve."""
    splits = split_dataset(load_dataset())
    probabilities_by_model = validation_probabilities(splits)
    output_path = Path(__file__).resolve().parents[2] / "docs/figures/validation_pr_curve.png"

    print(
        "Validation class balance: "
        f"{int((splits.y_validation == 1).sum())}/{len(splits.y_validation)} "
        f"({splits.y_validation.mean():.1%} failures)"
    )
    print(f"Threshold grid: {THRESHOLDS}")
    for model_name, probabilities in probabilities_by_model.items():
        table = evaluate_threshold_grid(splits.y_validation, probabilities)
        print(f"\n{model_name}; average precision: "
              f"{validation_precision_recall_data(splits.y_validation, probabilities)['average_precision']:.4f}")
        for row in table:
            if row["threshold"] in SELECTED_THRESHOLDS:
                print(row)

    save_precision_recall_plot(
        splits.y_validation,
        probabilities_by_model,
        output_path,
    )
    print(f"Saved validation precision-recall plot: {output_path}")
    print("Reserved test partition was not evaluated.")


if __name__ == "__main__":
    main()
