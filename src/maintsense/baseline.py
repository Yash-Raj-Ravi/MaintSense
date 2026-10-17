"""Reproducible, validation-only classical ML baselines for MaintSense."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

TARGET = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/raw/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv"
)


@dataclass(frozen=True)
class DataSplits:
    """Feature and target partitions; the test partition stays reserved."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the local AI4I CSV without including identifiers or failure flags."""
    frame = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def split_dataset(
    frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Create deterministic, stratified train/validation/test partitions."""
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1")

    X = frame[FEATURES]
    y = frame[TARGET]
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    validation_fraction_of_remainder = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=validation_fraction_of_remainder,
        random_state=random_state,
        stratify=y_train_validation,
    )
    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def make_preprocessor() -> ColumnTransformer:
    """Build transformations that a model pipeline fits on training rows only."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    """Return a majority dummy and two Logistic Regression comparisons."""
    return {
        "dummy_most_frequent": Pipeline(
            [("classifier", DummyClassifier(strategy="most_frequent"))]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
            ]
        ),
        "logistic_regression_balanced": Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def validation_metrics(
    y_true: pd.Series,
    y_pred,
    positive_scores,
) -> dict[str, float | list[list[int]]]:
    """Calculate binary metrics at the estimator's default 0.5 threshold."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, positive_scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_validation(splits: DataSplits) -> dict[str, dict[str, float | list[list[int]]]]:
    """Fit each candidate on train and score validation; never read test labels."""
    results = {}
    for name, estimator in build_models().items():
        estimator.fit(splits.X_train, splits.y_train)
        prediction = estimator.predict(splits.X_validation)
        probability = estimator.predict_proba(splits.X_validation)[:, 1]
        results[name] = validation_metrics(splits.y_validation, prediction, probability)
    return results


def main() -> None:
    """Run and print the reproducible validation comparison."""
    frame = load_dataset()
    splits = split_dataset(frame)
    results = evaluate_validation(splits)
    print(f"Dataset rows: {len(frame)}")
    print(
        "Split sizes (train/validation/test): "
        f"{len(splits.X_train)}/{len(splits.X_validation)}/{len(splits.X_test)}"
    )
    print(f"Random state: {RANDOM_STATE}; stratified proportions: 60/20/20")
    print("Final test partition was reserved and not evaluated.")
    for name, metrics in results.items():
        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
