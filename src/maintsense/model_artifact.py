"""Train and persist the selected inference pipeline from training rows only."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from maintsense.baseline import DATA_PATH, FEATURES, build_models, load_dataset, split_dataset

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "models/random_forest.joblib"
SELECTED_MODEL = "random_forest"


def train_and_save_artifact(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    artifact_path: Path = ARTIFACT_PATH,
) -> Pipeline:
    """Fit the selected preprocessing/model pipeline and save it with joblib."""
    missing_features = set(FEATURES).difference(X_train.columns)
    if missing_features:
        raise ValueError(f"Training data is missing required features: {sorted(missing_features)}")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have equal lengths")
    if X_train.empty:
        raise ValueError("training data must contain at least one row")

    pipeline = build_models()[SELECTED_MODEL]
    pipeline.fit(X_train.loc[:, FEATURES], y_train)
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    return pipeline


def main() -> None:
    """Build the ignored artifact from only the established training split."""
    splits = split_dataset(load_dataset(DATA_PATH))
    train_and_save_artifact(splits.X_train, splits.y_train)
    print(f"Saved training-only Random Forest pipeline to {ARTIFACT_PATH}")
    print("Validation and test partitions were not used for fitting or evaluation.")


if __name__ == "__main__":
    main()
