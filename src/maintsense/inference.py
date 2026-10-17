"""Single-record inference using the saved MaintSense pipeline artifact."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from maintsense.baseline import FEATURES, NUMERICAL_FEATURES
from maintsense.model_artifact import ARTIFACT_PATH

DECISION_THRESHOLD = 0.40
VALID_TYPES = {"H", "L", "M"}
EXCLUDED_INPUTS = {
    "UDI",
    "UID",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}


def validate_record(record: Mapping[str, object]) -> pd.DataFrame:
    """Validate one record and return only the six approved model features."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping of feature names to values")

    supplied = set(record)
    forbidden = supplied.intersection(EXCLUDED_INPUTS)
    if forbidden:
        raise ValueError(
            "Excluded identifiers/target/failure indicators are not accepted: "
            f"{sorted(forbidden)}"
        )

    missing = set(FEATURES).difference(supplied)
    if missing:
        raise ValueError(f"Missing required input features: {sorted(missing)}")

    extra = supplied.difference(FEATURES)
    if extra:
        raise ValueError(f"Unexpected input features: {sorted(extra)}")

    machine_type = record["Type"]
    if not isinstance(machine_type, str) or machine_type not in VALID_TYPES:
        raise ValueError(f"Type must be one of {sorted(VALID_TYPES)}")

    validated = {"Type": machine_type}
    for feature in NUMERICAL_FEATURES:
        value = record[feature]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{feature} must be a finite number")
        if not np.isfinite(value):
            raise ValueError(f"{feature} must be a finite number")
        validated[feature] = float(value)

    return pd.DataFrame([validated], columns=FEATURES)


def load_pipeline_artifact(artifact_path: Path = ARTIFACT_PATH) -> Pipeline:
    """Load and minimally validate a locally generated sklearn Pipeline."""
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}; run "
            "python -m maintsense.model_artifact to create it"
        )
    try:
        pipeline = joblib.load(artifact_path)
    except Exception as exc:
        raise ValueError(f"Could not load model artifact at {artifact_path}") from exc

    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict_proba"):
        raise TypeError("Model artifact must be a fitted scikit-learn Pipeline with predict_proba")
    if not hasattr(pipeline, "classes_"):
        raise ValueError("Model artifact Pipeline is not fitted")
    if 1 not in pipeline.classes_:
        raise ValueError("Model artifact does not contain the failure class 1")
    return pipeline


def predict_record(
    record: Mapping[str, object],
    pipeline: Pipeline,
) -> dict[str, float | int]:
    """Predict failure probability and class for one validated machine record."""
    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict_proba"):
        raise TypeError("pipeline must be a fitted scikit-learn Pipeline with predict_proba")
    if not hasattr(pipeline, "classes_"):
        raise ValueError("pipeline is not fitted")
    features = validate_record(record)
    failure_column = list(pipeline.classes_).index(1)
    probability = float(pipeline.predict_proba(features)[0, failure_column])
    return {
        "failure_probability": probability,
        "predicted_class": int(probability >= DECISION_THRESHOLD),
    }


def predict_from_artifact(
    record: Mapping[str, object],
    artifact_path: Path = ARTIFACT_PATH,
) -> dict[str, float | int]:
    """Load the saved pipeline and predict a single machine record."""
    return predict_record(record, load_pipeline_artifact(artifact_path))
