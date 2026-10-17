# Architecture

## Current state

The project currently has a local synthetic AI4I CSV, an exploratory Jupyter notebook, reusable Python modeling modules, and a locally generated inference pipeline. The raw dataset and generated model artifact are ignored by Git. A validation-selected configuration has been evaluated once on the reserved test partition; no serving API, web client, or container deployment exists.

```text
                       +--> notebooks/01_data_exploration.ipynb
                       |      descriptive inspection and visualizations
                       |
Local AI4I CSV --------+
(data/raw/, ignored)   |
                       +--> src/maintsense/baseline.py
  |-- stratified 60/20/20 split (seed 42)
  |-- train-fitted category encoding and model-appropriate numeric handling
  |-- dummy, Logistic Regression, Random Forest, and Gradient Boosting comparison
  |-- validation-only threshold analysis and precision-recall plot for tree models
  |-- validation-only comparison and frozen model/threshold selection
  |-- one-time final test evaluation of the frozen configuration
  |-- training-only Random Forest pipeline -> models/random_forest.joblib (ignored)
  +-- src/maintsense/inference.py loads pipeline -> one-record probability/class

tests/test_baseline.py   baseline unit tests
models/           generated model artifact placeholder
```

## Planned flow

The traditional ML workflow is implemented in `src/maintsense/`: `baseline.py` defines the fixed split and candidates, `threshold_analysis.py` evaluates validation thresholds, `model_selection.py` freezes a choice from validation and provides the separate final test evaluation, and `model_artifact.py` fits the selected Random Forest pipeline using training rows only. The serialized scikit-learn `Pipeline` is stored as `models/random_forest.joblib` using joblib; `.gitignore` excludes it. `inference.py` loads this artifact and requires exactly `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. `Type` must be H, L, or M; numerical inputs must be finite numbers. Identifiers, the target, and `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are rejected. Inference returns `failure_probability` and `predicted_class`, using the frozen threshold 0.40. No test rows were used to fit this artifact or evaluate inference. This is synthetic-dataset machine-failure classification, not a validated real-world early-warning system. Joblib artifacts should only be loaded from trusted sources and should be used with compatible Python/scikit-learn versions. A future FastAPI service may load the artifact; a future React/Vite client may call that API. Docker deployment is planned. These future components are not implemented.

Update this diagram and description when components or data flow change.
