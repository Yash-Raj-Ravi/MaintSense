# Architecture

## Current state

The project currently has a local synthetic AI4I CSV, an exploratory Jupyter notebook, and reusable Python baseline, threshold-analysis, validation-selection, and final-evaluation modules. The raw dataset is ignored by Git. A validation-selected configuration has been evaluated once on the reserved test partition; no deployed model artifact, serving API, web client, or container deployment exists.

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
  +-- one-time final test evaluation of the frozen configuration

tests/test_baseline.py   baseline unit tests
models/           generated model artifact placeholder
```

## Planned flow

The traditional ML workflow is implemented in `src/maintsense/`: `baseline.py` defines the fixed split and candidates, `threshold_analysis.py` evaluates validation thresholds, and `model_selection.py` freezes a choice from validation and provides the separate final test evaluation. No model artifact is saved. This is synthetic-dataset machine-failure classification, not a validated real-world early-warning system. A future FastAPI service may load a validated artifact and accept prediction inputs; a future React/Vite client may call that API. Docker deployment is planned. These future components are not implemented.

Update this diagram and description when components or data flow change.
