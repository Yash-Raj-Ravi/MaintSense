# Project Status

Last updated: 2026-10-17

## Completed

- Initial Git repository and commits exist on `main`; README and ignore rules exist.
- ML workspace directories exist: `notebooks/`, `src/maintsense/`, `models/`, and `tests/`.
- `notebooks/01_data_exploration.ipynb` is authored. The AI4I CSV is present locally and excluded from Git.
- The exploration notebook executed successfully with the project `.venv`; observed results are recorded in `EXPERIMENT_LOG.md`.
- Reproducible baseline implemented in `src/maintsense/baseline.py`; four unit tests pass.
- Dummy, two Logistic Regression variants, Random Forest, and Gradient Boosting evaluated on validation data. Results are recorded in `EXPERIMENT_LOG.md`.
- Validation threshold analysis and precision-recall plot completed for Random Forest and Gradient Boosting; no operating threshold selected.
- Validation-based selection rule frozen; Random Forest at threshold 0.40 selected for one final test evaluation.
- Frozen Random Forest configuration evaluated once on the reserved test partition; metrics are recorded in `EXPERIMENT_LOG.md`.
- Ten unit tests pass, including model-comparison, threshold, and selection/evaluation coverage.
- Project documentation system established and linked from the README.

## In progress

- Decide the operational prediction horizon and acceptable false-alarm versus missed-failure trade-off before recommending a threshold.

## Planned

- Finalize prediction-time feature availability and operational alert criteria before any real-world early-warning claims.
- Add the FastAPI backend, React/Vite frontend, and Docker deployment only after the modeling workflow is established.

## Known issues

- Prediction-time availability of the failure-mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) has not been established; executed cross-tabs show direct overlap with the target for four indicators.
- No API, frontend, or deployment code exists yet.

## Next actionable milestone

Review the final test estimate and resolve prediction-time feature availability, alert horizon, and operational costs before making any real-world early-warning claims. Treat this only as synthetic-dataset machine-failure classification.
