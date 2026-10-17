# Project Status

Last updated: 2026-10-17

## Completed

- Initial Git repository and commits exist on `main`; README and ignore rules exist.
- ML workspace directories exist: `notebooks/`, `src/maintsense/`, `models/`, and `tests/`.
- `notebooks/01_data_exploration.ipynb` is authored. The AI4I CSV is present locally and excluded from Git.
- The exploration notebook executed successfully with the project `.venv`; observed results are recorded in `EXPERIMENT_LOG.md`.
- Reproducible baseline implemented in `src/maintsense/baseline.py`; four unit tests pass.
- Dummy, two Logistic Regression variants, Random Forest, and Gradient Boosting evaluated on validation data. Results are recorded in `EXPERIMENT_LOG.md`.
- Five unit tests pass, including model-comparison coverage.
- Project documentation system established and linked from the README.

## In progress

- Decide the operational prediction horizon and acceptable false-alarm versus missed-failure trade-off.

## Planned

- Finalize prediction-time feature availability and operational alert criteria before any final test evaluation.
- Evaluate the final test set only after model and decision criteria are fixed.
- Add the FastAPI backend, React/Vite frontend, and Docker deployment only after the modeling workflow is established.

## Known issues

- Prediction-time availability of the failure-mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) has not been established; executed cross-tabs show direct overlap with the target for four indicators.
- No API, frontend, or deployment code exists yet.
- Baseline implementation and documentation changes from this milestone are uncommitted; current branch is `main` at `abe817f`.

## Next actionable milestone

Define the prediction horizon and the operational cost of missed failures versus false alarms. Keep the test partition sealed until model selection and alert criteria are fixed.
