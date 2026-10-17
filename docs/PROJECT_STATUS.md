# Project Status

Last updated: 2026-10-17

## Completed

- Initial Git repository and commit (`abfd24f` on `main`); README and ignore rules exist.
- ML workspace directories exist: `notebooks/`, `src/maintsense/`, `models/`, and `tests/`.
- `notebooks/01_data_exploration.ipynb` is authored. The AI4I CSV is present locally and excluded from Git.
- The exploration notebook executed successfully with the project `.venv`; observed results are recorded in `EXPERIMENT_LOG.md`.
- Project documentation system established and linked from the README.

## In progress

- Decide the prediction horizon, operational feature availability, and evaluation split before modeling.

## Planned

- Resolve feature availability, define the prediction horizon, and choose an evaluation split before training.
- Implement the first reproducible ML pipeline and tests, then evaluate beyond accuracy.
- Add the FastAPI backend, React/Vite frontend, and Docker deployment only after the modeling workflow is established.

## Known issues

- Prediction-time availability of the failure-mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) has not been established; executed cross-tabs show direct overlap with the target for four indicators.
- The repository has no implemented model, API, frontend, or automated test suite yet.
- Git worktree has uncommitted setup changes; current branch is `main` at `abfd24f`.

## Next actionable milestone

Document the real prediction-time availability of each candidate input and define the prediction horizon and evaluation split. Do not begin model training until the feature availability and evaluation design are documented.
