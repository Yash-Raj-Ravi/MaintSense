# Technical Decisions

Significant decisions are recorded here with their status. Proposed decisions are not implementation commitments until accepted and validated.

## D-001: Keep raw CSV data out of Git

- **Status:** Accepted; validated locally.
- **Context:** The AI4I source CSV is a raw dataset and should remain separate from source history.
- **Decision:** Ignore CSV files under `data/raw/` and keep the dataset locally.
- **Alternatives considered:** Commit the dataset; track it with Git LFS; fetch it as a separate setup step.
- **Reasoning:** Avoids adding raw data to ordinary Git history while retaining a predictable local path.
- **Consequences:** A fresh checkout needs the dataset supplied separately. Dataset provenance is documented in the README.
- **Validation:** `git check-ignore` matched the current nested CSV to `.gitignore` rule `data/raw/**/*.csv`.

## D-002: Separate notebooks from reusable package code

- **Status:** Accepted as the current project layout; implementation exists, not yet behaviorally validated.
- **Context:** Exploration needs an interactive home while future preprocessing and model code should be reusable by tests and a future API.
- **Decision:** Use `notebooks/` for analysis, `src/maintsense/` for reusable Python code, `tests/` for automated tests, and `models/` for generated model artifacts.
- **Alternatives considered:** Put all code in notebooks; keep modules at repository root; build API/frontend folders now.
- **Reasoning:** Keeps analysis distinct from reusable application code and defers unneeded services until there is a validated ML workflow.
- **Consequences:** Package, tests, and model artifacts are placeholders; there is no implemented pipeline yet.
- **Validation:** Directory and package initializer are present. No tests or model code exist to validate.

## D-003: Treat failure-mode indicators as potential target leakage

- **Status:** Proposed; pending prediction-time validation.
- **Context:** The dataset includes `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` alongside `Machine failure`.
- **Decision:** Exclude these columns from a future candidate feature set unless evidence shows they are available before the prediction decision.
- **Alternatives considered:** Include all columns; exclude the indicators by default; decide after inspecting their generation and timing.
- **Reasoning:** Failure-mode indicators may encode information observed during or after a failure, which would inflate an offline score without supporting advance warning.
- **Consequences:** The model feature set and useful prediction horizon remain unresolved.
- **Validation:** Notebook cross-tabs confirm direct target overlap for four indicators and near-direct overlap for `RNF`. The leakage concern is supported descriptively; actual pre-decision availability is still unknown, so the feature decision remains proposed.
