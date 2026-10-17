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

- **Status:** Accepted; implemented and covered by baseline tests.
- **Context:** Exploration needs an interactive home while future preprocessing and model code should be reusable by tests and a future API.
- **Decision:** Use `notebooks/` for analysis, `src/maintsense/` for reusable Python code, `tests/` for automated tests, and `models/` for generated model artifacts.
- **Alternatives considered:** Put all code in notebooks; keep modules at repository root; build API/frontend folders now.
- **Reasoning:** Keeps analysis distinct from reusable application code and defers unneeded services until there is a validated ML workflow.
- **Consequences:** A small reusable baseline and its tests are implemented; the trained baseline is not persisted as a model artifact.
- **Validation:** Package module exists; baseline unit tests cover feature selection, split reproducibility, and train-fitted preprocessing.

## D-003: Treat failure-mode indicators as potential target leakage

- **Status:** Proposed; pending prediction-time validation.
- **Context:** The dataset includes `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` alongside `Machine failure`.
- **Decision:** Exclude these columns from a future candidate feature set unless evidence shows they are available before the prediction decision.
- **Alternatives considered:** Include all columns; exclude the indicators by default; decide after inspecting their generation and timing.
- **Reasoning:** Failure-mode indicators may encode information observed during or after a failure, which would inflate an offline score without supporting advance warning.
- **Consequences:** The model feature set and useful prediction horizon remain unresolved.
- **Validation:** Notebook cross-tabs confirm direct target overlap for four indicators and near-direct overlap for `RNF`. The leakage concern is supported descriptively; actual pre-decision availability is still unknown, so the feature decision remains proposed.

## D-004: Use operating measurements and Type as baseline features

- **Status:** Accepted for the initial baseline; operational feature availability still needs confirmation.
- **Context:** The AI4I dataset includes identifiers, operational measurements, a categorical product type, and failure-mode indicators that may encode the target.
- **Decision:** Predict `Machine failure` from `Type`, air temperature, process temperature, rotational speed, torque, and tool wear. Exclude `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`.
- **Alternatives considered:** Include identifiers or failure-mode indicators; omit `Type`; treat every field as numeric.
- **Reasoning:** IDs are not operational measurements, failure-mode timing is unknown and strongly associated with the target, and `Type` is categorical.
- **Consequences:** This is a provisional baseline feature set, not proof that all inputs are available at a real prediction time.
- **Validation:** The source feature list and unit test assert the selected fields and exclusions; validation split evaluation ran using this feature set.

## D-005: Reserve stratified train, validation, and test partitions

- **Status:** Accepted and implemented.
- **Context:** Model selection requires validation data while the final test set must remain untouched until choices are frozen. Failures are rare.
- **Decision:** Use a reproducible stratified 60/20/20 train/validation/test split with random state 42. Fit preprocessing as part of each training-only model pipeline. Use validation metrics for baseline comparison; do not score the final test partition during selection.
- **Alternatives considered:** One train/test split; random unstratified partitions; cross-validation immediately; oversampling.
- **Reasoning:** Stratification preserves the rare class in each partition, the held-out validation set supports an initial comparison, and the separate test partition remains available for a later unbiased check. No oversampling is needed for this baseline comparison.
- **Consequences:** The split is a row-wise random split and may not represent temporal or machine-group generalization. The appropriate deployment split remains an open question.
- **Validation:** Fixed-seed split and stratification tests pass; baseline execution produced 6,000/2,000/2,000 rows. The final test set was not evaluated.

## D-006: Compare built-in tree ensembles without scaling

- **Status:** Accepted for the initial comparison; settings are not tuned.
- **Context:** The linear baseline needs comparison with nonlinear traditional ML candidates; scikit-learn is already installed.
- **Decision:** Add `RandomForestClassifier` and `GradientBoostingClassifier` using the same selected inputs and split. One-hot encode `Type`, pass numerical features through without scaling, and fit preprocessing inside each pipeline on training rows only. Use fixed random seeds; configure the forest with `n_jobs=1` for reliable execution in the current Windows environment.
- **Alternatives considered:** Install XGBoost/LightGBM; scale tree features; tune hyperparameters; oversample; inspect test results.
- **Reasoning:** The built-in estimators add no dependency, tree splits do not require numerical scaling, and the common validation protocol allows comparison without tuning to the held-out test set.
- **Consequences:** Results are fixed-setting validation comparisons; they do not establish real-world performance or settle the operational threshold.
- **Validation:** Both models fit and evaluate successfully; the new model-comparison test passes. Validation metrics are recorded in ML-002. Test partition remains unevaluated.
