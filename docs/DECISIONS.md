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

## D-007: Analyze thresholds on validation probabilities only

- **Status:** Accepted for analysis; no operating threshold selected.
- **Context:** Default-threshold metrics show different trade-offs between missed failures and false alarms, and there is not yet an agreed operational cost or warning horizon.
- **Decision:** Evaluate Random Forest and Gradient Boosting probabilities on the existing validation partition at cutoffs from 0.05 to 0.95 in 0.05 steps, including 0.50. Report representative cutoffs 0.25, 0.50, and 0.75 and a validation precision-recall curve. Classify probabilities equal to the threshold as positive.
- **Alternatives considered:** Select the highest F1 threshold; tune a cutoff to the validation set; score the test set; report only the default 0.50 cutoff.
- **Reasoning:** A fixed grid makes the precision/recall and false-alarm/missed-failure trade-off visible without claiming an optimum before operational costs are known. The untouched test set remains available for later confirmation.
- **Consequences:** The analysis informs a future decision but does not recommend a production threshold. A single validation split may not reflect future machine populations or time periods.
- **Validation:** Threshold functions and tests pass; validation metrics and plot are recorded in ML-003. Test set was not evaluated.

## D-008: Freeze a validation-selected model and threshold

- **Status:** Accepted, frozen, and evaluated once on the reserved test partition.
- **Context:** ML-001 through ML-003 compared five fixed traditional classifiers and showed validation trade-offs across probability thresholds. The project has no agreed operational cost ratio or alert horizon.
- **Decision:** Compare each of the five existing candidates at every fixed threshold from 0.05 to 0.95 using validation data only. Select the configuration with the highest validation F1; break ties by higher recall, fewer false positives, then the threshold nearest 0.50, with stable candidate/grid order as the final tie-break. This transparent benchmark rule selects Random Forest at 0.40. Freeze this choice before test evaluation.
- **Alternatives considered:** Select on accuracy; select only among tree candidates; choose a threshold from assumed alert costs; optimize on the test set.
- **Reasoning:** F1 balances precision and recall for the rare failure class, while the tie-breaks make missed failures and false alarms visible. The rule introduces no fabricated operational costs and keeps the test set out of selection.
- **Consequences:** The choice is specific to this single synthetic dataset, fixed split, candidate set, and rule. It does not identify a universally optimal model or operating threshold. The selected pipeline was refit on training rows only and evaluated on the reserved test partition once.
- **Validation:** The selection function compared all five candidates over all 19 grid values. Validation winner: Random Forest, threshold 0.40; precision 0.8824, recall 0.6618, F1 0.7563, 6 false positives, 23 false negatives, confusion matrix `[[1926, 6], [23, 45]]`, accuracy 0.9855, average precision 0.7772. The frozen configuration was then evaluated once on test: precision 0.8367, recall 0.6029, F1 0.7009, average precision 0.7817, accuracy 0.9825, 8 false positives, 27 false negatives, confusion matrix `[[1924, 8], [27, 41]]`. These test results did not trigger tuning or another evaluation.

## D-009: Persist the selected pipeline for local inference

- **Status:** Accepted and implemented.
- **Context:** The selected Random Forest and its preprocessing need to be reused consistently for single-record predictions.
- **Decision:** Fit the existing Random Forest scikit-learn `Pipeline` on only the established training partition and serialize that single pipeline with joblib to `models/random_forest.joblib`. Keep the already selected 0.40 decision threshold in the inference module. Keep the generated binary artifact out of Git.
- **Alternatives considered:** Serialize only the classifier; refit preprocessing separately at inference; commit the trained binary artifact; add a serving API now.
- **Reasoning:** Saving the complete pipeline preserves the fitted one-hot encoder and its categories alongside the fitted classifier. Joblib is already available through the installed Python environment; no dependency was installed. A local ignored artifact supports reproducible testing without committing generated model data.
- **Consequences:** Inference requires the six documented input features and compatible scikit-learn/joblib versions. Joblib files use Python serialization and must be loaded only from trusted sources. The artifact supports synthetic-dataset classification only and does not establish early-warning utility.
- **Validation:** Artifact was generated from `X_train`/`y_train` only; preprocessing and classifier are both present in the saved pipeline. Tests verify input validation, artifact loading, and identical pre-save/post-load probabilities. The reserved test set was not reevaluated.

## D-010: Serve predictions through a thin FastAPI layer

- **Status:** Accepted and implemented; runtime tests pass.
- **Context:** The persisted pipeline and single-record inference function are established; callers need a local HTTP interface.
- **Decision:** Add FastAPI endpoints for health, model metadata, and one-record predictions. Load the ignored pipeline once during application lifespan, validate request shape and values with Pydantic, and delegate prediction to `inference.predict_record`.
- **Alternatives considered:** Reload the artifact on each request; duplicate feature preprocessing or classification in the API; fail application startup when the artifact is absent.
- **Reasoning:** One lifespan load avoids repeated disk reads; existing inference remains the source of truth. Keeping the app alive without an artifact allows health diagnostics while returning HTTP 503 for prediction.
- **Consequences:** The API process needs a locally generated trusted artifact and compatible Python/scikit-learn/joblib versions. This endpoint classifies synthetic-dataset records and is not a real-world early-warning service. `requirements.txt` now records notebook, ML, API, server, and HTTP test-client dependencies because no dependency manifest previously existed.
- **Validation:** API tests cover valid inference, one-time loading, request validation, health/model-info, and missing-artifact behavior. No test-set evaluation or model selection was performed in this milestone.

## D-011: Use a Vite React client with a development API proxy

- **Status:** Accepted and implemented; frontend checks are run for this milestone.
- **Context:** The FastAPI prediction service is available, and the project needs a browser interface for single-record classification and transparent display of previously documented evaluation results.
- **Decision:** Create a TypeScript React dashboard in `frontend/`. Read the backend base URL from `VITE_API_BASE_URL`; in development, optionally proxy `/api` to `MAINTSENSE_API_PROXY_TARGET`. For direct cross-origin use, allow only explicitly configured origins through `MAINTSENSE_CORS_ORIGINS`.
- **Alternatives considered:** Hardcode a backend URL into components; duplicate model scoring in JavaScript; add a frontend server or database; allow every origin for CORS.
- **Reasoning:** A small typed HTTP client mirrors the Pydantic contract and keeps prediction logic in the backend. The Vite proxy provides same-origin local calls, while an explicit backend allow-list supports direct deployments without wildcard CORS.
- **Consequences:** The browser requires the API and local model artifact to be available. Vite `VITE_*` values are public build-time configuration and must not contain secrets. Evaluation metrics are displayed from ML-004 documentation and are not recomputed by the client.
- **Validation:** Six Vitest tests pass; ESLint and the TypeScript/Vite production build pass. The full Python suite passes 25 tests, including exact-origin CORS preflight coverage. No AI4I test-set evaluation or model/threshold changes were made for this milestone.
