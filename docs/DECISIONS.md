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
- **Validation:** The current frontend suite contains 12 Vitest tests. Commit `44fc192` passed frontend tests, ESLint, TypeScript, and the Vite production build. The full Python suite then passed 25 tests, including exact-origin CORS preflight coverage. No AI4I test-set evaluation or model/threshold changes were made for the status-handling commit.

## D-012: Provision the official AI4I dataset with an observed checksum

- **Status:** Accepted; checksum validated against an official UCI download.
- **Context:** The raw dataset is ignored by Git, so a fresh checkout needs a reproducible way to obtain the exact data used by the established workflow.
- **Decision:** Download UCI dataset 601's official ZIP to the fixed `data/raw/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv` location. Validate the extracted CSV with SHA-256 `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`, the expected official columns, and 10,000 data rows. Refuse to overwrite an existing destination.
- **Alternatives considered:** Commit the CSV, pin only a URL, trust an unverified local copy, or use a third-party mirror.
- **Reasoning:** A pinned digest detects changed, truncated, or invalid downloads and preserves the original dataset outside Git.
- **Consequences:** Initial provisioning needs network access to UCI; later runs validate the local CSV without downloading. The checksum is for the extracted CSV, not the ZIP container.
- **Validation:** Downloaded the official archive on 2026-10-17 from `https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip`, extracted `ai4i2020.csv`, and computed SHA-256 with PowerShell `Get-FileHash`. The official download and existing local CSV both measured 522,048 bytes and had the same digest. UCI dataset page: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset (DOI 10.24432/C5HS5C).

## D-013: Separate API liveness from prediction readiness

- **Status:** Accepted; implementation and API tests added, final validation pending.
- **Context:** The process should remain inspectable when its local artifact is missing, while an orchestrator needs a signal that prediction can run.
- **Decision:** Preserve `/health` as an HTTP 200 liveness/diagnostic endpoint and add `/ready`, returning HTTP 200 only when the application lifespan loaded a scikit-learn pipeline and HTTP 503 otherwise.
- **Alternatives considered:** Make `/health` fail when the model is unavailable; use `/predict` as a readiness probe; return success readiness unconditionally.
- **Reasoning:** Separating process liveness from prediction capability keeps diagnostics available while correctly withholding readiness.
- **Consequences:** Deployment health checks use `/ready`; `/health` still reports model availability. Responses do not reveal artifact paths or exception details.
- **Validation:** Healthy and missing-artifact API tests pass in the full 34-test Python suite. Wildcard CORS configuration is rejected by a regression test. No model evaluation was performed.

## D-014: Build separate API and reverse-proxied frontend containers

- **Status:** Accepted for the Docker Compose container workflow. The supplied production deployment is Vercel frontend to Render backend; current live behavior has not been independently verified here.
- **Context:** The API requires a validated joblib artifact. Docker Compose uses a same-origin frontend path, while the supplied production deployment uses separate Vercel and Render origins.
- **Decision:** Build an API image containing the validated model artifact; for Docker Compose, build the frontend with `VITE_API_BASE_URL=/api` and serve static files with Nginx. Nginx forwards `/api/*` to FastAPI while stripping the prefix. Compose waits for API readiness before starting the frontend. The supplied Vercel frontend calls the Render API directly and requires an exact allowed CORS origin. Artifact delivery is handled by D-016.
- **Alternatives considered:** Train or download at API startup; expose separate public origins and require CORS; use Vite preview in production; combine Node build tools with the Python runtime image.
- **Reasoning:** Missing artifacts fail early during image build, runtime performs no training or network data access, and same-origin routing avoids permissive CORS.
- **Consequences:** The model artifact must be available to the Docker build through the versioned release workflow in D-016. The Vite value is compiled into browser assets, so changing it requires rebuilding the frontend image. Direct cross-origin deployment must configure exact trusted origins.
- **Validation:** Historically, `docker compose config` and both Docker image builds passed in an earlier deployment-foundation check. During a later review, Docker Engine access was unavailable, so that review did not perform a current clean build or live container/reverse-proxy smoke check. The supplied deployment URLs do not independently verify current endpoint health, hosting settings, or deployed revisions. No model evaluation was performed.

## D-015: Separate pinned runtime and development dependencies

- **Status:** Accepted; pins captured from the installed Python 3.13.1 environment; development test-runner availability validated.
- **Context:** The prior manifest mixed broad API, ML, notebook, and testing ranges, allowing training/serving dependency drift.
- **Decision:** Put pinned API/inference packages in `requirements-runtime.txt`; layer notebook and test packages, including `pytest==9.1.1`, in `requirements-dev.txt`; keep `requirements.txt` as a backward-compatible full-development entry point; use `requirements.lock` as a constraints snapshot from the installed environment.
- **Alternatives considered:** Continue broad ranges; add a lock-generation tool; use a platform-specific package manager.
- **Reasoning:** This gives production a smaller install and constrains transitive resolution without adding a dependency-generation tool.
- **Consequences:** Upgrades require regenerating and reviewing pins. The constraints snapshot includes environment packages beyond the runtime set and platform-specific entries; only requested dependencies are installed.
- **Validation:** Pins were captured from `.venv`; the development requirements installed successfully, `python -m pytest --version` reported 9.1.1, `pip check` passed, and the complete Python suite passed (34 tests). Pip reports a pre-existing invalid `~ip` distribution warning in the local virtual environment. Runtime container dependency installation was validated during the prior successful image builds; the Docker Engine is unavailable in the current review environment.

## D-016: Fetch the existing model artifact from a versioned release

- **Status:** Accepted and implemented; whether the release asset was published or used for the supplied live deployment is unresolved from repository evidence.
- **Context:** The fitted joblib artifact and raw dataset are intentionally excluded from Git, but a clean Render Docker build needs the existing pipeline at the path expected by inference.
- **Decision:** Publish the unchanged `models/random_forest.joblib` as a manually created, versioned GitHub Release asset. Require its public HTTPS asset URL as `MODEL_ARTIFACT_URL` during Docker build and pin the measured SHA-256 and byte size in `scripts/fetch_model_artifact.py`. Verify the bytes before placing the file at `/app/models/random_forest.joblib`; retain the existing model-load validation.
- **Alternatives considered:** Commit the model or dataset; regenerate the model during image build or startup; download without verifying the contents; require a mounted/persistent disk.
- **Reasoning:** Release assets keep model binaries out of source history, checksum validation ensures the exact already-trained pipeline is used, and baking it into the image works with Render's ephemeral filesystem. No dataset, retraining, test evaluation, new secret, or new Python dependency is needed.
- **Consequences:** A public release asset is required for this Docker build path. Its download URL is public build configuration, not a credential. A changed artifact requires an intentional checksum/code review. The Docker build fails clearly if the URL is absent, outside the MaintSense release path, unreachable, or does not match the pinned size and digest. The repository does not establish whether the supplied live backend uses this path.
- **Validation:** The local artifact measured 4,872,018 bytes with SHA-256 `d22f14131f57f2ac8feb534e0551c50936f0d9da7528f08abc81e79e70e545f3`. Unit tests cover accepted/invalid URLs, successful verified download, and checksum rejection. The repository does not confirm whether a release was published or used by the supplied live backend. No model was retrained, and no AI4I test partition was evaluated.
