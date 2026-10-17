# Architecture

## Current state

The project has a local synthetic AI4I CSV, an exploratory notebook, reusable ML code, a locally generated inference pipeline, a FastAPI service, and a React/Vite dashboard. The raw dataset and generated model artifact are ignored by Git. A validation-selected configuration has been evaluated once on the reserved test partition; container deployment does not exist.

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
       +--> src/maintsense/api.py (implemented FastAPI service)
            |-- GET /health and GET /model-info
            +-- POST /predict -> existing inference function
                 +--> frontend/ React dashboard (Vite proxy or configured CORS)

tests/            baseline, inference, and API unit tests
models/           local ignored random_forest.joblib artifact
```

The FastAPI application is implemented in `src/maintsense/api.py`. Its
lifespan loads the ignored joblib pipeline once per app process. `GET /health`
reports API/model availability, `GET /model-info` describes the
frozen Random Forest, 0.40 threshold, required features, and synthetic-data
scope, and `POST /predict` validates one feature record with Pydantic then
calls the existing `inference.predict_record` function. The API does not
implement a second preprocessor or classifier. Missing artifacts leave the
service available in degraded health state and cause predictions to return
HTTP 503; malformed requests return HTTP 422.

The React/Vite/TypeScript client is implemented in `frontend/`. Its API client
uses `VITE_API_BASE_URL` and calls the three existing routes without
reimplementing inference. The development server can proxy `/api/*` to the
backend with `MAINTSENSE_API_PROXY_TARGET`, avoiding CORS in local development.
For direct cross-origin requests, the FastAPI app accepts a comma-separated
`MAINTSENSE_CORS_ORIGINS` allow-list. The performance panel renders the
documented ML-004 validation and one-time test metrics; these are static
historical results, not results from a new browser prediction.

The local startup commands and example requests are documented in the README.
Docker deployment remains planned.

## Data and artifact flow

The traditional ML workflow is implemented in `src/maintsense/`: `baseline.py` defines the fixed split and candidates, `threshold_analysis.py` evaluates validation thresholds, `model_selection.py` freezes a choice from validation and provides the separate final test evaluation, and `model_artifact.py` fits the selected Random Forest pipeline using training rows only. The serialized scikit-learn `Pipeline` is stored as `models/random_forest.joblib` using joblib; `.gitignore` excludes it. `inference.py` loads this artifact and requires exactly `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. `Type` must be H, L, or M; numerical inputs must be finite numbers. Identifiers, the target, and `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are rejected. Inference returns `failure_probability` and `predicted_class`, using the frozen threshold 0.40. The FastAPI service in `api.py` loads the pipeline once at startup and delegates predictions to this inference function; it exposes `/health`, `/model-info`, and `/predict`. Missing artifacts are reported through degraded health and HTTP 503 prediction responses. No test rows were used to fit this artifact or evaluate inference. This is synthetic-dataset machine-failure classification, not a validated real-world early-warning system. Joblib artifacts should only be loaded from trusted sources and should be used with compatible Python/scikit-learn versions. A React/Vite client may call the API; Docker deployment is planned. These future components are not implemented.

Update this diagram and description when components or data flow change.
