# Architecture

## Current system

Implemented components and flow:

```text
Official UCI AI4I ZIP
  -> scripts/provision_ai4i_dataset.py
     (SHA-256, CSV schema, and row-count validation)
  -> data/raw/.../ai4i2020.csv (local, Git-ignored)
  -> baseline.py (fixed stratified 60/20/20 split, seed 42)
  -> model_artifact.py (fit selected Random Forest on train rows only)
  -> models/random_forest.joblib (local, Git-ignored)
  -> inference.py -> api.py (/predict delegates to existing inference)
  -> frontend React/Vite dashboard

Production containers:
  frontend image: Vite build with VITE_API_BASE_URL=/api
       -> Nginx serves SPA and strips /api from proxied requests
       -> API image: python -m maintsense.server (PORT, no reload)
            -> loaded joblib Pipeline
```

`/health` is a diagnostic/liveness endpoint and reports model availability.
`/ready` returns HTTP 200 only after the artifact is loaded, otherwise 503.
`/model-info` reports the frozen model contract, and `/predict` validates
the request then calls `inference.predict_record`. The API never trains or
downloads data during startup. It loads the artifact once per app lifespan.

The frontend development proxy remains configured in `frontend/vite.config.ts`
through `MAINTSENSE_API_PROXY_TARGET`. Production uses
`frontend/nginx.conf`: same-origin `/api/*` requests are forwarded to FastAPI
with the `/api` prefix removed. CORS is disabled unless exact trusted origins
are provided with `MAINTSENSE_CORS_ORIGINS`.

## ML and data scope

The existing code implements exploration, traditional model comparison,
validation threshold analysis, frozen validation selection, one historical
final test evaluation, and persisted single-record inference. The selected
configuration remains Random Forest at the experimental threshold 0.40.
Artifact generation uses the original training partition only. No new
evaluation or model fitting on validation/test rows is part of deployment.

The provisioner downloads UCI dataset 601 from the official source, checks
the extracted CSV SHA-256, exact 14-column schema, and 10,000-row count, and
refuses to overwrite a file already at the fixed destination. The local CSV
and model artifact are excluded from Git. Python runtime packages are pinned
separately from notebook/test dependencies; `requirements.lock` constrains
the installed environment.

This remains machine-failure classification on a synthetic dataset, not a
validated real-world early-warning system. Feature timing, alert horizon,
probability calibration, operating costs, and external validation are open.

## Deployment files

- `Dockerfile`: API image; requires the generated artifact at build time and
  validates that it can be loaded.
- `frontend/Dockerfile`: production static bundle and Nginx reverse proxy.
- `compose.yaml`: connects the services and waits for API readiness.
- `.dockerignore` and `frontend/.dockerignore`: omit local data, environments,
  dependencies, and generated build output from image contexts.

Update this file when the implemented data flow or service topology changes.
