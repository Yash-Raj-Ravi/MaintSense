# MaintSense

MaintSense is a traditional machine-learning application for machine-failure
classification on the synthetic UCI AI4I 2020 dataset. It is not validated as
a real-world early-warning system. Its selected Random Forest uses the
configured experimental threshold 0.40, which is not production-approved.

## Project structure

- `scripts/provision_ai4i_dataset.py`: download and verify the official dataset.
- `notebooks/`: data exploration.
- `src/maintsense/`: split, model comparison, artifact generation, inference,
  and FastAPI service.
- `tests/`: Python tests.
- `frontend/`: React + Vite + TypeScript dashboard and frontend tests.
- `docs/`: project status, architecture, decisions, and experiment history.

See [project status](docs/PROJECT_STATUS.md), [decisions](docs/DECISIONS.md),
[experiment log](docs/EXPERIMENT_LOG.md), and
[architecture](docs/ARCHITECTURE.md) for current implementation details.

## Fresh clone: prepare Python and the local artifact

Use Python 3.13.1 for the pinned environment. The checked-in constraints were
captured from the project's verified Python 3.13.1 environment. The raw data
and generated artifact are intentionally not committed.

PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt -c requirements.lock
python scripts/provision_ai4i_dataset.py
$env:PYTHONPATH = "src"
python -m maintsense.model_artifact
python -c "from maintsense.inference import load_pipeline_artifact; load_pipeline_artifact(); print('Artifact is loadable')"
```

Bash:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt -c requirements.lock
python scripts/provision_ai4i_dataset.py
export PYTHONPATH=src
python -m maintsense.model_artifact
python -c "from maintsense.inference import load_pipeline_artifact; load_pipeline_artifact(); print('Artifact is loadable')"
```

The provisioner downloads the official UCI archive only if the fixed CSV path
is absent. It checks SHA-256 `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`,
the expected schema, and 10,000 rows. If a file already exists, it validates
it and never overwrites it. The digest was measured from the CSV extracted
from the official UCI archive and matched the existing local CSV. Source:
[UCI AI4I 2020 dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset),
DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), CC BY 4.0.

`python -m maintsense.model_artifact` reuses the existing selected pipeline
and fits it only on the established stratified 60/20/20 training partition
(seed 42). It does not evaluate validation or test records. The resulting
`models/random_forest.joblib` contains preprocessing and classifier together
and is ignored by Git. Do not load untrusted joblib artifacts.

## Run the API locally

For local development, start FastAPI with reload from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn maintsense.api:app --app-dir src --reload
```

The API listens on port 8000. `GET /health` is diagnostic/liveness;
`GET /ready` returns success only when the model is loaded; `GET /model-info`
describes the inference contract; `POST /predict` returns failure probability,
class, and threshold. A missing artifact makes `/ready` return HTTP 503 and
prediction requests return HTTP 503. The API never downloads data or trains
at startup.

The production process command is `python -m maintsense.server`. It binds to
`0.0.0.0`, uses platform `PORT` when present (otherwise 8000), and does not
enable reload. `MAINTSENSE_CORS_ORIGINS` is an optional comma-separated list
of exact trusted origins. Leave it unset for the production same-origin
reverse-proxy setup.

Example `POST /predict` request:

```json
{
  "Type": "M",
  "Air temperature [K]": 300.0,
  "Process temperature [K]": 310.0,
  "Rotational speed [rpm]": 1500,
  "Torque [Nm]": 40.0,
  "Tool wear [min]": 120
}
```

Response fields are `failure_probability`, `predicted_class`, and `threshold`.
The fixed 0.40 threshold is an experimental configuration, not an approved
operating point.

## Run the frontend locally

The Vite development proxy is preserved. Start the backend as above, then:

```powershell
Set-Location frontend
npm ci
$env:VITE_API_BASE_URL = "/api"
$env:MAINTSENSE_API_PROXY_TARGET = "http://127.0.0.1:8000"
npm run dev
```

The proxy forwards `/api/*` to FastAPI and removes `/api`. For a production
frontend build, use `VITE_API_BASE_URL=/api` at build time:

```powershell
$env:VITE_API_BASE_URL = "/api"
npm ci
npm run build
```

The production Nginx configuration in `frontend/nginx.conf` serves the SPA
and forwards `/api/*` to the `api` service, stripping the prefix. Vite
`VITE_*` variables are embedded in public browser assets; do not put secrets
in them.

## Build and run with Docker Compose

First complete dataset provisioning and artifact generation above. The API
image validates that the prebuilt artifact is present and loadable; it does
not download the dataset or train during startup.

```powershell
docker compose up --build
```

Open `http://localhost:8080`. Compose waits for the API `/ready` health check
before starting the Nginx frontend. Stop with `Ctrl+C`, then run
`docker compose down`. For a direct cross-origin deployment, set
`MAINTSENSE_CORS_ORIGINS` to the exact frontend origin and rebuild/recreate
the service; wildcard origins are not enabled.

## Checks

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python -m pip check
```

From `frontend/`, run `npm test`, `npm run lint`, and `npm run build`.
`.env` files, datasets, model artifacts, virtual environments, node modules,
and generated frontend build output are excluded from Git.

## Limitations

The model uses a synthetic dataset and does not establish prediction-time
feature availability, calibrated risk, a prediction horizon, acceptable
false-alarm burden, or real-world predictive value. Historical validation and
test metrics are documented in the experiment log; they are not repeated by
the deployment workflow and must not be interpreted as production guarantees.
