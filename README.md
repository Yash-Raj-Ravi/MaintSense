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

## Publish the existing model artifact for deployment

The model and dataset are not committed. Deployment does not need the raw
dataset and must not retrain the classifier. The backend image downloads the
existing model from a versioned GitHub Release and verifies it against the
repository-pinned SHA-256 before validating that the pipeline loads.

The local artifact currently expected by this checkout is 4,872,018 bytes
with SHA-256
`d22f14131f57f2ac8feb534e0551c50936f0d9da7528f08abc81e79e70e545f3`.
Before publishing it, verify your local file with:

```powershell
Get-FileHash models/random_forest.joblib -Algorithm SHA256
(Get-Item models/random_forest.joblib).Length
```

In GitHub, open the MaintSense repository's **Releases** page, draft a
release using a tag you choose for this exact artifact version, upload the
unchanged `models/random_forest.joblib` file, and publish the release. Do not
regenerate or edit the model to create the asset. Copy the asset's HTTPS
download URL; the build accepts only this repository's Release download URL
and the pinned checksum prevents a different file from being used. No release
or asset has been created by this project setup.

## Deploy on Render

Create two services from the same pushed repository:

1. **Backend Web Service:** select Docker, use the repository root as the
   root directory and `Dockerfile` as the Dockerfile path. Set
   `MODEL_ARTIFACT_URL` to the exact public GitHub Release asset URL copied
   above. Set `MAINTSENSE_CORS_ORIGINS` to the exact HTTPS origin of the
   frontend Static Site. Do not set it to `*`. Set the Render health check
   path to `/ready`; Render supplies `PORT`, which the server honors.
2. **Frontend Static Site:** set root directory to `frontend`, build command
   to `npm ci && npm run build`, and publish directory to `dist`. Set
   `VITE_API_BASE_URL` to the backend's public HTTPS origin. This is public
   browser configuration, not a secret.

The release asset must be publicly downloadable because the Docker build
fetches it without credentials. The SHA-256 is pinned in
`scripts/fetch_model_artifact.py`; a missing URL, failed download, wrong file,
or checksum mismatch stops the build. The verified artifact is placed at
`/app/models/random_forest.joblib`, where the existing startup/readiness
validation loads it. The dataset is not part of the image, and Docker build
or API startup does not train a model.

After these source changes are pushed and the release asset is published, test
the build from a clean clone (replace the placeholder with the URL copied from
the actual release):

```powershell
git clone https://github.com/Yash-Raj-Ravi/MaintSense.git MaintSense-clean-build
Set-Location MaintSense-clean-build
$env:MODEL_ARTIFACT_URL = "<exact public release asset download URL>"
docker build --pull --no-cache `
  --build-arg "MODEL_ARTIFACT_URL=$env:MODEL_ARTIFACT_URL" `
  -t maintsense-api:clean .
docker run --rm -d --name maintsense-api-check `
  -e PORT=10000 -p 10000:10000 maintsense-api:clean
Invoke-RestMethod http://127.0.0.1:10000/ready
docker stop maintsense-api-check
Remove-Item Env:MODEL_ARTIFACT_URL
```

Render Free web services use an ephemeral filesystem and spin down after
inactivity. The model is included in the built image, so it remains available
after a restart without a persistent disk; the backend may take time to wake
before the first prediction request.

## Build and run with Docker Compose

After publishing the model asset, set its copied public URL in the shell and
run Compose. The URL is not a credential; do not put tokens or secrets in it.

```powershell
$env:MODEL_ARTIFACT_URL = "<paste the exact GitHub Release asset download URL>"
docker compose up --build
```

Open `http://localhost:8080`. Compose waits for the API `/ready` health check
before starting the Nginx frontend. Stop with `Ctrl+C`, then run
`docker compose down`. Compose passes the URL as a required Docker build
argument. The build helper verifies the SHA-256 and refuses to overwrite an
existing destination artifact.

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
