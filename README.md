# MaintSense

A traditional machine-learning project for machine-failure
classification on the synthetic AI4I 2020 dataset. Its offline
results are not a validated real-world early-warning system.

## Project Goals

- Explore and analyze predictive maintenance data.
- Develop a reliable classification pipeline.
- Compare multiple machine learning algorithms.
- Evaluate models using appropriate classification metrics.
- Build a FastAPI prediction backend.
- Develop an interactive React frontend.
- Test and deploy the complete application.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI
- React, Vite, and TypeScript
- CSS
- Docker

## Project Status

Exploration, traditional model comparison, validation-based selection,
one final test evaluation, and local single-record inference are complete
for the synthetic dataset. See
[project status](docs/PROJECT_STATUS.md) for the current milestone.

## Project Documentation

- [Project status](docs/PROJECT_STATUS.md)
- [Technical decisions](docs/DECISIONS.md)
- [Experiment log](docs/EXPERIMENT_LOG.md)
- [Architecture](docs/ARCHITECTURE.md)

## Dataset

This project uses the AI4I 2020 Predictive Maintenance Dataset
from the UCI Machine Learning Repository.

Dataset source:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset contains simulated industrial operating data.
Results should not be interpreted as proof of real-world
industrial reliability.

## Local Prediction API

Install the project dependencies and create the ignored inference artifact
from the established training partition:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m maintsense.model_artifact
.\.venv\Scripts\python.exe -m uvicorn maintsense.api:app --app-dir src --reload
```

The API listens at `http://127.0.0.1:8000`. The generated artifact is kept at
`models/random_forest.joblib` and is not tracked by Git. Open
`http://127.0.0.1:8000/docs` for the interactive API schema.

Example request (the field names match the model feature names):

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

Send it to `POST /predict`. With the local saved artifact, the sample request
above returned:

```json
{
  "failure_probability": 0.0,
  "predicted_class": 0,
  "threshold": 0.4
}
```

That example is a single illustrative request result, not an evaluation metric.

`GET /health` reports whether the API loaded its model. `GET /model-info`
returns the model type, threshold, required features, and synthetic-data
scope. A missing or unreadable artifact leaves health and model-info
available; prediction requests then receive HTTP 503. Invalid requests
receive HTTP 422. These endpoints classify records in a synthetic dataset;
they are not validated real-world early-warning predictions.

## React Dashboard

The React + Vite + TypeScript dashboard lives under `frontend/`. Start the
backend in one PowerShell terminal using the commands above. In a second
terminal, from the repository root, install and run the frontend:

```powershell
Set-Location frontend
npm ci
$env:VITE_API_BASE_URL = "/api"
$env:MAINTSENSE_API_PROXY_TARGET = "http://127.0.0.1:8000"
npm run dev
```

Open the Vite URL printed in the terminal (normally port 5173). In development,
Vite proxies `/api/health`, `/api/model-info`, and `/api/predict` to the backend
and removes the `/api` prefix. This same-origin proxy avoids local browser CORS
configuration. `VITE_API_BASE_URL` controls the API base in the frontend; for a
direct backend URL, set it to the backend origin and leave the proxy target
unset. For direct cross-origin access, the backend must allow the exact UI
origin: set `MAINTSENSE_CORS_ORIGINS` in the backend terminal, for example to
`http://localhost:5173`. For deployed use, set this to the specific trusted
frontend origin or put both services behind a same-origin reverse proxy. The
backend does not allow arbitrary origins by default.

Frontend checks can be run from `frontend/` with `npm test`, `npm run lint`,
and `npm run build`. Local `.env.local`, `node_modules`, build output, and test
coverage are ignored by Git. Do not put secrets in Vite `VITE_*` variables;
they are compiled into browser assets.
