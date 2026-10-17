# Architecture

## Current state

The project currently has a local AI4I CSV, an exploratory Jupyter notebook, and a minimal Python package location. The raw dataset is ignored by Git. No model training, serving API, web client, or container deployment is implemented.

```text
Local AI4I CSV (data/raw/, Git-ignored)
                 |
                 v
notebooks/01_data_exploration.ipynb
                 |
                 +-- descriptive inspection and visualizations (executed; findings in EXPERIMENT_LOG.md)

src/maintsense/   reusable Python code placeholder
tests/            automated test placeholder
models/           generated model artifact placeholder
```

## Planned flow

After exploratory review, define eligible prediction-time features and a validation design. Reusable preprocessing and model code should live in `src/maintsense/`, with tests in `tests/` and generated model artifacts in `models/`. A future FastAPI service may load a validated artifact and accept prediction inputs; a future React/Vite client may call that API. Docker deployment is also planned. These future components and data flows are not implemented, and no model artifact currently exists.

Update this diagram and description when components or data flow change.
