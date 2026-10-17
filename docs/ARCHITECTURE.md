# Architecture

## Current state

The project currently has a local AI4I CSV, an exploratory Jupyter notebook, and a reusable Python baseline module. The raw dataset is ignored by Git. A baseline has been trained and compared on validation data, but no selected or deployed model artifact, serving API, web client, or container deployment exists.

```text
                       +--> notebooks/01_data_exploration.ipynb
                       |      descriptive inspection and visualizations
                       |
Local AI4I CSV --------+
(data/raw/, ignored)   |
                       +--> src/maintsense/baseline.py
  |-- stratified 60/20/20 split (seed 42)
  |-- train-fitted encoding/scaling pipelines
  |-- dummy and Logistic Regression validation comparison
  +-- final test partition held for later evaluation

tests/test_baseline.py   baseline unit tests
models/           generated model artifact placeholder
```

## Planned flow

The first traditional ML validation baseline is implemented in `src/maintsense/baseline.py`; it does not save a model artifact. The final test partition is reserved. A future FastAPI service may load a validated artifact and accept prediction inputs; a future React/Vite client may call that API. Docker deployment is also planned. These future components and data flows are not implemented.

Update this diagram and description when components or data flow change.
