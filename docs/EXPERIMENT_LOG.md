# Experiment Log

Record only work that was actually run. No model training or evaluation results exist yet.

## EDA-001: AI4I 2020 exploratory notebook

- **Status:** Completed exploratory analysis; no model trained.
- **Objective:** Inspect schema, data quality, target distribution, feature distributions and failure relationships, and possible leakage before choosing a modeling setup.
- **Dataset:** Local AI4I 2020 CSV at `data/raw/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv` (10,000 rows and 14 columns, confirmed by reading the CSV; this is not an analytical finding).
- **Preprocessing:** None; notebook reads the CSV directly. No rows or columns are intentionally transformed for modeling.
- **Features:** Exploratory views only. `Machine failure` is the target; `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are flagged as possible leakage. No final feature set selected.
- **Model:** None; model training is out of scope.
- **Parameters:** Not applicable.
- **Evaluation metrics:** None; no model evaluation performed.
- **Results:** Notebook executed from the project root using `.venv`. All 14 columns had zero missing values; there were zero exact duplicate rows and zero duplicate `Product ID` values. The target had 9,661 non-failure rows and 339 failure rows (3.39% failures; 96.61% non-failures). The five plotted continuous features were air temperature, process temperature, rotational speed, torque, and tool wear; plots and descriptive summaries are saved in the executed notebook. Observed numerical ranges were 295.3–304.5 K, 305.7–313.8 K, 1,168–2,886 rpm, 3.8–76.6 Nm, and 0–253 min, respectively.
- **Observations:** `TWF`, `HDF`, `PWF`, and `OSF` were positive only when `Machine failure` was 1 (46/46, 115/115, 95/95, and 98/98 positive rows, respectively). `RNF` was positive on 18 non-failure rows and 1 failure row (19 positives total). These indicators are highly suspicious as predictors because they directly describe failure modes; their availability before a maintenance decision is unknown. `Machine failure` is imbalanced, so accuracy alone would be misleading as a measure of operational usefulness. Descriptive type-level failure rates were H: 21/1,003 (2.09%), L: 235/6,000 (3.92%), and M: 83/2,997 (2.77%); these are dataset summaries, not causal effects or future performance estimates.
- **Next steps:** Establish feature availability at prediction time, define a useful alert horizon and operational costs, and select a deployment-relevant evaluation split before training any model.
