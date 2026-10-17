# Experiment Log

Record only work that was actually run. No deep learning models have been trained.

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

## ML-001: Traditional ML validation baselines

- **Status:** Completed validation comparison; final test partition remains untouched.
- **Objective:** Establish a reproducible majority-class reference and compare Logistic Regression with and without balanced class weights.
- **Dataset:** The same local AI4I CSV used in EDA-001 (10,000 rows).
- **Preprocessing:** `Type` one-hot encoded with unknown categories ignored; five numerical operating measurements standardized. Transformations are inside scikit-learn pipelines and are fit only on the 6,000 training rows. No oversampling.
- **Features:** `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. Excluded `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`.
- **Split:** Stratified 60% train / 20% validation / 20% final test; random state 42. Split sizes were 6,000 / 2,000 / 2,000. Only validation predictions and labels were used for metrics and comparison; the test partition was not evaluated.
- **Models and parameters:** `DummyClassifier(strategy="most_frequent")`; `LogisticRegression(max_iter=1000, random_state=42)`; and the same Logistic Regression with `class_weight="balanced"`. Logistic Regression predictions use the estimator's default 0.5 threshold; no threshold was tuned.
- **Validation results:** Confusion matrix rows are actual `[0, 1]`, columns predicted `[0, 1]`.

  | Model | Precision | Recall | F1 | Average precision (PR-AUC) | Accuracy | Confusion matrix |
  |---|---:|---:|---:|---:|---:|---|
  | Most-frequent dummy | 0.0000 | 0.0000 | 0.0000 | 0.0340 | 0.9660 | `[[1932, 0], [68, 0]]` |
  | Logistic Regression | 0.7647 | 0.1912 | 0.3059 | 0.4054 | 0.9705 | `[[1928, 4], [55, 13]]` |
  | Logistic Regression, balanced | 0.1425 | 0.8235 | 0.2430 | 0.3680 | 0.8255 | `[[1595, 337], [12, 56]]` |

- **Observations:** Failures are only 3.4% of the validation rows. The dummy gets 96.6% accuracy by predicting no failures and misses all 68 failures. Unweighted Logistic Regression has higher accuracy (97.05%) but detects only 13/68 failures. Balanced Logistic Regression detects 56/68 but predicts 337 false positives, with lower precision and accuracy. The unweighted model has higher average precision here than the balanced variant. Thus accuracy alone obscures the recall/false-alarm trade-off; model usefulness depends on the acceptable operational cost and warning horizon. These are validation results, not final test performance or deployment evidence.
- **Next steps:** Define the prediction horizon, false-alarm burden, and missed-failure cost. Confirm prediction-time availability of all selected inputs. Freeze model/threshold criteria before any final test evaluation; do not infer that class weighting is automatically superior.
