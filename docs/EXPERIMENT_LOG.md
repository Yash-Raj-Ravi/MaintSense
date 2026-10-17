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

## ML-002: Tree model comparison on the existing validation split

- **Status:** Completed validation comparison; the reserved test partition remains unevaluated.
- **Objective:** Compare Random Forest and Gradient Boosting against the three existing baseline candidates under the same validation protocol.
- **Dataset and split:** Same local AI4I dataset, selected features, stratified 60/20/20 split, and random state 42 as ML-001. Train/validation/test sizes remain 6,000/2,000/2,000. Only training and validation partitions were used for fitting and model comparison.
- **Preprocessing:** Categorical `Type` one-hot encoded with unknown categories ignored; five numerical operating measurements passed through without scaling, which is unnecessary for tree splits. Preprocessing remains in each candidate pipeline and is fit on training rows only. Dense one-hot output is used for tree estimator compatibility. No oversampling.
- **Features:** Unchanged from ML-001: `Type`, air temperature, process temperature, rotational speed, torque, and tool wear. Identifiers and failure-mode indicators remain excluded.
- **Models and parameters:** All five candidates were rerun: most-frequent `DummyClassifier`; standard Logistic Regression (`max_iter=1000`, seed 42); balanced Logistic Regression (`class_weight="balanced"`); `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)`; and `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)`. These are fixed comparison settings, not tuned hyperparameters. Classifiers use their default 0.5 prediction threshold; no threshold optimization was performed.
- **Validation class balance:** 68/2,000 positives (3.4%) for every model. Confusion matrix rows are actual `[0, 1]`, columns predicted `[0, 1]`.

  | Model | Precision | Recall | F1 | Average precision | Accuracy | Confusion matrix |
  |---|---:|---:|---:|---:|---:|---|
  | Most-frequent dummy | 0.0000 | 0.0000 | 0.0000 | 0.0340 | 0.9660 | `[[1932, 0], [68, 0]]` |
  | Logistic Regression | 0.7647 | 0.1912 | 0.3059 | 0.4054 | 0.9705 | `[[1928, 4], [55, 13]]` |
  | Logistic Regression, balanced | 0.1425 | 0.8235 | 0.2430 | 0.3680 | 0.8255 | `[[1595, 337], [12, 56]]` |
  | Random Forest | 0.9459 | 0.5147 | 0.6667 | 0.7772 | 0.9825 | `[[1930, 2], [33, 35]]` |
  | Gradient Boosting | 0.8235 | 0.6176 | 0.7059 | 0.7457 | 0.9825 | `[[1923, 9], [26, 42]]` |

- **Observations:** On this fixed validation split, Random Forest has the highest precision and average precision; Gradient Boosting has the highest recall and F1 among these five fixed-threshold candidates. Both tree candidates exceed the baselines on validation average precision, but this is a single validation comparison and does not establish real-world performance. The fixed threshold yields different false-alarm and missed-failure trade-offs. Accuracy alone remains insufficient given the 3.4% positive prevalence.
- **Next steps:** Confirm prediction-time availability and define the operational lead time and costs for false alarms versus missed failures. Any further candidate selection or threshold work must remain on validation data; keep the test partition untouched until criteria are frozen.

## ML-003: Validation-based probability threshold analysis

- **Status:** Completed validation analysis; no threshold selected for deployment and no test evaluation performed.
- **Objective:** Show how Random Forest and Gradient Boosting validation precision, recall, F1, false positives, and false negatives change over a fixed probability threshold grid.
- **Dataset and protocol:** Same AI4I rows, feature set, stratified 60/20/20 split, and seed 42 as ML-001/ML-002. Both tree models are refit on the training partition and produce probabilities only for the 2,000-row validation partition. The validation set has 68 failures (3.4%).
- **Threshold method:** Evaluate cutoffs from 0.05 through 0.95 in 0.05 increments, including 0.50. Scores greater than or equal to a cutoff predict failure. The values 0.25, 0.50, and 0.75 below are representative low/default/high cutoffs to make the trade-off legible; they are not a selected optimum. No threshold optimization or hyperparameter tuning was performed.
- **Results:** Confusion matrix rows are actual `[0, 1]`, columns predicted `[0, 1]`.

  | Model | Threshold | Precision | Recall | F1 | False positives | False negatives | Confusion matrix |
  |---|---:|---:|---:|---:|---:|---:|---|
  | Random Forest | 0.25 | 0.6098 | 0.7353 | 0.6667 | 32 | 18 | `[[1900, 32], [18, 50]]` |
  | Random Forest | 0.50 | 0.9459 | 0.5147 | 0.6667 | 2 | 33 | `[[1930, 2], [33, 35]]` |
  | Random Forest | 0.75 | 1.0000 | 0.1912 | 0.3210 | 0 | 55 | `[[1932, 0], [55, 13]]` |
  | Gradient Boosting | 0.25 | 0.6267 | 0.6912 | 0.6573 | 28 | 21 | `[[1904, 28], [21, 47]]` |
  | Gradient Boosting | 0.50 | 0.8235 | 0.6176 | 0.7059 | 9 | 26 | `[[1923, 9], [26, 42]]` |
  | Gradient Boosting | 0.75 | 0.9630 | 0.3824 | 0.5474 | 1 | 42 | `[[1931, 1], [42, 26]]` |

- **Precision-recall curves:** Generated from validation scores. The validation average precision is 0.7772 for Random Forest and 0.7457 for Gradient Boosting; the dashed reference is the 0.034 validation prevalence. See [validation precision-recall curve](figures/validation_pr_curve.png).
- **Observations:** Lowering Random Forest's threshold from 0.50 to 0.25 raises recall from 0.5147 to 0.7353, while false positives rise from 2 to 32. For Gradient Boosting, lowering 0.50 to 0.25 raises recall from 0.6176 to 0.6912 and false positives from 9 to 28. At 0.75, both models reduce false positives further while missing more failures. These are validation trade-offs only. Probability calibration was not assessed, so scores should not be interpreted as calibrated failure risk. No cutoff is recommended absent a prediction horizon and operational costs for missed failures and false alerts.
- **Next steps:** Establish the alert use case and the relative cost of missed failures versus false alarms, then agree on a decision rule using validation evidence. Keep the reserved test partition untouched until the model and decision criteria are fixed.

## ML-004: Validation model selection and final test evaluation

- **Status:** Completed; validation selection frozen and reserved test evaluated once.
- **Objective:** Select one existing candidate and threshold using only validation results, then perform one final evaluation on the reserved test partition.
- **Dataset and split:** Same local synthetic AI4I 2020 machine-failure classification dataset, leakage-safe features, stratified 60/20/20 split, and random state 42. Train/validation/test sizes are 6,000/2,000/2,000. Test results are not used for model or threshold tuning.
- **Selection criterion:** Compare all five existing candidates over the already established 19 thresholds (0.05–0.95, step 0.05). Maximize validation F1; tie-break by higher recall, fewer false positives, then proximity to 0.50, followed by stable candidate/grid order. This is a transparent benchmark rule, not an operational cost model or claim of universal optimality.
- **Validation comparison:** The best row per candidate under the same rule is shown below. Confusion matrices use actual `[0, 1]` rows and predicted `[0, 1]` columns.

  | Candidate | Threshold | Precision | Recall | F1 | FP | FN | Accuracy | Average precision | Confusion matrix |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---|
  | Most-frequent dummy | 0.50 | 0.0000 | 0.0000 | 0.0000 | 0 | 68 | 0.9660 | 0.0340 | `[[1932, 0], [68, 0]]` |
  | Logistic Regression | 0.20 | 0.4000 | 0.4412 | 0.4196 | 45 | 38 | 0.9585 | 0.4054 | `[[1887, 45], [38, 30]]` |
  | Logistic Regression (balanced) | 0.75 | 0.2617 | 0.5735 | 0.3594 | 110 | 29 | 0.9305 | 0.3680 | `[[1822, 110], [29, 39]]` |
  | Random Forest | 0.40 | 0.8824 | 0.6618 | 0.7563 | 6 | 23 | 0.9855 | 0.7772 | `[[1926, 6], [23, 45]]` |
  | Gradient Boosting | 0.35 | 0.7759 | 0.6618 | 0.7143 | 13 | 23 | 0.9820 | 0.7457 | `[[1919, 13], [23, 45]]` |

- **Frozen choice:** Random Forest at threshold 0.40, because it has the highest validation F1 across the five candidates and fixed grid. Gradient Boosting has the same validation recall at its own best row but lower F1 and more false positives. No operational costs were assumed.
- **Final test results:** After the choice was frozen, Random Forest was fitted on the 6,000 training rows and evaluated once on the 2,000 reserved test rows. Precision 0.8367, recall 0.6029, F1 0.7009, average precision 0.7817, accuracy 0.9825, 8 false positives, and 27 false negatives. Confusion matrix (actual `[0, 1]` rows, predicted `[0, 1]` columns): `[[1924, 8], [27, 41]]`. The test results were recorded as a final estimate; no model or threshold changes followed.
- **Limitations:** Threshold selection on one validation partition can overfit that partition. The test result is one estimate on this fixed random split and does not justify further tuning. AI4I is synthetic, and this is machine-failure classification, not a validated real-world early-warning system. Feature timing, prediction horizon, calibration, and deployment conditions remain unresolved.
