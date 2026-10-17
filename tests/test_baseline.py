"""Unit tests for the reproducible baseline split and preprocessing."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maintsense.baseline import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    build_models,
    evaluate_validation,
    split_dataset,
    validation_metrics,
)
from maintsense.threshold_analysis import (  # noqa: E402
    SELECTED_THRESHOLDS,
    THRESHOLDS,
    evaluate_threshold,
    evaluate_threshold_grid,
    validation_precision_recall_data,
    validation_probabilities,
)
from maintsense.model_selection import (  # noqa: E402
    evaluate_selected_test,
    select_validation_configuration,
)


def sample_frame(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "UDI": range(rows),
            "Product ID": [f"P{i}" for i in range(rows)],
            "Type": ["L", "M", "H", "L", "M"] * (rows // 5),
            "Air temperature [K]": rng.normal(300, 2, rows),
            "Process temperature [K]": rng.normal(310, 2, rows),
            "Rotational speed [rpm]": rng.normal(1500, 100, rows),
            "Torque [Nm]": rng.normal(40, 8, rows),
            "Tool wear [min]": rng.integers(0, 250, rows),
            TARGET: [0] * 90 + [1] * 10,
            "TWF": [0] * 90 + [1] * 10,
            "HDF": [0] * rows,
            "PWF": [0] * rows,
            "OSF": [0] * rows,
            "RNF": [0] * rows,
        }
    )
    return frame


class BaselineTests(unittest.TestCase):
    def test_features_exclude_identifiers_and_failure_modes(self):
        self.assertEqual(CATEGORICAL_FEATURES, ["Type"])
        self.assertEqual(len(NUMERICAL_FEATURES), 5)
        self.assertNotIn("UDI", FEATURES)
        self.assertNotIn("Product ID", FEATURES)
        for column in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
            self.assertNotIn(column, FEATURES)

    def test_split_is_reproducible_stratified_and_60_20_20(self):
        frame = sample_frame()
        first = split_dataset(frame)
        second = split_dataset(frame)

        self.assertEqual(
            (len(first.X_train), len(first.X_validation), len(first.X_test)),
            (60, 20, 20),
        )
        self.assertTrue(first.X_train.equals(second.X_train))
        self.assertTrue(first.X_validation.equals(second.X_validation))
        self.assertTrue(first.X_test.equals(second.X_test))
        self.assertAlmostEqual(first.y_train.mean(), 0.1)
        self.assertAlmostEqual(first.y_validation.mean(), 0.1)
        self.assertAlmostEqual(first.y_test.mean(), 0.1)
        self.assertTrue(set(first.X_train.index).isdisjoint(first.X_validation.index))
        self.assertTrue(set(first.X_train.index).isdisjoint(first.X_test.index))

    def test_preprocessing_fits_on_training_data_and_ignores_new_category(self):
        splits = split_dataset(sample_frame())
        model = build_models()["logistic_regression"]
        model.fit(splits.X_train, splits.y_train)

        preprocessor = model.named_steps["preprocessor"]
        scaler = preprocessor.named_transformers_["numerical"]
        expected_means = splits.X_train[NUMERICAL_FEATURES].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected_means)
        transformed = model.named_steps["preprocessor"].transform(
            pd.DataFrame(
                {
                    "Type": ["unseen-type"],
                    **{column: [splits.X_train[column].mean()] for column in NUMERICAL_FEATURES},
                }
            )
        )
        self.assertEqual(transformed.shape[0], 1)

    def test_all_five_candidates_run_on_validation_and_report_imbalance(self):
        splits = split_dataset(sample_frame())
        results = evaluate_validation(splits)

        self.assertEqual(
            set(results),
            {
                "dummy_most_frequent",
                "logistic_regression",
                "logistic_regression_balanced",
                "random_forest",
                "gradient_boosting",
            },
        )
        for metrics in results.values():
            self.assertEqual(metrics["positive_support"], 2)
            self.assertAlmostEqual(metrics["positive_prevalence"], 0.1)
            self.assertEqual(len(metrics["confusion_matrix"]), 2)
            for metric in [
                "precision",
                "recall",
                "f1",
                "average_precision",
                "accuracy",
            ]:
                self.assertIn(metric, metrics)

    def test_validation_metrics_include_required_values(self):
        metrics = validation_metrics(
            pd.Series([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            np.array([0.1, 0.4, 0.7, 0.9]),
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertIn("average_precision", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("accuracy", metrics)

    def test_threshold_metrics_use_inclusive_cutoff_and_count_errors(self):
        metrics = evaluate_threshold(
            pd.Series([0, 0, 1, 1]),
            np.array([0.2, 0.5, 0.7, 0.9]),
            threshold=0.5,
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 0.8)

    def test_threshold_grid_includes_default_and_representative_cutoffs(self):
        self.assertEqual(len(THRESHOLDS), 19)
        self.assertIn(0.5, THRESHOLDS)
        self.assertEqual(SELECTED_THRESHOLDS, (0.25, 0.5, 0.75))
        results = evaluate_threshold_grid(
            pd.Series([0, 1]),
            np.array([0.4, 0.6]),
        )
        self.assertEqual(len(results), len(THRESHOLDS))

    def test_tree_validation_probabilities_and_pr_curve_use_validation_rows(self):
        splits = split_dataset(sample_frame())
        probabilities = validation_probabilities(splits)

        self.assertEqual(set(probabilities), {"random_forest", "gradient_boosting"})
        for scores in probabilities.values():
            self.assertEqual(len(scores), len(splits.y_validation))
            self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
            curve = validation_precision_recall_data(splits.y_validation, scores)
            self.assertGreaterEqual(curve["average_precision"], 0)
            self.assertLessEqual(curve["average_precision"], 1)

    def test_validation_selection_compares_all_candidates_and_fixed_grid(self):
        splits = split_dataset(sample_frame())
        result = select_validation_configuration(splits)

        self.assertEqual(len(result["validation_results"]), 5 * len(THRESHOLDS))
        selected = result["selected"]
        self.assertIn(selected["model"], build_models())
        self.assertIn(selected["threshold"], THRESHOLDS)
        self.assertEqual(
            selected,
            max(
                result["validation_results"],
                key=lambda row: (
                    row["f1"],
                    row["recall"],
                    -row["false_positives"],
                    -abs(row["threshold"] - 0.50),
                ),
            ),
        )

    def test_selected_test_evaluator_returns_required_metrics(self):
        splits = split_dataset(sample_frame())
        result = evaluate_selected_test(
            splits,
            model_name="gradient_boosting",
            threshold=0.5,
        )

        self.assertEqual(result["model"], "gradient_boosting")
        self.assertEqual(result["threshold"], 0.5)
        for metric in [
            "precision",
            "recall",
            "f1",
            "average_precision",
            "accuracy",
            "confusion_matrix",
            "false_positives",
            "false_negatives",
        ]:
            self.assertIn(metric, result)
        self.assertEqual(
            result["false_positives"], result["confusion_matrix"][0][1]
        )
        self.assertEqual(
            result["false_negatives"], result["confusion_matrix"][1][0]
        )


if __name__ == "__main__":
    unittest.main()
