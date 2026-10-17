"""Tests for saved-pipeline loading and single-record inference."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maintsense.baseline import FEATURES, split_dataset  # noqa: E402
from maintsense.inference import (  # noqa: E402
    DECISION_THRESHOLD,
    load_pipeline_artifact,
    predict_from_artifact,
    predict_record,
)
from maintsense.model_artifact import train_and_save_artifact  # noqa: E402
from test_baseline import sample_frame  # noqa: E402


VALID_RECORD = {
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 120,
}


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.artifact_path = Path(cls.temp_dir.name) / "random_forest.joblib"
        splits = split_dataset(sample_frame())
        cls.pipeline = train_and_save_artifact(
            splits.X_train,
            splits.y_train,
            cls.artifact_path,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_valid_record_returns_probability_and_thresholded_class(self):
        result = predict_from_artifact(VALID_RECORD, self.artifact_path)

        self.assertGreaterEqual(result["failure_probability"], 0)
        self.assertLessEqual(result["failure_probability"], 1)
        self.assertIn(result["predicted_class"], [0, 1])
        self.assertEqual(
            result["predicted_class"],
            int(result["failure_probability"] >= DECISION_THRESHOLD),
        )

    def test_missing_unexpected_and_forbidden_inputs_are_rejected(self):
        missing = dict(VALID_RECORD)
        del missing["Torque [Nm]"]
        with self.assertRaisesRegex(ValueError, "Missing required"):
            predict_from_artifact(missing, self.artifact_path)

        for excluded in [
            "UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"
        ]:
            with self.subTest(excluded=excluded):
                record = {**VALID_RECORD, excluded: 1}
                with self.assertRaisesRegex(ValueError, "Excluded"):
                    predict_from_artifact(record, self.artifact_path)

        with self.assertRaisesRegex(ValueError, "Unexpected"):
            predict_from_artifact({**VALID_RECORD, "unexpected": 1}, self.artifact_path)

    def test_invalid_values_are_rejected(self):
        for value in [float("nan"), float("inf"), "300", True]:
            with self.subTest(value=value):
                record = {**VALID_RECORD, "Air temperature [K]": value}
                with self.assertRaisesRegex(ValueError, "finite number"):
                    predict_from_artifact(record, self.artifact_path)

        with self.assertRaisesRegex(ValueError, "Type must be one of"):
            predict_from_artifact({**VALID_RECORD, "Type": "X"}, self.artifact_path)

    def test_artifact_loading_and_pre_post_save_predictions_match(self):
        loaded = load_pipeline_artifact(self.artifact_path)
        before_save = predict_record(VALID_RECORD, self.pipeline)
        after_load = predict_record(VALID_RECORD, loaded)

        self.assertIn("preprocessor", loaded.named_steps)
        self.assertIn("classifier", loaded.named_steps)
        self.assertEqual(list(loaded.feature_names_in_), FEATURES)
        self.assertEqual(before_save["predicted_class"], after_load["predicted_class"])
        np.testing.assert_allclose(
            before_save["failure_probability"],
            after_load["failure_probability"],
            rtol=0,
            atol=0,
        )

    def test_missing_artifact_has_actionable_error(self):
        missing_path = Path(self.temp_dir.name) / "absent.joblib"
        with self.assertRaisesRegex(FileNotFoundError, "run python -m maintsense.model_artifact"):
            load_pipeline_artifact(missing_path)


if __name__ == "__main__":
    unittest.main()
