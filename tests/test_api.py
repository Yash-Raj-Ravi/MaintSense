"""HTTP-level tests for the FastAPI prediction backend."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maintsense.api import SYNTHETIC_SCOPE, create_app  # noqa: E402
from maintsense.baseline import split_dataset  # noqa: E402
from maintsense.inference import (  # noqa: E402
    DECISION_THRESHOLD,
    load_pipeline_artifact,
    predict_record,
)
from maintsense.model_artifact import train_and_save_artifact  # noqa: E402
from test_baseline import sample_frame  # noqa: E402


VALID_REQUEST = {
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 120,
}


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.artifact_path = Path(cls.temp_dir.name) / "random_forest.joblib"
        splits = split_dataset(sample_frame())
        cls.pipeline = train_and_save_artifact(
            splits.X_train, splits.y_train, cls.artifact_path
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        self.client_context = TestClient(create_app(self.artifact_path))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_health_reports_available_model(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"status": "ok", "model_available": True}
        )

    def test_ready_reports_loaded_model(self):
        response = self.client.get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"status": "ready", "model_available": True}
        )

    def test_model_info_describes_frozen_inference_contract(self):
        response = self.client.get("/model-info")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["model_type"], "RandomForestClassifier")
        self.assertEqual(payload["threshold"], 0.40)
        self.assertEqual(
            payload["required_features"],
            [
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ],
        )
        self.assertEqual(payload["scope"], SYNTHETIC_SCOPE)

    def test_predict_calls_existing_inference_contract(self):
        expected = predict_record(VALID_REQUEST, self.pipeline)

        response = self.client.post("/predict", json=VALID_REQUEST)
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(
            payload["failure_probability"], expected["failure_probability"]
        )
        self.assertEqual(payload["predicted_class"], expected["predicted_class"])
        self.assertEqual(payload["threshold"], DECISION_THRESHOLD)

    def test_missing_and_extra_fields_are_rejected(self):
        missing = dict(VALID_REQUEST)
        del missing["Torque [Nm]"]
        self.assertEqual(self.client.post("/predict", json=missing).status_code, 422)

        extra = {**VALID_REQUEST, "UDI": 12}
        self.assertEqual(self.client.post("/predict", json=extra).status_code, 422)

    def test_invalid_category_is_rejected(self):
        request = {**VALID_REQUEST, "Type": "X"}

        response = self.client.post("/predict", json=request)

        self.assertEqual(response.status_code, 422)

    def test_non_finite_numbers_are_rejected(self):
        for value in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(value=value):
                response = self.client.post(
                    "/predict",
                    content=json.dumps(
                        {**VALID_REQUEST, "Air temperature [K]": value}
                    ),
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("input", response.json()["detail"][0])
                self.assertNotIn("Traceback", response.text)

    def test_invalid_numeric_string_is_rejected_by_shared_inference_validation(self):
        request = {**VALID_REQUEST, "Air temperature [K]": "300"}
        response = self.client.post("/predict", json=request)

        self.assertEqual(response.status_code, 422)

    def test_artifact_is_loaded_once_when_app_lifespan_starts(self):
        with patch(
            "maintsense.api.load_pipeline_artifact", wraps=load_pipeline_artifact
        ) as loader:
            app = create_app(self.artifact_path)
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                client.post("/predict", json=VALID_REQUEST)
                client.post("/predict", json=VALID_REQUEST)

            loader.assert_called_once_with(self.artifact_path)

    def test_configured_cors_origin_allows_browser_preflight(self):
        app = create_app(self.artifact_path, cors_origins=["https://ui.example"])
        with TestClient(app) as client:
            response = client.options(
                "/predict",
                headers={
                    "Origin": "https://ui.example",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"], "https://ui.example"
        )

    def test_wildcard_cors_origin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "wildcard origins are not allowed"):
            create_app(self.artifact_path, cors_origins=["*"])


class MissingArtifactApiTests(unittest.TestCase):
    def test_missing_artifact_is_visible_and_prediction_returns_503(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            absent_artifact = Path(temp_dir) / "missing.joblib"
            with TestClient(create_app(absent_artifact)) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(
                    health.json(), {"status": "degraded", "model_available": False}
                )
                readiness = client.get("/ready")
                self.assertEqual(readiness.status_code, 503)
                self.assertEqual(
                    readiness.json(),
                    {"detail": "Prediction service is not ready."},
                )
                self.assertNotIn(str(absent_artifact), readiness.text)
                self.assertNotIn("Traceback", readiness.text)
                self.assertFalse(client.get("/model-info").json()["model_available"])

                prediction = client.post("/predict", json=VALID_REQUEST)

        self.assertEqual(prediction.status_code, 503)
        self.assertIn("model is unavailable", prediction.json()["detail"])
        self.assertNotIn(str(absent_artifact), prediction.text)
        self.assertNotIn("Traceback", prediction.text)


if __name__ == "__main__":
    unittest.main()
