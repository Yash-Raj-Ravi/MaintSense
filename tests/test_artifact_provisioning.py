"""Tests for fetching the pinned model artifact from a GitHub Release."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import fetch_model_artifact as artifact_fetcher  # noqa: E402


class ModelArtifactFetchTests(unittest.TestCase):
    RELEASE_URL = (
        "https://github.com/Yash-Raj-Ravi/MaintSense/releases/download/"
        "chosen-tag/random_forest.joblib"
    )

    def test_downloads_and_saves_checksum_verified_artifact(self):
        artifact_bytes = b"known fitted pipeline bytes"
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "models/random_forest.joblib"
            with (
                patch.object(
                    artifact_fetcher, "EXPECTED_SHA256", sha256(artifact_bytes).hexdigest()
                ),
                patch.object(artifact_fetcher, "EXPECTED_SIZE_BYTES", len(artifact_bytes)),
                patch.object(
                    artifact_fetcher, "urlopen", return_value=BytesIO(artifact_bytes)
                ) as opener,
            ):
                result = artifact_fetcher.fetch_model_artifact(
                    self.RELEASE_URL, destination
                )

            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), artifact_bytes)
            opener.assert_called_once_with(self.RELEASE_URL, timeout=60)

    def test_rejects_checksum_mismatch_without_creating_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "models/random_forest.joblib"
            with (
                patch.object(artifact_fetcher, "EXPECTED_SHA256", "0" * 64),
                patch.object(artifact_fetcher, "EXPECTED_SIZE_BYTES", 3),
                patch.object(
                    artifact_fetcher, "urlopen", return_value=BytesIO(b"bad")
                ),
            ):
                with self.assertRaisesRegex(
                    artifact_fetcher.ArtifactProvisionError, "SHA-256 mismatch"
                ):
                    artifact_fetcher.fetch_model_artifact(
                        self.RELEASE_URL, destination
                    )

            self.assertFalse(destination.exists())

    def test_rejects_untrusted_urls_without_network_access(self):
        with patch.object(artifact_fetcher, "urlopen") as opener:
            with self.assertRaisesRegex(
                artifact_fetcher.ArtifactProvisionError, "GitHub Release download URL"
            ):
                artifact_fetcher.fetch_model_artifact(
                    "https://example.com/random_forest.joblib",
                    Path("unused/random_forest.joblib"),
                )
        opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
