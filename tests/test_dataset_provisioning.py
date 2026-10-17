"""Unit tests for safe UCI dataset provisioning."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import provision_ai4i_dataset as provisioning  # noqa: E402


class DatasetProvisioningTests(unittest.TestCase):
    def make_csv(self) -> bytes:
        header = ",".join(provisioning.EXPECTED_COLUMNS)
        row = ",".join(["1", "M12345", "M"] + ["1"] * 11)
        return f"{header}\n{row}\n".encode("utf-8")

    def make_archive(self, csv_bytes: bytes) -> bytes:
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("ai4i2020.csv", csv_bytes)
        return buffer.getvalue()

    def test_extracts_expected_csv_from_official_archive_shape(self):
        csv_bytes = self.make_csv()
        self.assertEqual(
            provisioning.extract_csv(self.make_archive(csv_bytes)), csv_bytes
        )

    def test_rejects_checksum_and_schema_mismatches(self):
        csv_bytes = self.make_csv()
        with patch.object(provisioning, "DATASET_SHA256", "0" * 64):
            with self.assertRaisesRegex(provisioning.DatasetProvisionError, "checksum mismatch"):
                provisioning.validate_csv(csv_bytes)

        malformed = b"wrong,columns\n1,2\n"
        with patch.object(
            provisioning, "DATASET_SHA256", sha256(malformed).hexdigest()
        ):
            with patch.object(provisioning, "EXPECTED_ROWS", 1):
                with self.assertRaisesRegex(provisioning.DatasetProvisionError, "columns"):
                    provisioning.validate_csv(malformed)

    def test_provisions_downloaded_archive_to_fixed_destination(self):
        csv_bytes = self.make_csv()
        archive_bytes = self.make_archive(csv_bytes)
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "ai4i2020.csv"
            response = BytesIO(archive_bytes)
            with (
                patch.object(provisioning, "DATASET_PATH", destination),
                patch.object(provisioning, "DATASET_SHA256", sha256(csv_bytes).hexdigest()),
                patch.object(provisioning, "EXPECTED_ROWS", 1),
                patch.object(provisioning, "urlopen", return_value=response) as opener,
            ):
                actual_path = provisioning.provision_dataset()

            self.assertEqual(actual_path, destination)
            self.assertEqual(destination.read_bytes(), csv_bytes)
            opener.assert_called_once_with(provisioning.DATASET_URL, timeout=60)

    def test_existing_dataset_is_verified_without_overwrite_or_download(self):
        csv_bytes = self.make_csv()
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "ai4i2020.csv"
            destination.write_bytes(csv_bytes)
            with (
                patch.object(provisioning, "DATASET_PATH", destination),
                patch.object(provisioning, "DATASET_SHA256", sha256(csv_bytes).hexdigest()),
                patch.object(provisioning, "EXPECTED_ROWS", 1),
                patch.object(provisioning, "urlopen") as opener,
            ):
                self.assertEqual(provisioning.provision_dataset(), destination)
            opener.assert_not_called()
            self.assertEqual(destination.read_bytes(), csv_bytes)


if __name__ == "__main__":
    unittest.main()
