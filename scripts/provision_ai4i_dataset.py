"""Provision and validate the official UCI AI4I 2020 CSV."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
)
DATASET_PATH = (
    REPOSITORY_ROOT
    / "data/raw/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv"
)
DATASET_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"
EXPECTED_ROWS = 10_000
EXPECTED_COLUMNS = (
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)


class DatasetProvisionError(RuntimeError):
    """Raised when the downloaded or local dataset is not the expected file."""


def validate_csv(csv_bytes: bytes) -> None:
    """Verify the observed UCI checksum, header, and row count."""
    actual_sha256 = sha256(csv_bytes).hexdigest()
    if actual_sha256 != DATASET_SHA256:
        raise DatasetProvisionError(
            "AI4I CSV checksum mismatch: "
            f"expected {DATASET_SHA256}, received {actual_sha256}."
        )

    try:
        reader = csv.reader(StringIO(csv_bytes.decode("utf-8-sig", errors="strict")))
        header = next(reader)
        row_count = sum(1 for _ in reader)
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise DatasetProvisionError("AI4I CSV is empty or is not valid UTF-8 CSV.") from exc

    if tuple(header) != EXPECTED_COLUMNS:
        raise DatasetProvisionError("AI4I CSV columns do not match the official schema.")
    if row_count != EXPECTED_ROWS:
        raise DatasetProvisionError(
            f"AI4I CSV has {row_count} data rows; expected {EXPECTED_ROWS}."
        )


def extract_csv(archive_bytes: bytes) -> bytes:
    """Read the single named CSV from the official ZIP without extracting paths."""
    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            members = [
                item
                for item in archive.infolist()
                if PurePosixPath(item.filename).name == "ai4i2020.csv"
                and not item.is_dir()
            ]
            if len(members) != 1:
                raise DatasetProvisionError(
                    "Official UCI archive must contain exactly one ai4i2020.csv."
                )
            return archive.read(members[0])
    except BadZipFile as exc:
        raise DatasetProvisionError("The UCI download is not a valid ZIP archive.") from exc


def provision_dataset() -> Path:
    """Download if absent; validate existing data and never overwrite it."""
    if DATASET_PATH.exists():
        try:
            validate_csv(DATASET_PATH.read_bytes())
        except OSError as exc:
            raise DatasetProvisionError(f"Cannot read existing dataset: {exc}") from exc
        print(f"Verified existing dataset: {DATASET_PATH}")
        return DATASET_PATH

    try:
        with urlopen(DATASET_URL, timeout=60) as response:
            csv_bytes = extract_csv(response.read())
    except DatasetProvisionError:
        raise
    except Exception as exc:
        raise DatasetProvisionError(
            "Could not download the official UCI dataset. Check network access and retry."
        ) from exc

    validate_csv(csv_bytes)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DATASET_PATH.open("xb") as destination:
            destination.write(csv_bytes)
    except FileExistsError as exc:
        raise DatasetProvisionError(
            f"Dataset appeared during provisioning; refusing to overwrite {DATASET_PATH}."
        ) from exc
    print(f"Downloaded and verified dataset: {DATASET_PATH}")
    return DATASET_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        path = provision_dataset()
    except DatasetProvisionError as exc:
        parser.exit(1, f"Dataset provisioning failed: {exc}\n")
    print(f"SHA-256: {DATASET_SHA256}")
    print(f"Source: {DATASET_URL}")
    print(f"Validated {EXPECTED_ROWS} rows and {len(EXPECTED_COLUMNS)} columns.")


if __name__ == "__main__":
    main()
