"""Fetch the published MaintSense model artifact and verify its exact bytes."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

EXPECTED_SHA256 = "d22f14131f57f2ac8feb534e0551c50936f0d9da7528f08abc81e79e70e545f3"
EXPECTED_SIZE_BYTES = 4_872_018
EXPECTED_RELEASE_PREFIX = "/Yash-Raj-Ravi/MaintSense/releases/download/"
EXPECTED_ASSET_NAME = "random_forest.joblib"


class ArtifactProvisionError(RuntimeError):
    """Raised when a release artifact cannot be safely retrieved or verified."""


def validate_release_url(url: str) -> None:
    """Accept only HTTPS release-download URLs for this repo and model asset."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith(EXPECTED_RELEASE_PREFIX)
        or not parsed.path.endswith(f"/{EXPECTED_ASSET_NAME}")
    ):
        raise ArtifactProvisionError(
            "MODEL_ARTIFACT_URL must be the HTTPS GitHub Release download URL "
            "for this repository's random_forest.joblib asset."
        )

    release_tag = parsed.path[len(EXPECTED_RELEASE_PREFIX) : -len(EXPECTED_ASSET_NAME) - 1]
    if not release_tag:
        raise ArtifactProvisionError("The GitHub Release URL must include a release tag.")


def validate_artifact_bytes(artifact_bytes: bytes) -> None:
    """Reject artifacts unless both their measured size and SHA-256 match."""
    if len(artifact_bytes) != EXPECTED_SIZE_BYTES:
        raise ArtifactProvisionError(
            "Model artifact size mismatch: "
            f"expected {EXPECTED_SIZE_BYTES} bytes, received {len(artifact_bytes)}."
        )
    actual_sha256 = sha256(artifact_bytes).hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        raise ArtifactProvisionError(
            "Model artifact SHA-256 mismatch: "
            f"expected {EXPECTED_SHA256}, received {actual_sha256}."
        )


def fetch_model_artifact(url: str, destination: Path) -> Path:
    """Download and validate the pinned release artifact without overwriting."""
    validate_release_url(url)
    destination = Path(destination)

    if destination.exists():
        try:
            validate_artifact_bytes(destination.read_bytes())
        except OSError as exc:
            raise ArtifactProvisionError(
                "Cannot read the existing model artifact; refusing to overwrite it."
            ) from exc
        return destination

    try:
        with urlopen(url, timeout=60) as response:
            artifact_bytes = response.read(EXPECTED_SIZE_BYTES + 1)
    except Exception:
        # Do not include request details, headers, or exception text in build logs.
        raise ArtifactProvisionError(
            "Could not download the model artifact from the configured GitHub Release."
        ) from None

    validate_artifact_bytes(artifact_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as artifact_file:
            artifact_file.write(artifact_bytes)
    except FileExistsError as exc:
        raise ArtifactProvisionError(
            "A model artifact appeared during download; refusing to overwrite it."
        ) from exc
    except OSError as exc:
        raise ArtifactProvisionError("Could not save the verified model artifact.") from exc
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="public HTTPS GitHub Release asset URL")
    parser.add_argument("destination", type=Path, help="path for the verified model")
    args = parser.parse_args()

    try:
        path = fetch_model_artifact(args.url, args.destination)
    except ArtifactProvisionError as exc:
        parser.exit(1, f"Model artifact provisioning failed: {exc}\n")

    print(
        "Verified model artifact: "
        f"{EXPECTED_SIZE_BYTES} bytes, SHA-256 {EXPECTED_SHA256}"
    )
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
