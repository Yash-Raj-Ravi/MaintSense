# Project Status

Last updated: 2026-10-17

## Completed

- Git repository, project documentation, exploratory notebook, and local ignored AI4I dataset are established.
- Traditional ML comparison and validation-based selection are documented. The selected Random Forest at threshold 0.40 was evaluated once on the reserved test partition in the earlier milestone; this deployment milestone does not repeat that evaluation.
- Training-only pipeline persistence, single-record inference, FastAPI backend, and React/Vite dashboard are implemented.
- Frontend suite currently contains 12 tests; commit `44fc192` passed frontend tests, ESLint, TypeScript, and the Vite production build.
- Python dependency pins and a constraints lock have been captured from the verified Python 3.13.1 project environment.
- The official UCI CSV was downloaded independently and its measured SHA-256 matches the existing ignored CSV. The provisioner verified the fixed local copy; the existing artifact builder regenerated and load-validated the ignored artifact without evaluating validation/test partitions.
- Historical deployment-foundation checks: 37 Python tests passed, 10 frontend tests passed, ESLint and production build passed, Python compilation and `pip check` passed, and dependency dry-runs resolved against the lock. These are earlier milestone results, not a current deployment smoke test.
- Supplied production deployment facts: frontend at https://maint-sense.vercel.app and FastAPI backend at https://maintsense-api.onrender.com. Current endpoint health, deployed revisions, hosting settings, and artifact-release provenance have not been independently verified here.
- The API Docker build implementation requires a public versioned GitHub Release asset, verifies the pinned checksum, places the artifact at the expected path, and retains startup/readiness validation. Docker Compose remains a separate local/container workflow.

## In progress

- Verify the deployed Vercel/Render configuration and perform a current read-only endpoint smoke check. Confirm whether the live backend was built from the documented Release-asset workflow before updating artifact-release status.

## Planned

- Resolve prediction-time feature availability, alert horizon, calibration, and false-alarm versus missed-failure costs before any real-world early-warning claim.
- Record verified live-service behavior and configuration after checking the deployed endpoints and hosting settings.

## Known issues and limitations

- AI4I is synthetic; the application is not validated for real-world early-warning use.
- The local dataset and `models/random_forest.joblib` are excluded from Git. Docker builds fetch the existing trained artifact from a public, versioned GitHub Release asset and verify its pinned size and SHA-256; they do not need the dataset or train a model.
- The artifact uses Python/scikit-learn/joblib serialization and must be built and served using the pinned compatible runtime.
- Whether a versioned public GitHub Release asset was published and used by the live backend is unresolved from repository evidence.
- Historical note: during the deployment-foundation review, Docker Engine access was unavailable, so that review did not verify a clean build with the Release asset or a live container/reverse-proxy smoke test. The supplied deployed URLs do not independently establish those checks or the current endpoint health.
- The test partition remains the historical single evaluation from the model-selection milestone; no later evaluation is authorized or performed.

## Next actionable milestone

Confirm the deployed Vercel/Render settings and run a read-only check of `/health`, `/ready`, `/model-info`, and a valid `/predict` request. Record the checked deployment revision and artifact provenance without repeating model evaluation. Do not alter the model, threshold, or evaluation results.
