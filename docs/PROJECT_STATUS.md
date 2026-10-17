# Project Status

Last updated: 2026-10-17

## Completed

- Git repository, project documentation, exploratory notebook, and local ignored AI4I dataset are established.
- Traditional ML comparison and validation-based selection are documented. The selected Random Forest at threshold 0.40 was evaluated once on the reserved test partition in the earlier milestone; this deployment milestone does not repeat that evaluation.
- Training-only pipeline persistence, single-record inference, FastAPI backend, and React/Vite dashboard are implemented.
- Frontend suite currently contains 10 tests; prior frontend milestone passed tests, ESLint, and production build.
- Python dependency pins and a constraints lock have been captured from the verified Python 3.13.1 project environment.
- The official UCI CSV was downloaded independently and its measured SHA-256 matches the existing ignored CSV. The provisioner verified the fixed local copy; the existing artifact builder regenerated and load-validated the ignored artifact without evaluating validation/test partitions.
- Deployment checks: 37 Python tests pass, 10 frontend tests pass, ESLint and production build pass, Python compilation and `pip check` pass, and dependency dry-runs resolve against the lock.
- API Docker build retrieves the existing model from a versioned GitHub Release asset, verifies its pinned checksum, and retains startup/readiness validation. Compose and Render frontend/backend deployment instructions are documented.

## In progress

- Manual publication of the existing model artifact as a versioned GitHub Release asset is pending; then validate a clean Docker build and Render service startup.

## Planned

- Resolve prediction-time feature availability, alert horizon, calibration, and false-alarm versus missed-failure costs before any real-world early-warning claim.
- Run a deployment in a chosen hosting environment only after the deployment foundation passes review.

## Known issues and limitations

- AI4I is synthetic; the application is not validated for real-world early-warning use.
- The local dataset and `models/random_forest.joblib` are excluded from Git. Docker builds fetch the existing trained artifact from a public, versioned GitHub Release asset and verify its pinned size and SHA-256; they do not need the dataset or train a model.
- The artifact uses Python/scikit-learn/joblib serialization and must be built and served using the pinned compatible runtime.
- A versioned public GitHub Release asset must be created manually before a clean Docker build can succeed.
- The Docker Engine is unavailable in the current environment, so the updated GitHub-asset build and live container/reverse-proxy smoke test could not be run here.
- The test partition remains the historical single evaluation from the model-selection milestone; no later evaluation is authorized or performed.

## Next actionable milestone

Manually publish the existing model artifact as a versioned GitHub Release asset, then build the API image from a clean clone using its download URL and run the backend on Render with `/ready` health checks. Configure the separate frontend Static Site with the backend origin and exact CORS origin, then verify predictions end-to-end. Do not alter the model, threshold, or evaluation results.
