# Project Status

Last updated: 2026-10-17

## Completed

- Git repository, project documentation, exploratory notebook, and local ignored AI4I dataset are established.
- Traditional ML comparison and validation-based selection are documented. The selected Random Forest at threshold 0.40 was evaluated once on the reserved test partition in the earlier milestone; this deployment milestone does not repeat that evaluation.
- Training-only pipeline persistence, single-record inference, FastAPI backend, and React/Vite dashboard are implemented.
- Frontend suite currently contains 10 tests; prior frontend milestone passed tests, ESLint, and production build.
- Python dependency pins and a constraints lock have been captured from the verified Python 3.13.1 project environment.
- The official UCI CSV was downloaded independently and its measured SHA-256 matches the existing ignored CSV. The provisioner verified the fixed local copy; the existing artifact builder regenerated and load-validated the ignored artifact without evaluating validation/test partitions.
- Deployment checks: 34 Python tests pass, 10 frontend tests pass, ESLint and production build pass, Python compilation and `pip check` pass, and dependency dry-runs resolve against the lock.
- API/frontend Dockerfiles, Compose wiring, and the production Nginx reverse proxy are implemented. Compose configuration validation and both image builds have passed; a live container and reverse-proxy smoke test remains outstanding.

## In progress

- Live container startup and reverse-proxy smoke test remain blocked until a Docker Engine is available in the current environment.

## Planned

- Resolve prediction-time feature availability, alert horizon, calibration, and false-alarm versus missed-failure costs before any real-world early-warning claim.
- Run a deployment in a chosen hosting environment only after the deployment foundation passes review.

## Known issues and limitations

- AI4I is synthetic; the application is not validated for real-world early-warning use.
- The local dataset and `models/random_forest.joblib` are excluded from Git. Generate the artifact before building the API image.
- The artifact uses Python/scikit-learn/joblib serialization and must be built and served using the pinned compatible runtime.
- Docker images built successfully in the prior deployment check, but the Docker Engine is unavailable in the current review environment, so the live container/reverse-proxy smoke test could not be rerun here.
- The test partition remains the historical single evaluation from the model-selection milestone; no later evaluation is authorized or performed.

## Next actionable milestone

Start Docker Engine, run the API/frontend containers from the checksum-verified local dataset and training-only artifact, and confirm the `/api` reverse proxy end-to-end. Review and resolve any blockers before deployment. Do not alter the model, threshold, or evaluation results.
