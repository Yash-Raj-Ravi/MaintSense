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
- Production deployment smoke checks supplied for this update: the Vercel frontend at https://maint-sense.vercel.app and Render FastAPI backend at https://maintsense-api.onrender.com are deployed. `/health`, `/ready`, and `/model-info` returned HTTP 200 with the expected availability/readiness/model metadata; CORS preflight for the Vercel origin returned HTTP 200 with that origin allowed; and a prediction through the dashboard was manually confirmed. These checks do not verify the deployed artifact's exact checksum or provenance.
- The API Docker build implementation requires a public versioned GitHub Release asset, verifies the pinned checksum, places the artifact at the expected path, and retains startup/readiness validation. Docker Compose remains a separate local/container workflow.

## In progress

- Verify whether the live backend was built from the documented Release-asset workflow before updating artifact-release status. The deployed artifact's exact checksum and provenance remain unverified.

## Planned

- Resolve prediction-time feature availability, alert horizon, calibration, and false-alarm versus missed-failure costs before any real-world early-warning claim.
- Verify the deployed artifact checksum/provenance and hosting settings when evidence is available.

## Known issues and limitations

- AI4I is synthetic; the application is not validated for real-world early-warning use.
- The local dataset and `models/random_forest.joblib` are excluded from Git. Docker builds fetch the existing trained artifact from a public, versioned GitHub Release asset and verify its pinned size and SHA-256; they do not need the dataset or train a model.
- The artifact uses Python/scikit-learn/joblib serialization and must be built and served using the pinned compatible runtime.
- Whether a versioned public GitHub Release asset was published and used by the live backend is unresolved; the exact checksum/provenance inside the deployed Render instance has not been independently verified.
- Historical note: during the deployment-foundation review, Docker Engine access was unavailable, so that review did not verify a clean build with the Release asset or a live container/reverse-proxy smoke test. The later supplied live checks establish endpoint and dashboard behavior, but do not establish the image's build inputs or artifact provenance.
- The test partition remains the historical single evaluation from the model-selection milestone; no later evaluation is authorized or performed.

## Next actionable milestone

Obtain verifiable evidence for the artifact used by the live Render service (such as its release asset identity and checksum). Record it without repeating model evaluation. Do not alter the model, threshold, or evaluation results.
