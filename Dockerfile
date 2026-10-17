FROM python:3.13.1-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

# Set by the local Compose environment or Render service configuration. The
# URL points to a public, versioned GitHub Release asset; it is not a secret.
ARG MODEL_ARTIFACT_URL

COPY scripts/fetch_model_artifact.py ./scripts/fetch_model_artifact.py
RUN if [ -z "$MODEL_ARTIFACT_URL" ]; then \
      echo "Build failed: set MODEL_ARTIFACT_URL to the published GitHub Release asset URL." >&2; \
      exit 1; \
    fi && \
    python scripts/fetch_model_artifact.py \
      "$MODEL_ARTIFACT_URL" /app/models/random_forest.joblib

COPY requirements-runtime.txt requirements.lock ./
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    --requirement requirements-runtime.txt \
    --constraint requirements.lock

COPY src/ ./src/

# Fail the image build if the verified release artifact cannot be loaded.
RUN python -c "from maintsense.inference import load_pipeline_artifact; load_pipeline_artifact()"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/ready', timeout=3)"

CMD ["python", "-m", "maintsense.server"]
