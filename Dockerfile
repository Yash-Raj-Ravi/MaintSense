FROM python:3.13.1-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

COPY requirements-runtime.txt requirements.lock ./
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    --requirement requirements-runtime.txt \
    --constraint requirements.lock

COPY src/ ./src/
COPY models/random_forest.joblib ./models/random_forest.joblib

# Fail the image build if the provisioned training-only artifact is absent or invalid.
RUN python -c "from maintsense.inference import load_pipeline_artifact; load_pipeline_artifact()"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/ready', timeout=3)"

CMD ["python", "-m", "maintsense.server"]
