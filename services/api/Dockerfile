# Model service + web UI (FastAPI / Uvicorn). Loads models:/<name>@champion from MLflow.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /srv
COPY services/api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# the shared package (custom estimator classes, DB schema, registry helpers)
COPY pyproject.toml README.md /tmp/pkg/
COPY src /tmp/pkg/src
RUN pip install --no-deps /tmp/pkg && rm -rf /tmp/pkg

COPY services/api/app ./app

RUN useradd --create-home --uid 1000 app && chown -R app /srv
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# One worker is plenty for millisecond scikit-learn inference; scale out with replicas
# (each replica polls the registry for a new @champion every REGISTRY_POLL_SECONDS).
CMD ["sh", "-c", "uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1}"]
