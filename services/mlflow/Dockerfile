# MLflow tracking server + model registry (PostgreSQL backend, artifacts on a volume).
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 MLFLOW_DISABLE_AGENT_HINT=1
RUN pip install mlflow==3.16.1 psycopg2-binary==2.9.13 && mkdir -p /mlartifacts

EXPOSE 5000
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --workers 2 \
  --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
  --artifacts-destination /mlartifacts --serve-artifacts \
  --allowed-hosts ${MLFLOW_ALLOWED_HOSTS:-mlflow,mlflow:*,localhost,localhost:*,127.0.0.1:*}"]
