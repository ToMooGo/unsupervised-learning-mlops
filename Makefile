.PHONY: help install lint format test test-all quick train eval monitor monitor-up api up up-quick down logs relabel report notebooks clean

PY ?= python

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:         ## local dev environment (Python 3.12+)
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

lint:            ## ruff lint + format check
	ruff check . && ruff format --check .

format:          ## auto-format
	ruff check --fix . && ruff format .

test:            ## unit + API tests (~10 s)
	$(PY) -m pytest -m "not integration" -q

test-all:        ## + Prefect/MLflow integration test (~1-2 min)
	$(PY) -m pytest -q

quick:           ## train -> evaluate -> deploy with the fast config (local SQLite MLflow)
	$(PY) run_flow.py --config configs/quick_flow_config.yaml

train:           ## full experiment: regenerates reports/ (about 5 min on 2 CPUs)
	$(PY) run_flow.py --config configs/full_flow_config.yaml

eval:            ## re-evaluate the newest registered models against the quality gates
	$(PY) run_flow.py --config configs/full_flow_config.yaml --flow eval

monitor:         ## drift check on logged predictions
	$(PY) run_flow.py --config configs/monitor_flow_config.yaml

api:             ## serve the API + UI locally on :8000 (needs a trained @champion)
	cd services/api && uvicorn app.main:create_app --factory --reload --port 8000

DC = HOST_UID=$$(id -u) HOST_GID=$$(id -g) docker compose

up:              ## whole stack in Docker: Postgres, MLflow, Prefect, API, then the pipeline
	$(DC) up --build

up-quick:        ## same, with the 1-minute config
	FLOW_CONFIG=configs/quick_flow_config.yaml $(DC) up --build

monitor-up:      ## hourly drift check as a Prefect deployment (docker)
	$(DC) --profile monitor up -d monitor

down:            ## stop the stack (add -v to wipe volumes)
	docker compose down

logs:
	docker compose logs -f api pipeline

relabel:         ## retrain part B from the labels typed in the web UI
	LABEL_SOURCE=human $(DC) run --rm pipeline

report:          ## build the LaTeX technical report (pdflatex, run twice)
	cd reports/technical_report && pdflatex -interaction=nonstopmode technical_report.tex && pdflatex -interaction=nonstopmode technical_report.tex

notebooks:       ## re-execute the notebooks in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf mlruns mlflow.db app.db .pytest_cache .ruff_cache
