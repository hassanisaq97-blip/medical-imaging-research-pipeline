.PHONY: setup test test-fast test-slow lint format curate train-small train-gpu \
        api worker docker-up docker-down data-check quickstart clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup: ## Create a virtualenv and install the project + dev dependencies.
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(VENV)/bin/pre-commit install || true

test: ## Run the full test suite (fast + slow smoke tests). Requires Redis.
	$(VENV)/bin/pytest tests/ -v

test-fast: ## Run everything except the end-to-end training/inference smoke test.
	$(VENV)/bin/pytest tests/ -v -m "not slow"

test-slow: ## Run only the end-to-end curate->train->infer->QC smoke test.
	$(VENV)/bin/pytest tests/ -v -m "slow"

lint: ## Run ruff lint + format checks.
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/ruff format --check src tests scripts

format: ## Auto-format with ruff.
	$(VENV)/bin/ruff check --fix src tests scripts
	$(VENV)/bin/ruff format src tests scripts

data-check: ## Check whether Multimodal-HC is available locally; print instructions if not.
	$(PYTHON) -m medimg_pipeline data-check

curate: ## Build the curated dataset manifest + QC report from $$DATASET_ROOT.
	$(PYTHON) -m medimg_pipeline curate --data-root $${DATASET_ROOT:?set DATASET_ROOT or pass --data-root directly}

train-small: ## Run a small, laptop-friendly training run.
	$(PYTHON) -m medimg_pipeline train --config configs/train_small.yaml

train-gpu: ## Run a full-size training run (requires a CUDA GPU).
	$(PYTHON) -m medimg_pipeline train --config configs/train_gpu.yaml

quickstart: ## Run the synthetic-data end-to-end walkthrough (see examples/quickstart.sh).
	bash examples/quickstart.sh

api: ## Run the FastAPI job-queue API locally (expects Redis on $$REDIS_URL).
	$(PYTHON) -m medimg_pipeline api

worker: ## Run an RQ worker locally (expects Redis on $$REDIS_URL).
	$(PYTHON) -m medimg_pipeline worker

docker-up: ## Start api + worker + redis via docker compose.
	docker compose up --build -d

docker-down: ## Stop and remove the docker compose services.
	docker compose down

clean: ## Remove caches and local virtualenv.
	rm -rf $(VENV) .pytest_cache .ruff_cache **/__pycache__
