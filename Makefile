.DEFAULT_GOAL := help
.PHONY: help install demo test lint eval loadtest compare corpus up down ingest query clean

VENV ?= .venv
PY ?= $(shell [ -x $(VENV)/bin/python ] && echo $(VENV)/bin/python || echo python3)
Q ?= what is a vector database
STRATEGY ?= fixed
N ?= 50000
API_URL ?= http://localhost:8000/api/v1

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Create .venv (Python 3.12 or 3.13) and install dependencies
	python3.13 -m venv $(VENV)
	$(VENV)/bin/python -m pip install -r requirements-dev.txt

demo: ## Run the whole pipeline on the sample data (offline, no API keys)
	$(PY) scripts/demo.py

test: ## Run the tests
	$(PY) -m pytest -q

lint: ## Lint with ruff
	$(PY) -m ruff check .

eval: ## Chunking comparison (fixed vs semantic vs recursive), needs the corpus
	$(PY) scripts/compare_chunking.py --docs 3000 --queries 1000

corpus: ## Download the public corpus (Simple English Wikipedia, about 157 MB)
	$(PY) scripts/fetch_corpus.py

loadtest: ## Load test, N documents (default 50000), needs the corpus
	/usr/bin/time -l $(PY) scripts/load_test.py --n $(N) --strategy $(STRATEGY)

up: ## Start the API and ChromaDB with Docker Compose
	docker compose up -d --build
	@echo "API docs: http://localhost:8000/docs"

down: ## Stop the stack and delete its volumes
	docker compose down -v

ingest: ## Ingest the sample files into the running stack (STRATEGY=fixed|semantic|recursive)
	docker compose exec api python scripts/ingest_documents.py --strategy $(STRATEGY)

query: ## Query the running stack (Q="your question")
	@curl -s -X POST $(API_URL)/query -H "Content-Type: application/json" \
		-d '{"query": "$(Q)", "n_results": 3}' | $(PY) -m json.tool

clean: ## Remove local caches
	rm -rf chroma_data .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
