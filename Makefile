UV      ?= uv
COMPOSE ?= docker compose -f docker/docker-compose.yml
CONFIG  ?= configs/default.yaml
EXPERIMENT ?= configs/experiments/full_grid.yaml

.DEFAULT_GOAL := help
.PHONY: help install dev serve up down migrate download-corpus pull-models index query \
        bench bench-smoke test test-integration lint format typecheck check clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Sync the runtime environment only
	$(UV) sync --frozen --no-dev

dev:  ## Sync every dependency group and install the pre-commit hooks
	$(UV) sync --all-groups
	$(UV) run pre-commit install
	@test -f .env || cp .env.example .env

serve:  ## Run the HTTP API locally with reload
	$(UV) run uvicorn rag_bench.api.main:app --reload --port $${PORT:-8000}

up:  ## Start the full stack (api, postgres, qdrant, ollama)
	$(COMPOSE) up -d --wait

down:  ## Stop the stack and drop its volumes
	$(COMPOSE) down -v

migrate:  ## Apply database migrations
	$(UV) run alembic upgrade head

download-corpus:  ## Fetch and cache the evaluation corpus
	$(UV) run rag-bench corpus download

pull-models:  ## Pull the default Ollama model (several GB on first run)
	$(COMPOSE) exec ollama ollama pull $${OLLAMA_MODEL:-llama3.2:3b}

index:  ## Build an index from $(CONFIG)
	$(UV) run rag-bench index build --config $(CONFIG)

query:  ## Ask a question: make query Q="..."
	$(UV) run rag-bench query "$(Q)"

bench:  ## Run the full benchmark grid
	$(UV) run rag-bench benchmark run --experiment $(EXPERIMENT)

bench-smoke:  ## Run the grid against the 10-question smoke set
	$(UV) run rag-bench benchmark run --experiment $(EXPERIMENT) --eval-set data/eval/smoke.jsonl

test:  ## Run unit tests with the coverage gate
	$(UV) run pytest --cov --cov-report=term-missing --cov-report=xml

test-integration:  ## Run integration tests (needs Docker)
	$(UV) run pytest -m integration -p no:cacheprovider

lint:  ## Check formatting and lint rules
	$(UV) run ruff format --check .
	$(UV) run ruff check .

format:  ## Apply formatting and autofixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck:  ## Type-check the source tree
	$(UV) run mypy

check: lint typecheck test  ## The full quality gate

clean:  ## Remove caches and build artefacts
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage coverage.xml dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
