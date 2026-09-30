API := apps/api
UV_API := uv run --project $(API)
PORT ?= 8000
TEXT ?= I've had a headache for three days and it's getting worse.

.PHONY: help install dev test lint format typecheck check simulate tunnel requirements ingest index query

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install API dependencies (Python 3.12 via uv)
	cd $(API) && uv sync

dev:  ## Run the API with auto-reload
	cd $(API) && uv run uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port $(PORT)

test:  ## Run the test suite
	cd $(API) && uv run pytest

lint:  ## Lint and check formatting
	cd $(API) && uv run ruff check . ../../scripts && uv run ruff format --check . ../../scripts

format:  ## Auto-fix lint issues and format
	cd $(API) && uv run ruff check --fix . ../../scripts && uv run ruff format . ../../scripts

typecheck:  ## Static type check (mypy --strict)
	cd $(API) && uv run mypy app

check: lint typecheck test  ## Everything CI runs

simulate:  ## Simulate a phone call against the local API (TEXT="...")
	$(UV_API) python scripts/simulate_call.py --text "$(TEXT)" --api http://localhost:$(PORT)

ingest:  ## Chunk WHO PDFs in data/ into data/processed/chunks.jsonl
	cd $(API) && uv run python -m app.rag.ingestion $(ARGS)

index:  ## Embed chunks and upsert into Qdrant (ARGS="--local" for the embedded index)
	cd $(API) && uv run python -m app.rag.retrieval $(ARGS) index

query:  ## Dense search: make query Q="cough for 5 days" ARGS="--local"
	cd $(API) && uv run python -m app.rag.retrieval $(ARGS) query "$(Q)"

tunnel:  ## Expose the local API to Twilio via ngrok
	ngrok http $(PORT)

requirements:  ## Regenerate pinned requirements.txt from the uv lockfile
	cd $(API) && { printf '# Generated from apps/api/uv.lock by `make requirements`. Do not edit by hand:\n# add dependencies in apps/api/pyproject.toml (uv add ...) and regenerate.\n# Install: pip install -r requirements.txt   (Python 3.12)\n\n'; uv export --format requirements-txt --no-hashes --no-emit-project --no-header --quiet | grep -v '^\s*#'; } > ../../requirements.txt
