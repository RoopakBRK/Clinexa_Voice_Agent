API := apps/api
UV_API := uv run --project $(API)
PORT ?= 8000
TEXT ?= I've had a headache for three days and it's getting worse.

.PHONY: help install dev test lint format typecheck check simulate tunnel

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

tunnel:  ## Expose the local API to Twilio via ngrok
	ngrok http $(PORT)
