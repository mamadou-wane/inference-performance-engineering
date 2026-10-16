.PHONY: lint test lock

# lint and test install only the dev group from uv.lock and fail if uv.lock is out of date.
lint:
	uv run --locked --only-group dev ruff check src tests

# tests/needs_torch needs PyTorch and Transformers: uv run --locked pytest -m "not gpu and not checkpoint".
test:
	uv run --locked --only-group dev pytest -m "not gpu and not checkpoint" --ignore=tests/needs_torch

# Dependency changes are a separate step: edit pyproject.toml, then update the lock.
lock:
	uv lock
