# Chapter 2.2 — The task runner
.PHONY: test lint fmt

test:
	uv run pytest

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests
