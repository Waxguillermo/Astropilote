.PHONY: setup ingest index ablation eval app test lint format-check

PYTHON := uv run python

setup:
	uv sync --extra dev

ingest:
	$(PYTHON) -m astropilote.ingest.run_ingest

index:
	$(PYTHON) -m astropilote.index.build_index

ablation:
	$(PYTHON) -m astropilote.ingest.compare_chunking
	$(PYTHON) -m astropilote.retrieval.ablation

eval:
	$(PYTHON) eval/run_eval.py --questions eval/golden.jsonl --out eval/

app:
	$(PYTHON) -m astropilote.app.main

test:
	uv run pytest -v

lint:
	uv run ruff check src tests eval

format-check:
	uv run ruff format --check src tests eval
