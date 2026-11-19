.PHONY: setup ingest index ablation eval app phoenix redteam-baseline optimize-prompt ab-compare test lint format-check

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

# Local Phoenix UI + OTLP collector on http://localhost:6006 (run the app with PHOENIX_ENABLED=true).
phoenix:
	uvx --from arize-phoenix phoenix serve

# Prompt optimization: paid API runs (ANTHROPIC_API_KEY). The agent under test is ROUTER_MODEL;
# ANTHROPIC_MODEL writes the new policies.
BUDGET ?= 300
REPEATS ?= 3

# Baseline on the red-team set before optimizing (`-`: failed scenarios are the expected signal).
redteam-baseline:
	-$(PYTHON) eval/run_eval.py --scenarios eval/scenarios_redteam.jsonl --out eval/reports/redteam/results_baseline.jsonl --llm-mode api

# REFERENCE: baseline results of the same model (per-scenario token/step reference for the cost objectives).
REFERENCE ?= eval/reports/redteam/results_haiku55_baseline.jsonl

optimize-prompt:
	uv run --extra opt python -m astropilote.agentic.prompt_opt.optimize --budget $(BUDGET) --reference $(REFERENCE)

# Baseline vs. optimized policy: held-out red-team families + the 30-scenario regression set.
ab-compare:
	$(PYTHON) -m astropilote.agentic.prompt_ab --scenarios eval/scenarios_redteam.jsonl eval/scenarios.jsonl --split test --repeats $(REPEATS)

test:
	uv run pytest -v

lint:
	uv run ruff check src tests eval

format-check:
	uv run ruff format --check src tests eval
