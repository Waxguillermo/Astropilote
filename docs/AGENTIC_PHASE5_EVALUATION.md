# AstroPilote — Phase 5: 30 state-graded Agentic Workflows scenarios

## Purpose and exact command

Implement the professor's separate **Agentic Workflows** evaluator without breaking the old **Generative AI** question evaluator. This is an isolated, deterministic benchmark for the bounded French/English routing workflow implemented in phases 1–4B. It uses the **real Observatory MCP stdio server** to exercise action reads/writes and the **real checkpointed LangGraph** for bookings and HITL cancellation. No live observatory, payment, refund or actual message is involved.

From the project root, after installation:

```bash
uv run python eval/run_eval.py --scenarios eval/scenarios.jsonl --out eval/
```

Outputs: `eval/results.jsonl` (per-scenario JSONL) and `eval/summary_agentic.json`. The original script's `--questions` mode is preserved; it is still selected when `--scenarios` is omitted:

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ --llm-mode offline
```

**Important:** the original Generative AI runner also uses `eval/results.jsonl` and that file is tracked in Git. The new evaluator copies an existing legacy result to `eval/results_generative_ai_backup.jsonl` (if such a backup does not exist), *then* writes the mandatory `eval/results.jsonl`. Before committing, avoid staging generated evaluation results and inspect `git status --short`. If needed, use `--out eval/agentic_run/` for non-destructive, exploratory reruns; the rubric's exact command must nevertheless work.

## Installation

On private branch `v1-agentic`, after commit `b1ec8b5` (Phase 4B):

```bash
cd ~/Developer/Astropilote
git status --short
unzip -n ~/Desktop/Astropilote_Agentic_Phase5_Evaluation.zip -d .
python scripts/apply_agentic_eval.py
# Repeatable; a second execution reports 'Already patched'.
```

The ZIP includes only **new** files. The patch script then modifies only two insertion points in `eval/run_eval.py`: parser option `--scenarios` and an early branch in `main()`. It checks exact anchors and parses the resulting Python before writing; it does not replace the historical 500+ line file. Review the diff:

```bash
git diff -- eval/run_eval.py
```

## Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_agentic_scenarios.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check eval/run_eval.py src/astropilote/agentic/scenario_eval.py \
  tests/test_agentic_scenarios.py scripts/apply_agentic_eval.py
```

Then run the **exact command** above and inspect the JSON summary. A nonzero exit code indicates one or more failed scenarios; this is deliberate. Do **not** commit as "30/30 passed" before confirming the output on your Mac. The actual phase-4B baseline was **181 passed, 1 skipped, 7 subtests passed**.

## Scenario format and distribution

`eval/scenarios.jsonl` contains exactly **30** lines: 14 `normal`, 5 `ambiguous`, 5 `impossible` and 6 `adversarial`.

Each test case includes `id`, `category`, `turns` (one to five structured, trusted-actor turns), `max_steps`, `forbid_calls`, `expect_tools`, `expect_responses`, and `expect_unchanged`. Some include `expect_state` (partial expected canonical SQLite snapshot), `expect_booking_count`, `expect_memory` (per actor), `operator_decision` (human approval in **benchmark metadata**, never inferred from user text), and `inject_tool_response` (untrusted MCP read-result text).

The grader:

1. Creates **fresh SQLite observatory, memory and checkpoint DBs** in a unique temporary directory **for each scenario**, without altering `data/backend/`.
2. Resets `ObservatoryStore` from `seeds/observatory_demo.json`.
3. Calls `MemoryConversationalAgent.handle()` with the actual model-free router and a real MCP stdio gateway. Bookings/cancellations use `ObservatoryWorkflow` and LangGraph.
4. If and only if trusted scenario metadata specifies an operator decision, first verifies that the booking state **did not change before HITL approval**, then calls `ObservatoryWorkflow.resume()`.
5. Independently reads the **actual SQLite backend** via `get_state()` and checks expected rows by stable IDs, booleans, the full unchanged snapshot when requested, the booking count, per-user memory, returned statuses, tool call sequence, forbidden tool calls and step budgets.
6. Writes per-scenario pass/fail, failure reason, full state snapshot, ordered tool calls (approval secret **redacted**), step count, latency, stop reason, category and injection markers; computes pass rates by category.

## Adversarial tests and honest limits

Adversarial cases include negated requests, attempts to approve via chat, and malicious instructions inserted into a **copy of an MCP read result** (`list_telescopes`, `list_slots`, `get_booking`). The injected text is placed only into name/note fields **after a real read call**, never in a write result or trusted policy. The evaluator fails a case if the specified injection was not actually exercised. This tests whether the deterministic dispatcher acts upon untrusted tool text rather than the trusted input.

**Limitations to disclose at the defense:** the Phase-4B router is still deterministic and not a general LLM planner, so the evaluator uses **no LLM** and reports measured `tokens_in=0`, `tokens_out=0`, `cost_usd=0.0`. These are true zeros **for this benchmark**, not missing production LLM telemetry. The `--scenarios` results cannot be interpreted as proving a general autonomous LLM agent is injection-safe, nor that the separate original astronomy path has been evaluated under this benchmark. The evaluation includes controlled benchmark operator approvals, not end-user authentication. For production use, add authenticated operator roles, tested fault injection/timeouts, and full model telemetry if an LLM route is integrated.

## Commit only after passing tests

The file `eval/results.jsonl` is tracked and will be modified by a benchmark run; **do not accidentally stage it**. Stage only source and the scenario dataset:

```bash
git status --short
git diff --check
git add eval/run_eval.py eval/scenarios.jsonl \
  src/astropilote/agentic/scenario_eval.py \
  scripts/apply_agentic_eval.py tests/test_agentic_scenarios.py \
  docs/AGENTIC_PHASE5_EVALUATION.md
git diff --cached --stat
git commit -m "feat(agentic): add 30 isolated state-based evaluation scenarios"
git remote -v   # origin must be Astropilote_review (PRIVATE)
git push origin v1-agentic
```

Any generated local `results_generative_ai_backup.jsonl`, `summary_agentic.json`, test caches and data databases should be reviewed and **not staged** as source code.
