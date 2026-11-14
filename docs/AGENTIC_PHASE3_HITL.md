# Agentic Workflows — Phase 3: checkpointed LangGraph + MCP + human approval

## Scope

This phase adds a **typed, deterministic action workflow** beside AstroPilote's existing astronomy-chat LangGraph. It uses real MCP clients for observatory reads/writes and optionally Sky geocoding; it does **not** yet add LLM-driven natural-language action selection to `agent/graph.py`. That integration belongs to the next phase, and should not be claimed as complete in the oral defense.

The dangerous action `cancel_booking` is **not authorized by a model-provided `approved=true`**. The Observatory MCP server verifies a random expiring capability issued only by the trusted local controller **after** a LangGraph `interrupt()` has been resumed by a human operator. The capability binds the exact booking and user, is used once, is stored hashed, and is invalidated by a backend reset. No capability or secret is ever saved in graph state or user-facing tool traces.

This is a **local classroom simulation**. The CLI trusts whoever can execute commands on the host. Before exposing approval via FastAPI or the internet, add authenticated operator identity, roles, CSRF protection, and request authorization. Do not let an LLM call the approval-resume interface or the `CancellationApprovals.issue` method. Backend cancellation is irreversible in the simulated state; no real refund or payment occurs.

## Installation

Start on branch `v1-agentic`, after Phase 2 has been committed, with a clean worktree:

```bash
cd ~/Developer/Astropilote
git status --short
unzip -o ~/Desktop/Astropilote_Agentic_Phase3_HITL.zip -d .
uv add 'langgraph-checkpoint-sqlite>=3,<4'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_agentic_approval.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check src/astropilote/agentic src/astropilote/backend src/astropilote/mcp tests/test_agentic_approval.py
```

The ZIP replaces **exactly two pre-existing files** (`backend/store.py`, `mcp/observatory_server.py`) and creates six new files. Review `git diff` before committing. Avoid extracting a ZIP from an untrusted source. Make sure `uv.lock` is committed together with `pyproject.toml` after `uv add`.

## Demonstrate a normal booking (no approval required)

```bash
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json
uv run python -m astropilote.agentic.demo start \
  --thread phase3-book-demo --action book --user alice --slot S-01
uv run python -m astropilote.agentic.demo inspect --thread phase3-book-demo
```

Expected: `completed`, `list_slots` then `create_booking`, booking `BK-002` in the **actual** SQLite snapshot.

To exercise both MCP servers in one workflow, optionally add `--place Fontainebleau` to a **new** booking thread; this first queries `list_slots`, then `geocode` via the Sky MCP server, then `create_booking` via the Observatory MCP server. Use `EVAL_OFFLINE_MODE=1` in the environment for fixture-based offline reproducibility.

## Demonstrate a refusal (state unchanged)

Reset first; the checkpoint store is separate from the backend, so use a **fresh thread ID** on every new run:

```bash
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json
uv run python -m astropilote.agentic.demo start \
  --thread phase3-reject-demo --action cancel --user astro-club --booking BK-001
uv run python -m astropilote.agentic.demo inspect --thread phase3-reject-demo
uv run python -m astropilote.agentic.demo resume --thread phase3-reject-demo --reject
```

Expected: initial status `awaiting_human_approval`, then `human_rejected`. `BK-001` remains `confirmed`, `paid=true`, `refunded=false`.

## Demonstrate a human-approved cancellation (persist across processes)

```bash
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json
uv run python -m astropilote.agentic.demo start \
  --thread phase3-approve-demo --action cancel --user astro-club --booking BK-001
# Separate CLI invocation below restores checkpoints from SQLite:
uv run python -m astropilote.agentic.demo resume --thread phase3-approve-demo --approve
uv run python -m astropilote.agentic.demo inspect --thread phase3-approve-demo
```

Expected: `completed` and backend booking `BK-001` is now `cancelled`, `refunded=true`. A second `resume` on the same thread must fail (no pending approval). **Never reuse an approved thread ID after reset.**

## Architecture, limitations and next phase

`validate -> read (MCP) -> approve/interruption [cancel only] -> write (MCP)` with a bounded per-node step counter. `SqliteSaver` stores the LangGraph thread state durably, independent of the observatory database. The `resume` command requires a matching checkpoint waiting on node `approve`.

The capability is issued in `write`, not before `interrupt`, and is sent directly to the MCP server without being put into the LangGraph state, output or trace. The MCP server validates and consumes the capability atomically before executing the domain write. An invalid or missing capability returns `APPROVAL_REQUIRED`. A backend reset revokes all outstanding capabilities.

This phase deliberately does not claim to finish: natural-language intent-to-tool planning via the existing agent, production authentication, multi-step dependency plans, durable long-term user memory, 30 state-graded evaluation scenarios, or full model token/cost telemetry. The LangGraph graph is isolated to prevent regressions in the original astronomy assistant, and can be integrated as a subgraph/router in the next stage.

For the final grade, independently read `ObservatoryStore(db_path).get_state()` after each run, compare to `expect_state`, and also check `forbid_calls`, step limits, and stop reasons. The `--scenarios` evaluator is **not** implemented by this ZIP.

## Commit (only when tests pass)

```bash
git status --short
git diff --check
git add pyproject.toml uv.lock \
  src/astropilote/backend/store.py src/astropilote/backend/approvals.py \
  src/astropilote/mcp/observatory_server.py src/astropilote/agentic/ \
  tests/test_agentic_approval.py docs/AGENTIC_PHASE3_HITL.md
git diff --cached --stat
git commit -m "feat(agentic): add checkpointed MCP workflows and human approval gate"
git remote -v  # origin must be PRIVATE Astropilote_review
git push origin v1-agentic
```
