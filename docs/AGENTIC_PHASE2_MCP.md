# Agentic Workflows — Phase 2: two MCP servers

**Install after Phase 1**, while on private branch `v1-agentic`.

SDK: MCP Python **2.x** (`from mcp.server import MCPServer`). This API differs from MCP 1.x (`FastMCP`). The project requires Python 3.11+.

## Install and verify

```bash
cd ~/Developer/Astropilote
git branch --show-current
uv add 'mcp>=2,<3'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_mcp.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check src/astropilote/mcp tests/test_mcp.py
```

## Two separate stdio servers

Sky (six read-only astronomy tools):

```bash
uv run python -m astropilote.mcp.sky_server
```

Observatory (read state, create booking, **blocked** cancel):

```bash
uv run python -m astropilote.mcp.observatory_server
```

Both commands appear to hang by design: MCP uses stdin/stdout as protocol transport. Do not print debug output to stdout. Use Ctrl+C to end the server if running manually. Normally a client launches the processes for you.

## End-to-end demo (starts both processes automatically)

```bash
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json
uv run python -m astropilote.mcp.smoke_test
```

Expected: six tool names from each independent server; results from real MCP `geocode` and `list_slots` calls, with `is_error: false` and JSON data. No live networking needed for this demo.

## JSON result and error contract

Successful result (MCP `is_error=false`):

```json
{"ok":true,"data":{"id":"BK-002","slot_id":"S-01","user_id":"observer","status":"confirmed","paid":false,"refunded":false}}
```

Business failure (MCP `is_error=true`):

```json
{"ok":false,"error":{"code":"SLOT_NOT_FOUND","message":"Slot does not exist","details":{"slot_id":"S-NOT-THERE"}}}
```

All business results appear in both MCP `content` (JSON text) and `structured_content`. Schema/type failures also set MCP `is_error=true`, but SDK-generated error text can have a different shape.

**Dangerous cancellation is intentionally blocked in Phase 2.** `cancel_booking` returns `APPROVAL_REQUIRED` without writing to SQLite, regardless of LLM-controlled arguments. In Phase 3, build a LangGraph checkpointed `interrupt()` and a trusted authorization gate. A tool argument such as `approved=true` is *not* human approval. `reset_backend` is deliberately not exposed as a model-callable MCP tool: the evaluator/operator resets the database directly.

**The servers are not yet connected to the LangGraph agent.** This phase demonstrates MCP exposure and backend functionality; Phase 3 integrates the client and HITL approval.

## Git

```bash
git add pyproject.toml uv.lock src/astropilote/mcp tests/test_mcp.py docs/AGENTIC_PHASE2_MCP.md
git commit -m 'feat(agentic): add two typed MCP servers and protocol tests'
git remote -v
git push origin v1-agentic  # only if origin points to the PRIVATE Astropilote_review repository
```
