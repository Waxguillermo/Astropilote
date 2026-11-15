# AstroPilote — Phase 4A: conservative conversational MCP router

## Scope (what this phase adds)

This **local CLI** is a conversational front controller. It dispatches a bounded set
of French/English requests into the **existing Phase-3 LangGraph/MCP booking
workflow**, or to the **existing Generative AI astronomy agent** for normal
astronomy questions. It is not yet a general LLM-based intent planner, a
persistent conversational memory, an authenticated web API, or the final
30-scenario evaluator. In particular, **`/api/chat` is not modified**.

The router is deliberately deterministic and **fail-closed** for actions:

- Read telescope slots from the Observatory MCP server, optionally filtering
  by one explicit ISO date, an upcoming weekday, and morning/afternoon/evening.
  Local slot times are converted from UTC to **Europe/Paris**.
- Create a booking **only when exactly one `S-xx` slot is specified**. The actor
  is taken from the **trusted local `--user` CLI argument**, not the sentence.
- Ask for clarification when a booking/cancellation target is missing,
  multiple targets are mentioned, a request negates an action, or the user only
  asks *whether* it is possible. No backend write is initiated in these cases.
- Begin a cancellation **only for exactly one `BK-xxx` booking**. The existing
  LangGraph `interrupt()` halts before the write. A natural-language
  confirmation (e.g. `oui`, `approve`, `confirme`) **never resumes HITL**.
- Read the details of a booking only if the booking's `user_id` matches the
  trusted CLI actor.
- Route other astronomy questions to `astropilote.agent.graph.run_agent`,
  keeping the previous astronomy code unchanged. This fallback requires the
  previously configured LLM / RAG environment.

The backend and payments are **simulated**. The client must not expose its
`--user` flag or the CLI's HITL `resume --approve` via an unauthenticated HTTP
route. The CLI is designed only for a trusted local operator.

**Important limitation:** requests such as “the best telescope next Tuesday”
are *not* turned into a model-created multi-step plan yet; the router only
executes explicitly supported patterns and asks for clarification otherwise.
The original astronomy graph remains separate from the stateful action graph.

## Installation

Start from a clean `v1-agentic` checkout containing Phase 3, after commit
`3d9416e` or later:

```bash
cd ~/Developer/Astropilote
git branch --show-current
git status --short
unzip -n ~/Desktop/Astropilote_Agentic_Phase4_Conversation.zip -d .
```

No new dependency is needed (Phase 3 already installed MCP and LangGraph).

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_agentic_conversation.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check src/astropilote/agentic/conversation.py src/astropilote/agentic/chat.py tests/test_agentic_conversation.py
```

There are **44 new test cases**: 42 pure/isolated tests and 2 actual
LangGraph+MCP tests. The isolated tests were executed by the author; the two
real MCP integration tests require your Mac environment and must be validated
before committing. If all new tests and all previous tests succeed, expect
**149 passed, 1 skipped, 7 subtests passed**. This is a *target*, not a claim
about your computer's results.

## Reproducible local demonstrations

```bash
cd ~/Developer/Astropilote
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json

# Read-only availability: 2026-11-17 Tuesday evening, Europe/Paris
uv run python -m astropilote.agentic.chat \
  --user alice --date 2026-11-14 \
  --text 'Quels télescopes sont disponibles mardi soir ?'

# Ambiguous request: NO write
uv run python -m astropilote.agentic.chat \
  --user alice --text 'Je veux réserver un télescope'

# Explicit safe booking (creates BK-002 in simulation)
uv run python -m astropilote.agentic.chat \
  --user alice --thread phase4-book-demo \
  --text 'Réserve le créneau S-01 pour moi'

# Reset so BK-001 is again confirmed; start dangerous cancellation
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json
uv run python -m astropilote.agentic.chat \
  --user astro-club --thread phase4-cancel-demo \
  --text 'Annule ma réservation BK-001'

# TRUSTED LOCAL OPERATOR: explicitly refuse and verify unchanged state
uv run python -m astropilote.agentic.demo resume \
  --thread phase4-cancel-demo --reject
```

Expected: the availability query lists slots `S-01`, `S-03`; the ambiguous
request returns `needs_clarification` and performs no write; the booking
completes; cancellation initially yields `awaiting_human_approval`, then
`human_rejected` with `BK-001` unchanged. Use a **new thread ID** if rerunning
an action: checkpoints persist across CLI processes. To demonstrate an approved
cancellation, use a fresh thread and the trusted operator CLI only, as in
`docs/AGENTIC_PHASE3_HITL.md`.

To try an astronomy question, use your configured Ollama/API environment:

```bash
uv run python -m astropilote.agentic.chat \
  --user alice --text 'Dans quelle constellation se trouve M31 ?'
```

## Git (after the tests succeed)

```bash
git diff --check
git add src/astropilote/agentic/conversation.py \
        src/astropilote/agentic/chat.py \
        tests/test_agentic_conversation.py docs/AGENTIC_PHASE4_CONVERSATION.md
git commit -m 'feat(agentic): add conservative natural-language MCP router'
git remote -v  # origin must be the PRIVATE Astropilote_review repo
git push origin v1-agentic
```

## Next steps

Add true conversational context / memory, robust intent extraction with a
schema-validated planner if necessary, and the 30-case state-based evaluator.
Document what the agent does and does not do, especially for indirect prompt
injections in tool results and any future approval authentication boundary.
