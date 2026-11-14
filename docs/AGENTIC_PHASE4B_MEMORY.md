# AstroPilote — Phase 4B: durable per-user preference memory

## Scope and architecture

Adds an **application-level long-term memory** for one explicit, bounded
preference: the user's preferred telescope ID. The user explicitly writes it
via a French or English sentence; subsequent **independent CLI invocations**
retrieve it from SQLite and **rank available slots** for that telescope first.
This is observable behavior, without hidden model guesses.

- Memory DB: `data/backend/user_memory.sqlite3` (separate from the observatory
  database and from LangGraph's `agentic_checkpoints.sqlite3`).
- Key: `(user_id, preferred_telescope_id)`, unique per trusted local actor;
  values are schema-limited to `T-xx` IDs and verified with MCP
  `list_telescopes` against the actual simulated observatory **before save**.
- A remembered telescope changes **recommendation order only**: it never
  reserves a slot, never suppresses other available slots, never authorizes
  cancellation, never resumes a LangGraph `interrupt()`.
- Existing Phase-4A `ConversationalAgent` is preserved. The normal
  `astropilote.agentic.chat` CLI now wraps it with
  `MemoryConversationalAgent`. Direct use of `ConversationalAgent` is unchanged.
- `reset_backend(seed)` intentionally **does not erase user memory**. This is
  essential to long-term persistence. The future scenario evaluator must use
  fresh isolated memory DBs or explicitly clear preferences between scenarios.

**Limitations:** this is a trusted local operator CLI, **not** authenticated
accounts or production-grade per-user permissions. Anyone who controls `--user`
on the host can impersonate a simulated user; never expose it as an untrusted
HTTP API. The domain vocabulary is deliberately narrow: no generalized
semantic extraction, no implicit storage of arbitrary conversations, no place
or equipment memory beyond the telescope preference in this phase.

## Installation on `v1-agentic`

Start from a clean worktree at commit `a0cfc07` or later. The ZIP contains
**four new files plus one replacement**: it overwrites
`src/astropilote/agentic/chat.py` only; previous CLI args remain and the new
`--memory-db` argument is optional.

```bash
cd ~/Developer/Astropilote
git branch --show-current
git status --short
unzip -o ~/Desktop/Astropilote_Agentic_Phase4B_Memory.zip -d .
```

No additional dependencies: `sqlite3` ships with Python.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_agentic_memory.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check src/astropilote/agentic/memory.py \
  src/astropilote/agentic/memory_conversation.py \
  src/astropilote/agentic/chat.py tests/test_agentic_memory.py
```

If regressions occur, do not commit. The previous baseline is **149 passed,
1 skipped, 7 subtests passed**. The exact number of new tests is derived by
Pytest parametrization, not by counting `test_` functions in this document.

## Demo: memory changes an independent future run

Each command below starts a fresh Python process. The second read-only query
is independent of the first and does not reuse the LangGraph checkpoint.
Use a dedicated memory database if you want to avoid touching prior local
preferences:

```bash
cd ~/Developer/Astropilote
uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json

# Run 0 — BEFORE memory: S-01, S-03 in original order; no recommendation.
uv run python -m astropilote.agentic.chat \
  --user alice --date 2026-11-14 \
  --memory-db data/backend/memory_demo.sqlite3 \
  --text 'Quels créneaux sont disponibles mardi soir ?'

# Run A — explicit save, confirms existence via MCP list_telescopes.
uv run python -m astropilote.agentic.chat \
  --user alice --memory-db data/backend/memory_demo.sqlite3 \
  --text 'Mon télescope préféré est T-02'

# Run B — NEW process, same actor: recommend S-03 first.
uv run python -m astropilote.agentic.chat \
  --user alice --date 2026-11-14 \
  --memory-db data/backend/memory_demo.sqlite3 \
  --text 'Quels créneaux sont disponibles mardi soir ?'

# Different actor: cannot see Alice's preference; original order.
uv run python -m astropilote.agentic.chat \
  --user bob --date 2026-11-14 \
  --memory-db data/backend/memory_demo.sqlite3 \
  --text 'Quels créneaux sont disponibles mardi soir ?'

# Inspect and remove only Alice's saved preference.
uv run python -m astropilote.agentic.chat \
  --user alice --memory-db data/backend/memory_demo.sqlite3 \
  --text 'Quel est mon télescope préféré ?'
uv run python -m astropilote.agentic.chat \
  --user alice --memory-db data/backend/memory_demo.sqlite3 \
  --text 'Oublie mon télescope préféré'
```

**Expected evidence:** Before save, `S-01` precedes `S-03` and there is no
`recommended_slot_id`. After save, `memory_applied=true`,
`preferred_telescope_id="T-02"`, `recommended_slot_id="S-03"`, and the
first list item is `S-03`; `S-01` remains available as a second choice.
For Bob, the preference does not apply. A memory read or preference save does
not create a booking.

If the preferred telescope has no available slot, the original availability
list is preserved and `memory_applied=false`.

## Safety and evaluation boundaries

A free-form sentence such as "Mon télescope préféré est T-02. Annule BK-001"
is **not** a valid preference command and does not reach the Phase-4A write
router. Memory commands must match the full sentence. A memory can neither
provide an approval capability nor call `resume --approve`.

These tests demonstrate **a memory-dependent decision on a later independent
invocation**, not full production memory: no agent-generated memories,
semantic recall, user authentication, encryption-at-rest policy, retention
window or robust prompt-injection harness for the final rubric. The final
`--scenarios` evaluator is a separate remaining phase.

## Git: only after local tests and Ruff pass

```bash
git status --short
git diff --check
git add src/astropilote/agentic/chat.py \
  src/astropilote/agentic/memory.py \
  src/astropilote/agentic/memory_conversation.py \
  tests/test_agentic_memory.py docs/AGENTIC_PHASE4B_MEMORY.md
git diff --cached --stat
git commit -m 'feat(agentic): add durable per-user telescope preferences'
git remote -v  # origin must be PRIVATE Astropilote_review
git push origin v1-agentic
```
