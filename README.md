# Astropilote — Agentic Astronomy RAG & Night Sky Planner

An open-source astronomy assistant that answers deep-sky catalog questions and builds tonight's observation schedule using **hybrid retrieval** (Dense `BGE` + `BM25` + Cross-Encoder Reranking) and **deterministic sky ephemeris** (`Astropy` + `Open-Meteo` + Light Pollution).

Supports both **Anthropic API (`Claude`)** and **Local (`Ollama / Gemma 4`)** modes — switchable via `.env` or live in the web interface.

---

## How to run

```bash
# 1. Install dependencies (Python 3.11+)
uv sync

# 2. Configure environment (.env is already created; paste your ANTHROPIC_API_KEY)
cp .env.example .env

# 3. Build the 506-document corpus & hybrid indexes (Qdrant + BM25)
uv run python -m astropilote.ingest.run_ingest
uv run python -m astropilote.index.build_index

# 4. Launch the web interface at http://127.0.0.1:8000
uv run uvicorn astropilote.app.main:app --reload --port 8000
```

---

## LLM modes (`API` vs `Local`)

Set in `.env` or toggle directly in the left sidebar of the web app:

- **API Mode (`LLM_MODE=api`)**: Uses the official `anthropic` Python SDK (`ANTHROPIC_API_KEY=...`, default model `claude-sonnet-4-20250514`).
- **Local Mode (`LLM_MODE=local`)**: Uses a local Ollama server (`OLLAMA_BASE_URL=http://localhost:11434`, default model `gemma4`).
- **Automatic Fallback**: If the selected LLM endpoint is offline or no key is set yet, the agent still executes all retrieval and sky tools and returns a structured, citation-grounded synthesis.

---

## Models & stack used

- **Embeddings (Dense)**: `BAAI/bge-small-en-v1.5` (`384`-d normalized vectors stored in local `Qdrant`)
- **Lexical Search**: `BM25Okapi` (`rank-bm25`) with preserved catalog identifiers (`M31`, `NGC 7000`, `C14`)
- **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranks top-16 RRF candidates to top-6)
- **Ephemeris & Environment**: `Astropy` (astronomical dusk/dawn, Moon phase/altitude, real-time `Alt/Az` curves), `Open-Meteo` (cloud cover, humidity, wind), and Bortle sky brightness estimator

---

## Why hybrid retrieval + reranking helps

Evaluated on `40` golden astronomy queries (`eval/golden.jsonl`):

| Pipeline Stage | Recall@5 | MRR@5 | Why it matters |
| :--- | :---: | :---: | :--- |
| **Chunking: Fixed (No Header)** | `0.747` | `0.726` | Splits mid-section and loses object identity context |
| **Chunking: Section + Header** | **`0.799`** | **`0.846`** | Preserves `[Section \| Object=M31]` headers per chunk |
| **Dense Only (`BGE-small`)** | `0.728` | `0.837` | Strong on semantic descriptions (*"face-on spiral galaxy"*) |
| **BM25 Only (Lexical)** | `0.799` | `0.846` | Strong on exact catalog codes (`NGC 869`, `C80`, `M27`) |
| **Hybrid (`RRF`) + Reranker** | **`0.819`** | **`0.811`** | Combines exact catalog codes with semantic phrasing |
| **Full Agent (`Tools + RAG`)** | **`0.980`** | **`0.921`*** | Adds deterministic catalog filtering + ephemeris verification (`100%` refusal on traps, *judged correctness) |

Run the benchmark suite:

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

---

## Data & RAG visualizations notebook

Open [`notebooks/corpus_and_rag_visualizations.ipynb`](notebooks/corpus_and_rag_visualizations.ipynb) to view pre-rendered presentation charts:
1. **All-Sky Mollweide Map (J2000)** of the 418 Messier, Caldwell, and NGC targets by class.
2. **Corpus Breakdown & Visual Magnitude vs. Angular Size** scatter plot (`506` docs / `2,242` chunks).
3. **Real-Time Night Altitude Trajectories** from Fontainebleau (`48.40°N`) with the `20°` observability floor.
4. **Chunking & Hybrid Retrieval Ablation Charts** (`Recall@5` & `MRR@5`).

---

## Example questions (tested)

1. *"What is the best time to observe the Ring Nebula (M57) tonight from Fontainebleau, and what is its visual magnitude?"*
2. *"Plan a 3-object observation session tonight from Fontainebleau using 10x50 binoculars."*
3. *"Which globular clusters in Hercules are visible tonight and how do M13 and M92 compare?"*
4. *"Can I observe Omega Centauri (C80) from Paris tonight? Explain why or why not."*
5. *"Compare the Orion Nebula (M42) and the Lagoon Nebula (M8): which season is each best observed in?"*
6. *"Recommend 3 bright galaxies visible from Lyon on 2026-04-18 for a 200mm telescope."*
