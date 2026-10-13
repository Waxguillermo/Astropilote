# AstroPilote — Visual Astronomy & Astrophotography Session Copilot

> **ENPC — Generative AI I Group Project**  
> **Code Freeze:** Tuesday, October 20, 2026 (`v1.0`) · **Defense:** Wednesday, October 21, 2026

**AstroPilote** is a local-first **Retrieval-Augmented Generation (RAG) and tool-using agentic assistant** designed for visual observers and astrophotographers. Given an observation site, a date/time window, and an optical rig (telescope aperture, focal length, camera sensor, or filters), AstroPilote builds a verified, step-by-step night observation and imaging plan backed by astronomical ephemerides and cited encyclopedia sources.

---

## 1. System Architecture

AstroPilote is structured around three stages:

1. **Offline Data & Indexing (`ingest/`, `index/`)**:
   - **Corpus (~500 documents)**: 110 Messier objects, 109 Caldwell objects, ~200 Herschel 400 deep-sky targets, and 88 IAU constellations from **English Wikipedia** (`CC BY-SA 4.0`), enriched with **OpenNGC** (`CC-BY-SA-4.0`) astrometric and photometric metadata (Right Ascension, Declination, visual/photographic magnitude, angular size in arcminutes, surface brightness, object class).
   - **Section-aware Chunking**: Each article is split by semantic section (*Observation*, *Astrophotography / Imaging*, *Characteristics*, *History*) up to ~400 tokens, prefixed with a structured object header (`[M031 | Andromeda Galaxy | Spiral Galaxy | Mag 3.4 | Size 178'x63']`) so every chunk remains self-contained.
   - **Dual Index**: Lexical index via `bm25s` + dense vector index stored locally in **Qdrant** with metadata payload filtering.

2. **Hybrid Retrieval & Reranking (`retrieval/`)**:
   - Parallel retrieval of top-20 lexical candidates (**BM25**) and top-20 semantic candidates (**Dense embeddings**).
   - **Reciprocal Rank Fusion (RRF, $k=60$)** merges both lists into 20 unique candidates.
   - **Cross-Encoder Reranking** scores `(query, chunk)` pairs and keeps the top-5 passages for the agent.

3. **Bounded LangGraph Agent & Deterministic Verification (`agent/`, `tools/`, `tracking/`)**:
   - **`guard`**: Extracts user intent (location, date, rig/aperture, focal length, imaging vs. visual mode) and blocks prompt injections, off-topic queries, or impossible long-range weather requests.
   - **`agent` ⇄ `tools` (max 6 steps)**: Calls 6 specialized tools across 3 families:
     - *Catalog & RAG*: `search_catalog`, `get_object`
     - *Sky Ephemerides (`skyfield`)*: `night` (astronomical twilight, Moon phase & rise/set), `visibility` (altitude/airmass curve every 30 min, FOV framing check)
     - *Real-World Context*: `geocode` (OpenStreetMap Nominatim), `weather` (Open-Meteo cloud cover & seeing; deterministic fixtures during evaluation)
   - **`compose` & `validate`**: Produces a structured `AstroAnswer` (Pydantic) and runs a **pure-Python verifier** checking that (a) every cited ID was actually retrieved, (b) reported altitudes match Skyfield within $\pm 2^\circ$, (c) observation slots fall within astronomical darkness, and (d) abstained answers cite zero sources. Triggers at most 1 retry if verification fails.

---

## 2. Exact Models & Versions

Per course requirements, all open-weight and optional API models used in this repository are explicitly pinned below:

| Component | Default Local Model (Open-Weight) | Version / Checkpoint | Alternative / Ablation |
| :--- | :--- | :--- | :--- |
| **Agent & Composer LLM** | `qwen3:8b` (via Ollama) | `Qwen/Qwen3-8B` (Q4_K_M) | `llama3.1:8b` / API model (`gpt-4o-mini` / `gemini-2.5-flash`) |
| **Dense Embeddings** | `BAAI/bge-m3` | Hugging Face `bge-m3` (1024-d) | `intfloat/multilingual-e5-base` (768-d) |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | Hugging Face `bge-reranker-v2-m3` | No reranker (RRF top-5 baseline) |
| **Evaluation Judge** | `qwen3:8b` / API Judge | Configured via `.env` | Cross-checked on 10 manual human-graded answers |

---

## 3. Quickstart & Commands

### Prerequisites
- Python **3.11+** and [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally with `ollama pull qwen3:8b`

### Installation
```bash
# 1. Clone and install dependencies in an isolated virtual environment
uv sync --all-extras

# 2. Configure environment variables (never commit .env)
cp .env.example .env
```

### Step-by-Step Pipeline
```bash
# 1. Collect & clean the ~500 Wikipedia + OpenNGC documents and build section chunks
uv run python -m astropilote.ingest.run_ingest

# 2. Build local BM25 and Qdrant vector indexes
uv run python -m astropilote.index.build_index

# 3. Run the mandatory evaluation suite on the 40-question golden set
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/

# 4. Launch the interactive Streamlit UI (shows answer, sky plan, citations & agent trace)
uv run streamlit run src/astropilote/app/main.py
```

---

## 4. Repository Structure

```text
astropilote/
├── data/
│   ├── DATA_CARD.md             # Corpus selection, licensing, and schema documentation
│   ├── raw/                     # Raw OpenNGC CSV + Wikipedia JSON dumps (git-ignored)
│   ├── processed/               # Cleaned documents and section chunks (git-ignored)
│   └── index/                   # Local BM25 + Qdrant storage (git-ignored)
├── docs/
│   └── architecture.html        # Interactive visual architecture blueprint (English)
├── eval/
│   ├── golden.jsonl             # 40-question evaluation benchmark (factual, planning, OOD, injections)
│   ├── run_eval.py              # Mandatory CLI evaluation runner writing eval/results.jsonl
│   └── fixtures/                # Deterministic geocoding & weather fixtures for reproducible evals
├── notebooks/                   # Data inspection (20 random docs & 20 random chunks analysis)
├── src/astropilote/
│   ├── config.py                # Centralized Pydantic Settings (.env loader)
│   ├── ingest/                  # Wikipedia + OpenNGC collection, cleaning, section chunking
│   ├── index/                   # BM25s and Qdrant dense index builders
│   ├── retrieval/               # Hybrid search (BM25 + Dense), RRF fusion, Cross-Encoder reranker
│   ├── tools/                   # 6 agent tools: catalog, skyfield ephemerides, geocode, weather
│   ├── agent/                   # AstroState, Pydantic schemas, LangGraph nodes & workflow
│   ├── tracking/                # Token usage, latency (s), tool call trace, and cost ($) tracker
│   └── app/                     # Streamlit UI displaying observation plans & telemetry
└── tests/                       # Unit tests for schemas, tools, and retrieval
```
