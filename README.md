# AstroPilote — Visual Astronomy & Astrophotography RAG Copilot

An open-source **Google ADK (`google-adk`) & Hybrid RAG copilot** that answers deep-sky astronomy questions and builds verified night-sky observation and astrophotography schedules over a curated corpus of **506 canonical targets** (110 Messier, 109 Caldwell, 199 brightest Herschel 400 deep-sky objects, and all 88 IAU constellations). Each query is routed through a safety & intent **guard**, answered by combining **BM25 (`bm25s`)** and **Qdrant** dense vector search with **Reciprocal Rank Fusion (RRF)**, refined with a **Cross-Encoder reranker**, enriched with deterministic **Skyfield** topocentric ephemerides, synthesized by **Gemma 4** (or Gemini/OpenAI/Anthropic via **LiteLLM**), and audited by a **pure-Python verifier** before reaching the custom **FastAPI Observatory Web UI** (or **Chainlit**).

```
question ──▶ guard ──▶ Google ADK LlmAgent ⇄ 6 tools (BM25 + Qdrant RRF ──▶ cross-encoder rerank + Skyfield ephemeris)
                                                                                       │
         verified answer + SVG sky chart ◀── Python verifier (4 invariants) ◀── Gemma 4 compose
```

## How to run

```bash
# 1. Environment + dependencies  (tested on Python 3.12 with uv)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv sync --extra dev

# 2. Environment variables  (defaults to local Gemma 4 via Ollama; no paid API key required)
cp .env.example .env          # optionally edit .env to set LLM_MODEL=gemma4 or GEMINI_API_KEY

# 3. Ingest the 506-document corpus & build the BM25 + Qdrant index ONCE  (~90 MB embedding model on first run)
uv run python -m astropilote.ingest.run_ingest
uv run python -m astropilote.index.build_index

# 4. Launch the Custom FastAPI Observatory Web UI  (opens http://127.0.0.1:8000)
uv run python -m astropilote.app.main
# (Optional alternative: launch the Chainlit chat interface)
# uv run chainlit run src/astropilote/app/chainlit_app.py -w

# 5. Run the 40-question evaluation benchmark & unit test suite
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
uv run pytest -v
```

> If your system default `python` is 3.13/3.14, some ML wheels may not exist yet.
> Using `uv venv --python 3.12 .venv` (or the `Makefile` shortcuts: `make setup`, `make ingest`, `make index`, `make app`, `make eval`, `make test`) guarantees a reproducible Python 3.12 environment.

## Models & Stack used

| Role | Model / Component |
|---|---|
| Agent Runtime | **Google ADK** (`google-adk 2.11`) `LlmAgent` + `LiteLlm` + pure-Python `verify_astro_answer` |
| Chat LLM | **Gemma 4** (`ollama_chat/gemma4` local default; swappable to `qwen3:8b`, `gemini-2.5-flash`, or `gpt-4o-mini`) |
| Dense Embeddings | `BAAI/bge-small-en-v1.5` (384-d) — local `SentenceTransformer` on Apple Silicon `mps` / CUDA |
| Lexical + Vector Store | **`bm25s`** (`data/index/bm25`) + **Qdrant** local persistence (`data/index/qdrant`), fused via RRF ($k=60$) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` — `sentence-transformers` `CrossEncoder` |
| Ephemeris & Weather | **Skyfield** IAU 2000B GMST + analytical solar/lunar twilight solver (`-18°`), **Open-Meteo**, **Nominatim** |
| Web Frontend & API | **FastAPI** + custom zero-dependency **HTML5/CSS3/SVG Observatory UI** (plus optional **Chainlit** UI) |

## Why section-aware chunking + hybrid RRF + reranking helps

Astronomical queries mix **exact alphanumeric designations** (`M31`, `NGC 7000`, `Caldwell 14`, `RA 00h 42m`) with **semantic observing concepts** (*"bright planetary nebula for a small refractor under moonlight"*).
- **Dense similarity alone** (`72.82% Recall@5`) struggles with exact catalog numbers like `C014` vs `M014`.
- **BM25 lexical search alone** (`79.87% Recall@5`) matches catalog codes well, but misses semantic synonyms when a user describes a target without naming its NGC/Messier number.
- **Section-aware chunking with contextual headers** (`[M031 | Andromeda Galaxy | Type: galaxy | Constellation: AND | Mag: 3.4]`) keeps each chunk self-contained so downstream chunks (*Observing & Astrophotography Notes*) never lose their parent object's identity.
- **Reciprocal Rank Fusion (RRF, $k=60$) + Cross-Encoder Reranking** combines both signals and jointly re-scores `(question, chunk)` pairs while diversifying results to at most 1 chunk per document (`_diversify_by_doc`), lifting retrieval `Recall@5` to **`81.92%`** (and **`97.95%`** end-to-end when combined with the agent's `get_object` tool).

### Experimental Ablations (`make ablation`)

**1. Chunking Strategy Comparison (`eval/chunking_experiment.json`)**

| Strategy | Chunks | Avg Size | `Recall@5` | `HitRate@5` | `MRR@5` |
|---|---:|---:|---:|---:|---:|
| `1_fixed_no_header` (512 chars, 64 overlap, raw text) | 2,511 | 510 chars | `74.74%` | `88.46%` | `72.56%` |
| `2_fixed_with_header` (512 chars + contextual metadata header) | 3,359 | 495 chars | `75.26%` | `92.31%` | `82.18%` |
| **`3_section_with_header` (Production: Section-aware + Header)** | **2,242** | **552 chars** | **`79.87%`** | **`92.31%`** | **`84.62%`** |

**2. Retrieval Pipeline Ablation (`eval/retrieval_ablation.json`)**

| Retrieval Mode | `Recall@5` | `HitRate@5` | `MRR@5` | Avg Latency (`MPS`) |
|---|---:|---:|---:|---:|
| `1_bm25_only` (`bm25s` lexical) | `79.87%` | `92.31%` | `84.62%` | `0.4 ms` |
| `2_dense_only` (`BAAI/bge-small-en-v1.5` + Qdrant) | `72.82%` | `88.46%` | `83.65%` | `9.2 ms` |
| `3_hybrid_rrf` (`BM25` + `Dense` RRF $k=60$) | `81.15%` | `92.31%` | `82.82%` | `9.9 ms` |
| **`4_hybrid_rrf_rerank` (Hybrid RRF + Cross-Encoder)** | **`81.92%`** | **`92.31%`** | **`81.09%`** | **`129.0 ms`** |

## Agent tools & deterministic verifier

The Google ADK agent is equipped with **6 typed `@adk_tool` functions** (`src/astropilote/tools/`):
1. `search_catalog(query, catalog, object_type, constellation, max_magnitude, narrowband_only, top_k)` — Hybrid RAG search with structured Qdrant/BM25 payload filters.
2. `get_object(object_id, focal_length_mm)` — Direct lookup of coordinates, magnitude, angular size, and APS-C camera sensor ($23.5 \times 15.6\text{ mm}$) field-of-view framing advice.
3. `geocode(place_name)` — Resolves observation sites (`Fontainebleau`, `Chamonix`, `Paris`, etc.) via deterministic fixtures or OpenStreetMap Nominatim.
4. `night(lat, lon, obs_date)` — Computes astronomical darkness (`Sun <= -18°`), Moon illumination percentage, and Moonrise/Moonset times.
5. `visibility(object_ids, lat, lon, obs_date, focal_length_mm)` — Computes topocentric altitude, azimuth, compass direction, and airmass every 30 minutes from `20:30` to `04:00` local time.
6. `weather(lat, lon, obs_date)` — Fetches nighttime cloud cover and seeing (with automatic refusal if `obs_date` is $>14$ days in the future).

Before any `AstroAnswer` is returned, `verify_astro_answer` (`src/astropilote/agent/validator.py`) checks **4 hard invariants** (triggering up to 1 self-correction retry if violated):
- **Citation Grounding:** $\forall s \in \text{cited\_sources},\; s \in \text{retrieved\_doc\_ids}$
- **Altitude Fidelity:** $\forall p \in \text{plan},\; |\text{alt}_{\text{plan}}(p) - \text{alt}_{\text{ephemeris}}(p)| \le 2.0^\circ$
- **Darkness Window:** $\forall p \in \text{plan},\; p.\text{start} \in [\text{dark\_start}, \text{dark\_end}] \land \text{alt}(p) \ge 20^\circ$
- **Abstention Consistency:** $\text{abstained} = \text{True} \implies \text{sources} = \emptyset \land \text{plan} = \emptyset$

### End-to-End Evaluation Benchmark (`eval/summary_metrics.json`)

| Metric (`40` questions: `26` answerable, `14` traps) | Score |
|---|---:|
| **End-to-End Retrieval Recall** | **`97.95%`** |
| **End-to-End Retrieval MRR** | **`82.37%`** |
| **Prudence (Abstention / Forecast Refusal / Injection Defense)** | **`100.00%`** (`14/14`) |
| **Judged Correctness** | **`92.08%`** |
| **Median Latency (`p50`) / `p95` Latency** | **`0.13s` / `0.24s`** |

## Example questions (tested)

*Answers below are produced by the AstroPilote pipeline; bracketed IDs are verified corpus citations.*

**Q1 — Saturday night, October 17, 2026, I will be in Fontainebleau with an 80 mm refractor (480 mm focal length). What targets should I observe or image?**
- **Night Conditions:** Astronomical darkness `20:40 -> 06:20` | Moon `36% (sets 22:30)` | Clouds `10%`.
- **Recommended Session Plan:**
  - `20:40–21:10` **Hercules Globular Cluster** `[M013]` (`44.6° W`, easy — Broadband continuum target)
  - `21:00–21:30` **Dumbbell Nebula** `[M027]` (`65.2° SW`, medium — Narrowband Ha/OIII friendly, cuts through early moonlight)
  - `21:30–22:00` **Double Cluster in Perseus** `[C014]` (`55.6° NE`, easy — Broadband open cluster pair NGC 869 & NGC 884)
  - `22:00–22:30` **Andromeda Galaxy** `[M031]` (`69.4° E`, easy — Ideal framing: target `178'` fills `105.8%` of APS-C width `168.3'×111.7'`)
  - `22:30–23:00` **Pleiades** `[M045]` (`42.3° E`, easy — Best after Moonset at 22:30)
*Sources: `M013`, `M027`, `C014`, `M031`, `M045`.*

**Q2 — What are the two brightest satellite galaxies of the Andromeda Galaxy (M31) that can be seen in the same wide field of view?**
The Andromeda Galaxy `[M031]` is accompanied by two bright dwarf elliptical satellite galaxies visible in the same wide-field telescope or telephoto frame: **Messier 32** `[M032]` (compact elliptical galaxy, visual magnitude `8.1`, angular size `8.5'`) and **Messier 110** `[M110]` (NGC 205, visual magnitude `8.1`, angular size `19.5'`).
*Sources: `M031`, `M032`, `M110`.*

**Q3 — Can I observe Omega Centauri (Caldwell 80) from Paris on May 15, 2026 at midnight?**
**Omega Centauri `[C080]`** in Centaurus (`Dec -47.48°`) reaches a maximum altitude of **`-6.3°`** from Paris (`48.86°N`) and never rises above the horizon. It is not observable from this latitude.
*Sources: `C080`, `CON_CEN`.*

**Q4 — What can you observe of the supergiant elliptical galaxy IC 1101 in Abell 2029 with an amateur telescope?**
*Abstained:* `IC 1101 (supergiant elliptical galaxy in Abell 2029)` is outside the AstroPilote 506-document corpus (which covers Messier, Caldwell, bright Herschel 400 deep-sky objects, and the 88 IAU constellations).
*Sources: `[]` (verified abstention).*

## Notes

- **Custom Observatory UI extras:** Interactive **SVG Night Sky Altitude & Airmass Chart** (`20:30` to `04:00` local time with `-18°` darkness band and `20°` horizon line), real-time **APS-C Sensor FOV Framing Visualizer**, **3-mode Theme Switcher** including **Scotopic Red Night-Vision Mode** for telescope field use, and native HTML `<dialog closedby="any">` modals for inspecting any cited `[DOC_ID]` or viewing the live RAG ablation tables.
- **Submission hygiene:** `.env`, `.venv/`, `data/raw/`, `data/processed/`, and `data/index/` are git-ignored. All corpus data and indexes are deterministically rebuilt from open-licensed catalogs (`CC-BY-SA-4.0` / `CC-BY-SA-3.0`, documented in [`data/DATA_CARD.md`](data/DATA_CARD.md) and [`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md)).
- **Key files:**
  - `src/astropilote/ingest/` — Catalog fetcher (`OpenNGC` + `Wikipedia`), cleaner, and 3-way chunking experiment (`compare_chunking.py`)
  - `src/astropilote/index/build_index.py` — One-command `bm25s` and `Qdrant` dense vector indexer (`MPS` / `CUDA` / `CPU`)
  - `src/astropilote/retrieval/` — Hybrid retriever (`hybrid.py`) and 4-way retrieval ablation (`ablation.py`)
  - `src/astropilote/tools/` — Six `@adk_tool` functions (`catalog.py`, `sky.py`, `world.py`)
  - `src/astropilote/agent/` — Google ADK `LlmAgent` (`adk_agent.py`), workflow orchestrator (`graph.py`, `nodes.py`), and verifier (`validator.py`)
  - `src/astropilote/app/` — FastAPI server (`main.py`), custom Observatory frontend (`static/`), and Chainlit app (`chainlit_app.py`)
  - `eval/` — Golden benchmark (`golden.jsonl`), evaluation runner (`run_eval.py`), and JSON metrics (`summary_metrics.json`, `retrieval_ablation.json`, `chunking_experiment.json`)
