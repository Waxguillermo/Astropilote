# AstroPilote — Step-by-Step Engineering Log

This document records the architectural decisions, hardware benchmarks, and empirical ablation studies conducted while building **AstroPilote** (ENPC Generative AI I course project).

---

## Step 1 — Environment, Hardware Detection & Stack Selection (`2026-10-13`)

### Objective
Replace the initial prototype scaffold with a 100% open-source, production-grade agentic RAG and spherical astronomy stack—with **zero Streamlit** dependencies—using **Google Agent Development Kit (`google-adk`)** and local **Gemma 4 (`gemma4`)** inference.

### Decisions & Rationale
1. **Agent Orchestration (`google-adk==2.11.0` + `litellm==1.105.0`)**:
   - Constructed `LlmAgent` (`src/astropilote/agent/adk_agent.py`) wired with `LiteLlm(model="ollama_chat/gemma4")` as the default local model, plus plug-and-play support for OpenAI-compatible local servers (`LM Studio`, `vLLM`, `llama.cpp`), native Gemini (`gemini-2.5-flash`), OpenAI, and Anthropic.
   - Added ADK lifecycle callbacks (`before_model_callback`, `before_tool_callback`, `after_tool_callback`) to enforce a hard upper bound of `max_agent_steps = 6` and record sub-millisecond per-tool telemetry (`ToolCallRecord`).
2. **Hardware Acceleration (`Apple M4` · `MPS`)**:
   - Detected Apple Silicon Metal Performance Shaders (`torch.backends.mps.is_available() == True`) in `get_torch_device()`, accelerating both `SentenceTransformer` dense encoding and `CrossEncoder` reranking by **1.75×** over CPU (`23.27s` vs `40.32s` for indexing 2,242 chunks).
3. **Frontend & Backend (`FastAPI` + Custom Modern Web UI + `Chainlit`)**:
   - Eliminated Streamlit completely. Built a FastAPI backend (`src/astropilote/app/main.py`) serving a bespoke **Observatory Web UI** (`src/astropilote/app/static/`) with a 3-mode theme switcher (**Deep Space Dark**, **Scotopic Night-Vision Red** for telescope field use, and **Daylight Lab Light**), interactive SVG altitude/airmass curves, an APS-C camera FOV framing visualizer, and native HTML `<dialog closedby="any">` source inspection modals.
   - Provided a companion **Chainlit** app (`src/astropilote/app/chainlit_app.py`) with expandable `cl.Step` tool traces and side-drawer chunk inspection.

---

## Step 2 — Corpus Normalization & 3-Way Chunking Ablation (`2026-10-13`)

### Objective
Ingest 506 real astronomical documents (`110 Messier`, `109 Caldwell`, `199 brightest Herschel 400`, `88 IAU Constellations`) from OpenNGC (`CC-BY-SA-4.0`) and Wikipedia (`CC-BY-SA-3.0`), fix split-constellation designations, and experimentally evaluate chunking strategies.

### Bug Discovered & Fixed
- OpenNGC splits *Serpens* into *Serpens Caput* (`SE1`) and *Serpens Cauda* (`SE2`). Normalizing `SE1`/`SE2` $\rightarrow$ `SER` in `src/astropilote/ingest/catalog.py` restored proper constellation metadata for Messier 5 (`M005`), Messier 16 (`M016`, Eagle Nebula), and Caldwell 57 (`C057`).

### 3-Way Chunking Experiment (`eval/chunking_experiment.json`)
We ran `python -m astropilote.ingest.compare_chunking` across all answerable queries in `eval/golden.jsonl`:

| Strategy | Total Chunks | Avg Size | `Recall@5` | `HitRate@5` | `MRR@5` |
|---|---:|---:|---:|---:|---:|
| `1_fixed_no_header` (512 chars, 64 overlap, raw text) | 2,511 | 510 chars | `74.74%` | `88.46%` | `72.56%` |
| `2_fixed_with_header` (512 chars + `[ID \| Name \| Type \| Const \| Mag]` header) | 3,359 | 495 chars | `75.26%` | `92.31%` | `82.18%` |
| **`3_section_with_header` (Production: Section-aware + Contextual Header)** | **2,242** | **552 chars** | **`79.87%`** | **`92.31%`** | **`84.62%`** |

**Conclusion:** Section-aware splitting with prepended catalog headers reduces index footprint by **33%** vs fixed-with-header while improving `Recall@5` by **+5.13 pts** and `MRR@5` by **+12.06 pts** over naive fixed-size chunking.

---

## Step 3 — Hybrid RAG (`BM25` + `Qdrant` Dense) & Cross-Encoder Reranking Ablation (`2026-10-13`)

### Objective
Compare single-modality lexical (`bm25s`) and dense (`BAAI/bge-small-en-v1.5` in local Qdrant) retrieval against Reciprocal Rank Fusion ($k=60$) and Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with document-level diversification (`_diversify_by_doc`).

### 4-Way Retrieval Ablation (`eval/retrieval_ablation.json`)
We ran `python -m astropilote.retrieval.ablation` on `eval/golden.jsonl` using Apple M4 `mps`:

| Pipeline Configuration | `Recall@5` | `HitRate@5` | `MRR@5` | Avg Latency (`MPS`) |
|---|---:|---:|---:|---:|
| `1_bm25_only` (`bm25s` lexical) | `79.87%` | `92.31%` | `84.62%` | `0.4 ms` |
| `2_dense_only` (`BAAI/bge-small-en-v1.5` + Qdrant cosine) | `72.82%` | `88.46%` | `83.65%` | `9.2 ms` |
| `3_hybrid_rrf` (`BM25` + `Dense` fused via RRF $k=60$) | `81.15%` | `92.31%` | `82.82%` | `9.9 ms` |
| **`4_hybrid_rrf_rerank` (Hybrid RRF + Cross-Encoder Reranking)** | **`81.92%`** | **`92.31%`** | **`81.09%`** | **`129.0 ms`** |

**Key Engineering Insight:**
1. Without document diversification, multiple chunks from a single verbose target (e.g., 4 sections of `M042`) crowd out companion targets (`M043`, `CON_ORI`), capping `Recall@5` at `63.72%`. Enforcing `max_per_doc=1` in top-$k$ selection lifts `Recall@5` by **+18.20 percentage points** (`63.72%` $\rightarrow$ `81.92%`).
2. Fusing Stage-1 RRF ranks with Stage-2 Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) ranks via RRF prevents generic web-trained MS-MARCO logits from overriding exact catalog designations (`C014`, `M031`) while promoting semantically matched paragraphs (`Recall@5 = 81.92%`).

---

## Step 4 — Spherical Ephemerides (`Skyfield` IAU GMST) & Sensor FOV Framing (`2026-10-13`)

### Objective
Equip the agent with deterministic physical tools (`search_catalog`, `get_object`, `night`, `visibility`, `geocode`, `weather`) decorated with `@adk_tool` (`src/astropilote/tools/base.py`) so they work natively inside Google ADK's `LlmAgent(tools=[...])`.

### Features Implemented
- **IAU Sidereal & Twilight Solver (`src/astropilote/tools/sky.py`)**: Integrates `skyfield.api.load.timescale(builtin=True)` for high-precision Greenwich Mean Sidereal Time (GMST) alongside analytical solar/lunar orbital elements. Computes true astronomical darkness (`Sun altitude <= -18.0°`), Moon illumination percentage, Moonrise/Moonset, 30-minute topocentric `(altitude_deg, azimuth_deg, compass, airmass)` curves, and summer solstice twilight persistence (e.g., Paris on June 21 where the Sun only reaches `-17.8°`).
- **Optical Rig & APS-C Sensor Framing (`get_object`)**: Computes exact field of view $\text{FOV} = \frac{57.2958 \times 60 \times \text{sensor\_mm}}{f_{\text{mm}}}$ for a standard APS-C sensor ($23.5 \times 15.6\text{ mm}$) and advises whether a target requires a mosaic, fits ideally, or needs cropping.

---

## Step 5 — Pure-Python Verifier & End-to-End Evaluation (`2026-10-13`)

### Objective
Guarantee zero hallucinated citations and zero physically invalid observation schedules via `verify_astro_answer` (`src/astropilote/agent/validator.py`), and benchmark the full system on the 40-question `eval/golden.jsonl` suite.

### Verifier Invariants Checked Before Returning Any Answer
1. **Citation Grounding**: Every cited `[DOC_ID]` in `answer.sources` and `plan[i].sources` must exist in `state["retrieved"]`.
2. **Ephemeris Altitude Fidelity**: Every `PlanItem.altitude_deg` must match the calculated topocentric ephemeris curve within $\pm 2.0^\circ$.
3. **Astronomical Darkness Enforcement**: Every `PlanItem.start` must fall inside the local astronomical darkness window (`Sun <= -18°`) with altitude $\ge 20^\circ$.
4. **Abstention Consistency**: Whenever `answer.abstained == True`, `sources` and `plan` must be strictly empty (`[]`).

### Final Evaluation Metrics (`eval/summary_metrics.json`)
Running `uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/` produced:

| Metric | Score |
|---|---:|
| Total Questions (`eval/golden.jsonl`) | `40` (`26` answerable · `14` abstention/forecast/injection traps) |
| **End-to-End Retrieval Recall** (`search_catalog` + `get_object`) | **`97.95%`** |
| **End-to-End Retrieval MRR** | **`82.37%`** |
| **Prudence (Abstention / Refusal / Injection Accuracy)** | **`100.00%`** (`14 / 14`) |
| **Judged Correctness** | **`92.08%`** |
| **Median Latency (`p50`)** | **`0.132 s`** |
| **95th Percentile Latency (`p95`)** | **`0.244 s`** |
