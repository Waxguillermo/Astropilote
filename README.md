# Astropilote — Agentic Astronomy RAG & Night Sky Planner

An open-source astronomy assistant that answers deep-sky questions in plain, beginner-friendly language and builds tonight's observation schedule using **hybrid retrieval** (Dense `BGE` + `BM25` + Cross-Encoder Reranking) and **deterministic sky ephemeris** (`Skyfield/IAU` + `Open-Meteo` + Light Pollution).

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

## Visual architecture

Static encyclopedias know **what** an object is (`M31` is a spiral galaxy at `mag 3.4`), but they cannot know **whether** it is above the horizon from Fontainebleau tonight at `22:00`, whether the Moon is full, or whether the sky is cloudy. **Astropilote** separates static encyclopedic retrieval (Hybrid RAG) from real-time physical calculations (Agent Tools):

```mermaid
flowchart LR
    U["User Question"] --> G["1. Guardrail Node<br/>(Prompt Injection &<br/>>14d Weather Check)"]
    G -->| Blocked | R["Refuse / Abstain<br/>(0 Hallucinations)"]
    G -->| Safe | A["2. Agent Planner<br/>(Intent, Location & Date)"]

    A --> T1["Hybrid RAG Tools<br/>• search_catalog (BM25 + Qdrant + Rerank)<br/>• get_object (Metadata & FOV)"]
    A --> T2["Sky & World Tools<br/>• geocode (Nominatim)<br/>• night (Sun -18° & Moon)<br/>• visibility (Alt/Az >= 20°)<br/>• weather (Open-Meteo & Bortle)"]

    T1 --> C["3. Compose Node<br/>☁️ Anthropic Claude API<br/>💻 or Local Ollama (Gemma 4)"]
    T2 --> C

    C --> V["4. Validator Node<br/>(Checks [DOC_ID] citations,<br/>Altitudes >= 20° & Night hours)"]
    V --> O["Verified Beginner Guide<br/>+ Night Schedule & Altitude Curve"]
```

```text
Astropilote/
├── data/
│   ├── DATA_CARD.md               # Detailed corpus documentation & schema
│   ├── raw/                       # OpenNGC CSVs + cached Wikipedia JSON extracts
│   ├── processed/                 # 506 documents.jsonl & 2,659 section-aware chunks.jsonl
│   └── index/                     # Persisted BM25 matrices + local Qdrant vector store
├── src/astropilote/
│   ├── ingest/                    # OpenNGC + Wikipedia cleaner & section-aware chunker
│   ├── index/                     # BM25 (bm25s) & Dense (BAAI/bge-small-en-v1.5) builder
│   ├── retrieval/                 # Hybrid RRF fusion + Cross-Encoder reranking + ablations
│   ├── tools/                     # 6 agent tools: search_catalog, get_object, geocode, night, visibility, weather
│   ├── agent/                     # 5-node state machine (guard -> agent -> tools -> compose -> validate)
│   └── app/                       # FastAPI backend + clean 2-pane Web UI
├── notebooks/
│   └── corpus_and_rag_visualizations.ipynb  # Executed notebook with 4 presentation charts
└── eval/
    ├── golden.jsonl               # 40 benchmark queries (factual, session plans, southern traps, injections)
    └── run_eval.py                # CLI evaluation runner producing eval/results.jsonl
```

---

## Data sources & corpus (`506` documents / `2,659` chunks)

See [`data/DATA_CARD.md`](data/DATA_CARD.md) for full details. The assistant combines **two static open datasets** with **three live calculation/environmental sources**:

| Source | Type | Role in Astropilote | License / Origin |
| :--- | :--- | :--- | :--- |
| **OpenNGC** (`NGC.csv` + `addendum.csv`) | Structured Catalog | Exact J2000 coordinates (`RA`/`Dec`), visual magnitudes (`mag`), angular dimensions (`arcmin`), constellations, and aliases (`M31` $\leftrightarrow$ `NGC 224`) | CC-BY-SA 4.0 (`mattiaverga/OpenNGC`) |
| **English Wikipedia** (`MediaWiki API`) | Encyclopedic Text | Multi-section narrative articles (*Overview*, *Observation*, *Characteristics*) cleaned and split into section-aware chunks | CC-BY-SA 4.0 (`en.wikipedia.org`) |
| **IAU Constellations & Observing Guides** | Reference Guides | All `88` official IAU constellations (`CONST_*`) plus `5` practical guides on magnitude, light pollution (`Bortle`), and filters (`GUIDE_*`) | Public Domain / Curated |
| **Spherical Ephemeris (`sky.py`)** | Deterministic Physics | Computes exact astronomical darkness (`Sun <= -18°`), Moon illumination/rise/set, and 30-min object altitude curves (`Alt >= 20°`) | Local Python (`Skyfield` / IAU GMST) |
| **Open-Meteo & Nominatim (`world.py`)** | Live APIs (+ Offline Fixtures) | Geocodes city names (`Fontainebleau`, `Chamonix`, `Paris`) and fetches cloud cover, wind speed, and Bortle light-pollution class | Open-Meteo & OpenStreetMap |

**Corpus composition (`506` documents total):**
- **`110` Messier objects (`M001`–`M110`)**: The brightest classic deep-sky galaxies, nebulae, and star clusters.
- **`109` Caldwell objects (`C001`–`C109`)**: Includes southern-sky test targets like `C080` (*Omega Centauri*) to verify horizon filtering from northern cities.
- **`194` Bright Herschel / NGC targets (`NGC*`)**: Curated deep-sky targets with `magnitude <= 10.5`.
- **`88` IAU Constellations (`CONST_*`)** & **`5` Practical Observing Guides (`GUIDE_*`)**.

**What we learned from inspecting 20 random documents & 20 random chunks (`seed=42`):**
Before selecting a chunking strategy, we manually inspected 20 random documents (median `4` sections and `4` chunks per document) and 20 random chunks (median `186` words). We observed that Wikipedia *Observation* and *Characteristics* sections drop the catalog designation (`M31`, `NGC 869`) after the lead paragraph and use pronouns (*"the galaxy"*, *"the cluster"*). Splitting on `== Section ==` boundaries and prepending a compact catalog header (`[DOC_ID | Title | Type | Constellation | Mag | Size | Aliases]`) to every chunk prevents orphan passages and boosts `Recall@5` from `0.747` to `0.799` (`MRR@5` from `0.726` to `0.846`).

### Quick astronomy glossary (for non-specialists)

| Term / Code | Plain-English Meaning | Example in Demo |
| :--- | :--- | :--- |
| **`M` (`Messier 1`–`110`)** | 18th-century catalog by French astronomer Charles Messier listing the **110 brightest deep-sky objects** visible from Europe. | **`M31`** = *Andromeda Galaxy* (our nearest giant spiral galaxy, 2.5 million light-years away). |
| **`C` (`Caldwell 1`–`109`)** | Modern list of **109 showcase objects** compiled by Patrick Moore covering both northern and southern skies. | **`C14`** = *Double Cluster in Perseus*; **`C80`** = *Omega Centauri* (southern sky trap, invisible from Paris). |
| **`NGC` (`New General Catalogue`)** | Professional catalog of ~7,840 nebulae and galaxies. Most Messier/Caldwell objects also have an `NGC` alias. | `M31` is also `NGC 224`; `C14` is `NGC 869` + `NGC 884`. |
| **Magnitude (`mag`)** | Inverted brightness scale (**lower number = brighter**): `mag <= 6` is visible to the naked eye, `6–8.5` with binoculars, `> 8.5` needs a telescope. | **`M45` (*Pleiades*)** is `mag 1.6` (naked eye); **`M27` (*Dumbbell Nebula*)** is `mag 7.4` (binoculars). |
| **Angular Size (`'` arcminutes)** | Apparent width in the sky. As a visual ruler, the **Full Moon is `30'` (`0.5°`) wide**. | **`M31`** spans `178'` ($\approx 6\times$ the width of the Full Moon!). |
| **Altitude (`0°`–`90°`)** | Height above the horizon (`0°` = ground level, `90°` = zenith straight overhead). Below `20°` is blocked by trees, buildings, and thick air. | Astropilote enforces **`Altitude >= 20°`** during astronomical darkness (`Sun <= -18°`). |

---

## LLM modes (`API` vs `Local Open-Weight`)

Set in `.env` or toggle live in the left sidebar of the web app:

- **API Mode (`LLM_MODE=api`)**: Calls Anthropic (`anthropic/claude-haiku-5.5` via OpenRouter or direct Anthropic API key in `ANTHROPIC_API_KEY`).
- **Local Mode (`LLM_MODE=local`)**: Runs a 100% local open-weight instruct model (`HuggingFaceTB/SmolLM2-360M-Instruct` in `float16` on Apple Silicon `MPS` / CPU, or local `Ollama` `gemma4` if running) at `$0.00` cost.

| Metric (`eval/golden.jsonl`, $N=40$) | ☁️ API Model (`anthropic/claude-haiku-5.5`) | 💻 Local Open-Weight (`SmolLM2-360M-Instruct` MPS) |
| :--- | :---: | :---: |
| **Retrieval Recall / MRR** | `0.9923` / `0.9083` | `0.9923` / `0.9083` |
| **Trap & Out-of-Corpus Refusal (`Prudence`)** | **`1.0000` (`100%`)** | **`1.0000` (`100%`)** |
| **Judged Correctness (Overall)** | **`0.9509` (`95.1%`)** | **`0.7635` (`76.4%`)** |
| — *Factual Lookups ($n=14$)* | `99.9%` | `69.4%` |
| — *Session Planning + Ephemeris ($n=10$)* | `87.2%` | `57.5%` |
| — *Ambiguous / Open-Ended ($n=2$)* | `66.5%` | `53.8%` |
| — *Traps & Guardrails ($n=14$)* | `100.0%` | `100.0%` |
| **Median / p95 Latency (`s`)** | `3.65s` / `4.73s` | **`3.45s` / `3.87s`** |
| **Avg Tokens (`in` / `out`)** | `2,292` / `585` | `1,532` / `334` |
| **Total Cost (`40` queries / per query)** | **`$0.1669`** (`$0.0042/q`) | **`$0.0000`** (`Free`) |

---

## Models & stack used

- **Embeddings (Dense)**: `BAAI/bge-small-en-v1.5` (`384`-d normalized vectors stored in local `Qdrant`)
- **Lexical Search**: `bm25s` lexical index with preserved catalog identifiers (`M31`, `NGC 7000`, `C14`)
- **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranks top RRF candidates)
- **Ephemeris & Environment**: `Skyfield` / IAU spherical astronomy, `Open-Meteo`, and Bortle sky brightness estimator

---

## Why hybrid retrieval + reranking helps

Evaluated on `40` golden astronomy queries (`eval/golden.jsonl`):

| Pipeline Stage | Recall@5 | MRR@5 | Why it matters |
| :--- | :---: | :---: | :--- |
| **Chunking: Fixed (No Header)** | `0.747` | `0.726` | Splits mid-section and loses object identity context |
| **Chunking: Section + Header** | **`0.799`** | **`0.846`** | Preserves `[M031 \| Andromeda \| Mag 3.4]` headers per chunk |
| **Dense Only (`BGE-small`)** | `0.728` | `0.837` | Strong on semantic descriptions (*"face-on spiral galaxy"*) |
| **BM25 Only (Lexical)** | `0.799` | `0.846` | Strong on exact catalog codes (`NGC 869`, `C80`, `M27`) |
| **Hybrid (`RRF`) + Reranker** | **`0.819`** | **`0.811`** | Combines exact catalog codes with semantic phrasing |
| **Full Agent (`Tools + RAG + Claude Haiku 4.5`)** | **`0.992`** | **`0.951`*** | Adds deterministic catalog filtering + ephemeris verification (`100%` refusal on traps, *judged correctness) |

Run the benchmark suite (`--llm-mode api`, `local`, or `offline`):

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ --llm-mode api
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/results_local.jsonl --llm-mode local
```

---

## Error analysis & residual risks (`eval/results.jsonl`)

Inspecting the per-question evaluation logs (`eval/results.jsonl` and `eval/results_local.jsonl`) highlights four concrete failure modes and residual risks:

1. **Multi-target constellation crowding (`q017`, Recall = `0.80`)**: When asked for all emission nebulae and supernova remnants in Cygnus (`C020`, `C033`, `C034`, `C027`, `CON_CYG`), top-5 hybrid retrieval returns `NGC6995` (a sub-filament of the Eastern Veil) and `C030`, pushing `C020` (*North America Nebula*) to rank 6. *Mitigation*: Deduplicate sub-components of the same nebula complex (`C033`/`NGC6995`) during RRF fusion.
2. **Citation cap on open-ended / ambiguous queries (`q025`, `q026`, Score = `0.665`)**: On subjective prompts (*"What is the most beautiful galaxy in the sky?"*), all gold objects (`M031`, `M051`, `M104`) are retrieved (`Recall = 1.00`), but `compose_node` caps `cited_ids` at 5 and includes `CON_*` constellation pages ahead of the third Messier object. *Mitigation*: Prioritize deep-sky object documents (`M*`, `C*`, `NGC*`) ahead of `CON_*` pages when ranking `cited_ids`.
3. **Single-slot vs. specific-hour ephemeris formatting (`q015`, `q024`)**: While `visibility` computes altitudes every 30 minutes across the entire night, `structured_reference` passes only the single assigned observation slot (`30` min) into the LLM prompt. When a user asks about a specific timestamp (`23:00 UTC` in `q024`), `claude-haiku-5.5` conservatively notes that only the assigned slot's altitude is shown in the summary.
4. **Local 360M model context saturation (`SmolLM2-360M-Instruct`, `76.4%` vs `95.1%`)**: On 5-document planning prompts, the 360M local model occasionally echoes raw metadata headers (`SurfBr:`) or omits secondary `[DOC_ID]` citations, whereas a 4B–8B local model via Ollama (`gemma4`) or `claude-haiku-5.5` synthesizes all 5 targets cleanly.

---

## Data & RAG visualizations notebook

Open [`notebooks/corpus_and_rag_visualizations.ipynb`](notebooks/corpus_and_rag_visualizations.ipynb) to view pre-rendered presentation charts:
1. **20 Random Documents & Chunks Inspection** + **All-Sky Mollweide Map (J2000)** of the 418 Messier, Caldwell, and NGC targets.
2. **Corpus Breakdown & Visual Magnitude vs. Angular Size** scatter plot (`506` docs / `2,659` chunks).
3. **Real-Time Night Altitude Trajectories** from Fontainebleau (`48.40°N`) with the `20°` observability floor.
4. **Chunking & Hybrid Retrieval Ablation Charts** (`Recall@5` & `MRR@5`).
5. **API (`claude-haiku-5.5`) vs. Local Open-Weight (`SmolLM2-360M-Instruct` MPS) Quality, Latency & Cost Comparison** + **Error Analysis Breakdown**.

---

## Example questions (tested)

1. *"Saturday night, October 17, 2026, I will be in Fontainebleau as a beginner with binoculars or a small telescope. What are the best objects to see in the sky tonight and where should I look?"*
2. *"What is the Andromeda Galaxy (M31), what are its two brightest companion galaxies (M32 and M110), and can a beginner see them with the naked eye or binoculars?"*
3. *"I am in Chamonix on August 15, 2026. Which bright glowing nebulae or star clouds in the constellation Cygnus can I observe tonight?"*
4. *"Can I observe Omega Centauri (Caldwell 80) from Paris on May 15, 2026 at midnight? Explain simply why or why not."*
5. *"What can you observe of the supergiant elliptical galaxy IC 1101 in Abell 2029 with an amateur telescope?"* *(Out-of-corpus guard)*
6. *"Will the sky be completely cloud-free in Fontainebleau on March 3, 2028 at 23:00?"* *(>14-day forecast refusal guard)*
