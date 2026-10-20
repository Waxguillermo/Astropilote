# Data Card — AstroPilote Corpus v2 (`567` Astronomical Documents)

## 1. Overview & Motivation

**AstroPilote** combines encyclopedic text descriptions (for visual appearance, star-hopping tips, and astrophotography framing/filter guidance) with structured astrometric metadata (for exact altitude, airmass, field-of-view, and WCS plate-solving calculations).

To satisfy the course requirement of a **coherent ~500-document corpus** with rich metadata filtering, we assemble **567 curated astronomical entities** (`447` point-localizable deep-sky objects + `88` official IAU constellations + `32` practical observing & astrophotography guides) by joining two open-data sources:

1. **English Wikipedia Astronomy Articles** (Textual RAG Corpus)
2. **OpenNGC Database** (Structured Astrometric & Photometric Metadata)

---

## 2. Corpus Composition (`567` Documents)

| Subset | Count | Identifier Prefix | Selection Rationale |
| :--- | :---: | :--- | :--- |
| **Messier Catalog** | 110 | `M001` – `M110` | The foundational northern/mid-latitude deep-sky catalog (including `M102` / `NGC 5866` *Spindle Galaxy*); primary targets for amateur visual astronomy and wide-to-medium field astrophotography. |
| **Caldwell Catalog** | 109 | `C001` – `C109` | Patrick Moore's complement to Messier (e.g., North America Nebula `C020` / `NGC7000`, Double Cluster `C014`, Veil Nebula `C033`, Omega Centauri `C080`), with manual J2000 centres filled for `C009`, `C041`, and `C099` so every sky object has coordinates. |
| **Herschel 400 / NGC Subset** | 200 | `NGCxxxx` | The 200 brightest non-Messier/non-Caldwell deep-sky objects from OpenNGC ($m_v \le 11.5$, single/double stars excluded). |
| **Astrophoto Subset (v2)** | 28 | `ICxxxx`, `NGCxxxx` | Iconic imaging targets without visual magnitude in OpenNGC (`IC0434` Horsehead, `IC1805` Heart, `IC1848` Soul, `IC5070` Pelican, `NGC2024` Flame, `NGC3628` Hamburger Galaxy…). |
| **IAU Constellations** | 88 | `CON_ORI`, `CON_CYG`, … | All 88 official International Astronomical Union constellations, providing sky-region context, seasonal visibility, and star-hopping references. |
| **Practical Guides (v2)** | 32 | `GUIDE_BORTLE`, … | Curated English Wikipedia observing & astrophotography guides (`GUIDE_BORTLE`, `GUIDE_DARK_FRAME`, `GUIDE_EQUATORIAL_MOUNT`, `GUIDE_AVERTED_VISION`, `GUIDE_HYDROGEN_ALPHA`…). |
| **Total** | **`567` docs** | | Complete reference corpus (`447` localizable DSOs + `88` constellations + `32` guides). |

---

## 3. Data Sources & Licensing

| Source | Role | License | URL / Provenance |
| :--- | :--- | :--- | :--- |
| **English Wikipedia** | Full article text (sections: *Lead*, *Observation*, *Characteristics*, *Imaging*) | **CC BY-SA 4.0** | MediaWiki REST / Action API (`en.wikipedia.org`) |
| **OpenNGC** (Mattia Verga) | RA/Dec (J2000), object type, visual & blue magnitudes, major/minor axes (`arcmin`), surface brightness, constellation, common names | **CC-BY-SA-4.0** | `https://github.com/mattiaverga/OpenNGC` (`NGC.csv` + `addendum.csv`) |

*Collection date and API revision IDs are recorded in `data/processed/corpus_manifest.json` during ingestion.*

---

## 4. Document & Chunk Schema

Each chunk indexed in BM25 and Qdrant carries:

- `chunk_id` (`str`): Unique chunk identifier, e.g., `M031#observation-0`
- `doc_id` (`str`): Canonical object identifier used as `gold_docs` in `eval/golden.jsonl` (e.g., `M031`, `C014`, `NGC7000`, `CON_ORI`)
- `title` (`str`): Primary name (e.g., `"Andromeda Galaxy (Messier 31)"`)
- `section` (`str`): Article section title (e.g., `"Observation"`, `"Characteristics"`)
- `text` (`str`): Context-enriched passage (prepended with object header + cleaned section text, $\le 400$ tokens)
- **Structured Metadata (for Qdrant payload filtering & agent tools)**:
  - `catalog` (`Literal["Messier", "Caldwell", "Herschel", "Constellation", "Astrophoto", "Guide"]`)
  - `object_type` (`str`): `"galaxy"`, `"emission_nebula"`, `"planetary_nebula"`, `"globular_cluster"`, `"open_cluster"`, `"supernova_remnant"`, `"constellation"`, `"guide"`
  - `constellation` (`str`): 3-letter IAU code (e.g., `"AND"`, `"HER"`, `"CYG"`)
  - `ra_deg` (`float | None`): Right Ascension in decimal degrees (J2000)
  - `dec_deg` (`float | None`): Declination in decimal degrees (J2000)
  - `magnitude` (`float | None`): Apparent visual magnitude ($m_v$)
  - `surface_brightness` (`float | None`): Mag / $\text{arcsec}^2$ (critical for astrophotography exposure planning)
  - `size_major_arcmin` (`float | None`): Major angular diameter in arcminutes (critical for telescope focal length / sensor FOV framing)
  - `size_minor_arcmin` (`float | None`): Minor angular diameter in arcminutes
  - `narrowband_friendly` (`bool`): `True` for emission/planetary nebulae & supernova remnants (`H-alpha` / `OIII` / dual-band filter targets that can be imaged even under moonlight)

---

## 5. Cleaning & Preprocessing Rules

1. **Boilerplate Removal**: Strip Wikipedia citations (`[1]`, `[citation needed]`), infobox raw wikitext, navigation templates, *"See also"*, *"References"*, *"External links"*, and coordinate macros.
2. **Identifier Normalization**: Canonicalize Messier (`M1` $\rightarrow$ `M001`, `M31` $\rightarrow$ `M031`) and Caldwell (`C14` $\rightarrow$ `C014`) IDs while storing all aliases (`["M31", "M031", "NGC 224", "Andromeda Galaxy"]`) in metadata and the chunk header so both BM25 and dense search match any designation.
3. **Deduplication**: Objects belonging to both Messier/Caldwell and NGC are indexed once under their primary Messier (`Mxxx`) or Caldwell (`Cxxx`) ID, with their `NGC` designation preserved as an alias.

---

## 6. Corpus v2 Extension — Astrophotography Targets & Practical Guides

**Motivation.** Inspecting v1 showed that the brightness-based `Herschel` selection silently drops most diffuse nebulae, because OpenNGC gives them no visual magnitude. The assistant is meant to answer *"what should I photograph tonight?"*, yet iconic imaging targets (Horsehead, Heart, Soul, Pelican, Flame…) were missing, and the corpus could not answer *how* to observe or image (light pollution, mounts, calibration frames, filters).

| Change | Count | Identifier | Source | Rule |
| :--- | :---: | :---: | :--- | :--- |
| **Astrophoto subset** (new) | 28 | OpenNGC name, e.g. `IC0434`, `NGC2024` | OpenNGC (astrometry) + English Wikipedia (text) | Curated list `ASTROPHOTO_TARGETS` in `ingest/catalog.py`: famous nebulae without V-Mag + classic galaxies just below the bright-NGC cut. Object type overridden when OpenNGC only says `Neb`. |
| **Guide subset** (new) | 32 | `GUIDE_*`, e.g. `GUIDE_BORTLE` | English Wikipedia | Curated list `OBSERVING_GUIDES`: sky quality, visual technique, equipment, astrophotography workflow. Only the first 6 sections are kept (`MAX_GUIDE_SECTIONS`). Topics refused by the guard (collimation, sensor datasheets, exoplanets) are deliberately left out. |
| **M102** (fix) | +1 | `M102` | OpenNGC `NGC5866` + Wikipedia "NGC 5866" | OpenNGC has no Messier number for M102; it is added as the Spindle Galaxy (most common identification, still debated). |
| **Bright-NGC subset** (fix) | 200 | `NGCxxxx` | OpenNGC | Stars, double stars and untyped rows are excluded (v1 contained 11 stars, e.g. NGC 1990). The next brightest real objects take their place. |
| **C009, C041, C099** (fix) | 3 | `Cxxx` | Manual (approximate, to check against the Wikipedia infobox) | These had no OpenNGC coordinates or constellation (`VAR`); approximate J2000 centres are now filled in so visibility can be computed. |

**Expected total:** 567 documents (110 Messier + 109 Caldwell + 200 bright NGC + 28 Astrophoto + 88 constellations + 32 guides), minus any guide whose Wikipedia article cannot be fetched. The exact totals, chunk counts and the list of documents whose text had to be synthesized are written by the ingestion run to `data/processed/corpus_manifest.json` (`catalog_breakdown`, `synthetic_fallback_count`, `synthetic_fallback_doc_ids`, `skipped_guides`).

**Guide-specific rules.**
- A guide is **never synthesized**: if Wikipedia returns nothing usable, the guide is skipped and listed in `skipped_guides` (an invented "how-to" text would be a hallucination source).
- Guide chunks carry a header without sky fields: `[GUIDE_BORTLE | Bortle scale (Observing guide) | Type: practical observing guide | Aliases: …]`.
- Retrieval: object searches exclude the `Guide` catalog; the agent runs a separate `search_catalog(catalog="Guide")` when the question is about technique, equipment or sky quality, so guides never push named objects down the ranking.

**Reproducing v1 for ablations:** `uv run python -m astropilote.ingest.run_ingest --no-astrophoto --no-guides`.

**Evaluation:** 10 extra questions in `eval/golden_extension.jsonl` (4 new objects, 4 guides, 1 planning, 1 out-of-corpus trap: IC 2118 must still be refused).
