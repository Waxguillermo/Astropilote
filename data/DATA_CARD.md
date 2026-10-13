# Data Card — AstroPilote Corpus (`~500` Astronomical Documents)

## 1. Overview & Motivation

**AstroPilote** combines encyclopedic text descriptions (for visual appearance, star-hopping tips, and astrophotography framing/filter guidance) with structured astrometric metadata (for exact altitude, airmass, and field-of-view calculations).

To satisfy the course requirement of a **coherent ~500-document corpus** with rich metadata filtering, we assemble **507 curated astronomical entities** by joining two open-data sources:

1. **English Wikipedia Astronomy Articles** (Textual RAG Corpus)
2. **OpenNGC Database** (Structured Astrometric & Photometric Metadata)

---

## 2. Corpus Composition (`507` Documents)

| Subset | Count | Identifier Prefix | Selection Rationale |
| :--- | :---: | :---: | :--- |
| **Messier Catalog** | 110 | `M001` – `M110` | The foundational northern/mid-latitude deep-sky catalog; primary targets for amateur visual astronomy and wide-to-medium field astrophotography. |
| **Caldwell Catalog** | 109 | `C001` – `C109` | Patrick Moore's complement to Messier (e.g., North America Nebula `C020` / `NGC7000`, Double Cluster `C014`, Veil Nebula `C033`), rich in emission nebulae prized by astrophotographers. |
| **Herschel 400 Subset** | ~200 | `NGCxxxx` | The 200 brightest non-Messier/non-Caldwell objects from the Astronomical League's Herschel 400 list, sorted by visual magnitude ($m_v \le 11.0$), providing challenging targets for 150–250 mm apertures and long-exposure rigs. |
| **IAU Constellations** | 88 | `CON_ORI`, `CON_CYG`, … | All 88 official International Astronomical Union constellations, providing sky-region context, seasonal visibility, and border/star-hopping references. |
| **Total** | **~507** | | Coherent, complete reference corpus for amateur visual & astrophotography planning. |

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
  - `catalog` (`Literal["Messier", "Caldwell", "Herschel", "Constellation"]`)
  - `object_type` (`str`): `"galaxy"`, `"emission_nebula"`, `"planetary_nebula"`, `"globular_cluster"`, `"open_cluster"`, `"supernova_remnant"`, `"constellation"`
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
