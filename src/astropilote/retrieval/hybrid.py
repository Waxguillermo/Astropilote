"""Hybrid retrieval (BM25 + Qdrant Dense), Reciprocal Rank Fusion (RRF), and Cross-Encoder Reranking."""

from __future__ import annotations

import atexit
import logging
import math
import os
import re
import threading
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from astropilote.agent.schemas import CatalogName, Chunk, ChunkMetadata
from astropilote.config import settings
from astropilote.device import get_torch_device
from astropilote.index.build_index import QDRANT_COLLECTION

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

try:
    import bm25s
except ImportError:  # pragma: no cover
    bm25s = None  # type: ignore[assignment]

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range
except ImportError:  # pragma: no cover
    QdrantClient = Any  # type: ignore[assignment,misc]
    FieldCondition = Filter = MatchAny = MatchValue = Range = Any  # type: ignore[assignment,misc]

_GALAXY_OBJECT_TYPES: tuple[str, ...] = ("galaxy", "galaxy_pair", "galaxy_triplet")

_CHUNK_HEADER_RE = re.compile(r"^\[[^\n]*\]\s*\n(?:Section:\s*[^\n]*\n)?")
_FR_ACCENT_RE = re.compile(r"[àâäçéèêëîïôöùûüÿœæ]", re.IGNORECASE)
_FR_ELISION_RE = re.compile(r"\b(?:l|d|qu|c|n|s|j)['’](?=\w)", re.IGNORECASE)

_FR_INDICATOR_WORDS: frozenset[str] = frozenset(
    {
        "le",
        "la",
        "les",
        "de",
        "du",
        "des",
        "quel",
        "quelle",
        "quels",
        "quelles",
        "pourquoi",
        "comment",
        "est-ce",
        "nebuleuse",
        "nébuleuse",
        "etoile",
        "étoile",
        "galaxie",
        "amas",
        "lunette",
        "telescope",
        "télescope",
        "pose",
        "poses",
        "darks",
        "biais",
        "monture",
        "oculaire",
        "jumelles",
        "empilement",
    }
)

_FR_STOPWORDS: frozenset[str] = frozenset(
    {
        "de",
        "la",
        "le",
        "les",
        "du",
        "des",
        "un",
        "une",
        "et",
        "ou",
        "dans",
        "sur",
        "pour",
        "avec",
        "par",
        "est",
        "ce",
        "cette",
        "ces",
        "que",
        "qui",
        "quoi",
        "quel",
        "quelle",
        "quels",
        "quelles",
        "pourquoi",
        "comment",
        "quand",
        "combien",
        "au",
        "aux",
        "ne",
        "pas",
        "sont",
        "faut",
        "peut",
        "qu",
        "l",
        "d",
    }
)


@dataclass
class MetadataFilter:
    """Optional structured filters applied during catalog search."""

    catalog: CatalogName | None = None
    object_type: str | None = None
    constellation: str | None = None
    max_magnitude: float | None = None
    min_size_arcmin: float | None = None
    narrowband_only: bool = False
    # e.g. "Guide": keep practical guides out of object searches (they have their own search)
    exclude_catalog: CatalogName | None = None

    def matches(self, meta: ChunkMetadata) -> bool:
        """Evaluate whether a chunk's metadata satisfies all active filter criteria."""
        if self.catalog and meta.catalog.lower() != self.catalog.lower():
            return False
        if self.exclude_catalog and meta.catalog.lower() == self.exclude_catalog.lower():
            return False
        if self.object_type:
            wanted_type = self.object_type.lower()
            actual_type = meta.object_type.lower()
            if wanted_type == "galaxy":
                if actual_type not in _GALAXY_OBJECT_TYPES:
                    return False
            elif actual_type != wanted_type:
                return False
        if self.constellation and meta.constellation.upper() != self.constellation.upper():
            return False
        if self.max_magnitude is not None:
            if meta.magnitude is None or meta.magnitude > self.max_magnitude:
                return False
        if self.min_size_arcmin is not None:
            if meta.size_major_arcmin is None or meta.size_major_arcmin < self.min_size_arcmin:
                return False
        if self.narrowband_only and not meta.narrowband_friendly:
            return False
        return True

    def to_qdrant_filter(self) -> Filter | None:
        """Convert active criteria into a Qdrant `Filter` clause."""
        must: list[FieldCondition] = []
        if self.catalog:
            must.append(FieldCondition(key="catalog", match=MatchValue(value=self.catalog)))
        if self.object_type:
            wanted_type = self.object_type.lower()
            if wanted_type == "galaxy":
                must.append(
                    FieldCondition(
                        key="object_type",
                        match=MatchAny(any=list(_GALAXY_OBJECT_TYPES)),
                    )
                )
            else:
                must.append(FieldCondition(key="object_type", match=MatchValue(value=wanted_type)))
        if self.constellation:
            must.append(
                FieldCondition(
                    key="constellation", match=MatchValue(value=self.constellation.upper())
                )
            )
        if self.max_magnitude is not None:
            must.append(FieldCondition(key="magnitude", range=Range(lte=self.max_magnitude)))
        if self.min_size_arcmin is not None:
            must.append(
                FieldCondition(key="size_major_arcmin", range=Range(gte=self.min_size_arcmin))
            )
        if self.narrowband_only:
            must.append(FieldCondition(key="narrowband_friendly", match=MatchValue(value=True)))
        must_not: list[FieldCondition] = []
        if self.exclude_catalog:
            must_not.append(
                FieldCondition(key="catalog", match=MatchValue(value=self.exclude_catalog))
            )
        if not must and not must_not:
            return None
        return Filter(must=must or None, must_not=must_not or None)


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[Chunk]:
    """Fuse multiple ranked chunk lists using Reciprocal Rank Fusion (RRF, default k=60).

    Formula:
        score(d) = sum_{L in ranked_lists} 1 / (rrf_k + rank_L(d))
    where `rank_L(d)` is 1-indexed (1, 2, 3, ...).
    """
    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for chunk_list in ranked_lists:
        for rank_idx, chunk in enumerate(chunk_list, start=1):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank_idx))
            if cid not in chunk_by_id:
                chunk_by_id[cid] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    fused: list[Chunk] = []
    for cid in sorted_ids:
        ch = chunk_by_id[cid].model_copy(deep=True)
        ch.score = round(rrf_scores[cid], 6)
        fused.append(ch)
    return fused


_SHARED_QDRANT_CLIENTS: dict[str, QdrantClient] = {}
_QDRANT_CLIENTS_LOCK = threading.Lock()


def _safe_close_qdrant_client(client: Any) -> None:
    """Close a QdrantClient cleanly and disarm its `__del__` finalizer before interpreter shutdown."""
    if client is None:
        return
    try:
        client.close()
    except (OSError, RuntimeError, AttributeError):
        pass
    try:
        inner = getattr(client, "_client", None)
        if inner is not None and hasattr(inner, "_flock_file"):
            inner._flock_file = None
    except (OSError, RuntimeError, AttributeError):
        pass


def _close_all_shared_qdrant_clients() -> None:
    """atexit callback: close all cached Qdrant clients while `sys.meta_path` is still active."""
    with _QDRANT_CLIENTS_LOCK:
        clients = list(_SHARED_QDRANT_CLIENTS.values())
        _SHARED_QDRANT_CLIENTS.clear()
    for client in clients:
        _safe_close_qdrant_client(client)


atexit.register(_close_all_shared_qdrant_clients)


_FR_TO_EN_ASTRO_MAP: tuple[tuple[str, str], ...] = (
    # Deep-sky objects (French common names -> canonical English & catalog IDs)
    ("amérique du nord", "North America Nebula C020 NGC7000 NGC 7000 Deneb Cygnus"),
    ("amerique du nord", "North America Nebula C020 NGC7000 NGC 7000 Deneb Cygnus"),
    ("galaxie d'andromède", "Andromeda Galaxy M031 M31 NGC 224"),
    ("andromède", "Andromeda M031 M31"),
    ("nébuleuse d'orion", "Orion Nebula M042 M42"),
    ("pléiades", "Pleiades M045 M45 Seven Sisters"),
    ("anneau de la lyre", "Ring Nebula M057 M57 Lyra"),
    ("nébuleuse de la lyre", "Ring Nebula M057 M57 Lyra"),
    ("nébuleuse de l'haltère", "Dumbbell Nebula M027 M27"),
    ("amas d'hercule", "Hercules Globular Cluster M013 M13"),
    ("nébuleuse du crabe", "Crab Nebula M001 M1 supernova remnant"),
    ("galaxie du tourbillon", "Whirlpool Galaxy M051 M51 NGC 5194"),
    ("galaxie du sombrero", "Sombrero Galaxy M104"),
    ("galaxie du triangle", "Triangulum Galaxy M033 M33"),
    ("galaxie de bode", "Bode's Galaxy M081 M81"),
    ("galaxie du cigare", "Cigar Galaxy M082 M82"),
    ("galaxie du moulinet", "Pinwheel Galaxy M101"),
    ("galaxie de l'aiguille", "Needle Galaxy C038 NGC 4565"),
    ("galaxie du sculpteur", "Sculptor Galaxy C065 NGC 253"),
    ("nébuleuse de la lagune", "Lagoon Nebula M008 M8"),
    ("nébuleuse trifide", "Trifid Nebula M020 M20"),
    ("nébuleuse de l'aigle", "Eagle Nebula M016 M16"),
    ("double amas de persée", "Double Cluster C014 C14 Perseus NGC 869"),
    ("tête de cheval", "Horsehead Nebula IC0434 IC 434"),
    ("nébuleuse de la flamme", "Flame Nebula NGC2024 NGC 2024"),
    ("nébuleuse de la rosette", "Rosette Nebula C049 NGC2237 NGC 2237"),
    ("dentelles du cygne", "Veil Nebula Cygnus C033 C034 NGC 6992 NGC 6960"),
    ("pélican", "Pelican Nebula IC5070 IC 5070"),
    ("nébuleuse du coeur", "Heart Nebula IC1805 IC 1805"),
    ("nébuleuse du cœur", "Heart Nebula IC1805 IC 1805"),
    ("nébuleuse de l'âme", "Soul Nebula IC1848 IC 1848"),
    ("trompe d'éléphant", "Elephant's Trunk Nebula IC1396 IC 1396"),
    ("nébuleuse de la bulle", "Bubble Nebula C011 NGC 7635"),
    ("nébuleuse du croissant", "Crescent Nebula C027 NGC 6888"),
    ("nébuleuse de l'iris", "Iris Nebula C004 NGC 7023"),
    ("nébuleuse hélix", "Helix Nebula C063 NGC 7293"),
    ("œil de chat", "Cat's Eye Nebula C006 NGC 6543"),
    ("oeil de chat", "Cat's Eye Nebula C006 NGC 6543"),
    ("oméga du centaure", "Omega Centauri C080 NGC 5139"),
    ("amas de la crèche", "Beehive Cluster Praesepe M044 M44"),
    ("voie lactée", "Milky Way"),
    # Calibration, guides & observing techniques
    ("dark frame", "dark frame calibration thermal noise hot pixels GUIDE_DARK_FRAME"),
    ("darks", "dark frame calibration thermal noise hot pixels GUIDE_DARK_FRAME"),
    ("flats", "flat field correction vignetting dust calibration GUIDE_FLAT_FIELD"),
    ("flat field", "flat field correction vignetting GUIDE_FLAT_FIELD"),
    ("biais", "bias frame readout noise calibration"),
    ("offset", "bias frame readout noise calibration"),
    ("collimater", "collimation reflecting telescope mirror optical alignment"),
    ("collimation", "collimation reflecting telescope mirror optical alignment"),
    ("bortle", "Bortle scale light pollution skyglow GUIDE_BORTLE"),
    (
        "pollution lumineuse",
        "light pollution Bortle scale skyglow GUIDE_LIGHT_POLLUTION GUIDE_BORTLE",
    ),
    ("seeing", "astronomical seeing atmospheric turbulence GUIDE_SEEING"),
    ("turbulence", "astronomical seeing atmospheric turbulence GUIDE_SEEING"),
    ("masse d'air", "air mass airmass atmospheric extinction GUIDE_AIRMASS"),
    ("crépuscule", "twilight astronomical darkness GUIDE_TWILIGHT"),
    ("vision décalée", "averted vision rod cells GUIDE_AVERTED_VISION"),
    ("adaptation nocturne", "dark adaptation night vision eye GUIDE_DARK_ADAPTATION"),
    ("cheminement", "star hopping finder chart GUIDE_STAR_HOPPING"),
    ("magnitude limite", "limiting magnitude naked-eye GUIDE_LIMITING_MAGNITUDE"),
    ("brillance de surface", "surface brightness GUIDE_SURFACE_BRIGHTNESS"),
    ("planisphère", "planisphere star wheel GUIDE_PLANISPHERE"),
    ("cherche-étoiles", "planisphere star wheel GUIDE_PLANISPHERE"),
    ("marathon de messier", "Messier marathon GUIDE_MESSIER_MARATHON"),
    ("monture équatoriale", "equatorial mount polar alignment tracking GUIDE_EQUATORIAL_MOUNT"),
    ("mise en station", "polar alignment equatorial mount GUIDE_EQUATORIAL_MOUNT"),
    ("monture azimutale", "altazimuth mount alt-az GUIDE_ALTAZIMUTH_MOUNT"),
    ("monture", "telescope mount equatorial altazimuth"),
    ("autoguidage", "autoguiding"),
    ("télescope", "telescope"),
    ("newton", "Newtonian telescope reflecting mirror"),
    ("dobson", "Dobsonian telescope GUIDE_DOBSONIAN"),
    ("lunette", "refracting telescope refractor apochromat GUIDE_REFRACTOR"),
    ("jumelles", "binoculars 10x50 7x50 GUIDE_BINOCULARS"),
    ("oculaire", "eyepiece magnification exit pupil GUIDE_EYEPIECE"),
    ("grossissement", "magnification eyepiece focal length GUIDE_EYEPIECE"),
    ("barlow", "Barlow lens magnification GUIDE_BARLOW"),
    ("rapport f/d", "focal ratio f-number GUIDE_F_NUMBER"),
    ("filé d'étoiles", "star trails Earth rotation GUIDE_STAR_TRAILS"),
    ("filés d'étoiles", "star trails Earth rotation GUIDE_STAR_TRAILS"),
    ("h-alpha", "H-alpha hydrogen-alpha narrowband nebula GUIDE_HYDROGEN_ALPHA"),
    ("hydrogène alpha", "H-alpha hydrogen-alpha narrowband GUIDE_HYDROGEN_ALPHA"),
    ("oiii", "OIII doubly ionized oxygen narrowband filter GUIDE_OXYGEN_III"),
    ("bande étroite", "narrowband filter H-alpha OIII GUIDE_HYDROGEN_ALPHA GUIDE_OXYGEN_III"),
    ("astrophotographie", "astrophotography long exposure stacking GUIDE_ASTROPHOTOGRAPHY"),
    ("temps de pose", "exposure time sub-exposure astrophotography GUIDE_ASTROPHOTOGRAPHY"),
    ("pose", "exposure time astrophotography"),
    ("empilement", "image stacking calibration GUIDE_ASTROPHOTOGRAPHY"),
    ("empiler", "image stacking"),
    ("ciel profond", "deep-sky object nebula galaxy cluster"),
    ("nébuleuse planétaire", "planetary nebula OIII"),
    ("nébuleuse", "nebula"),
    ("galaxie", "galaxy"),
    ("amas globulaire", "globular cluster"),
    ("amas ouvert", "open cluster"),
    ("amas", "star cluster"),
    ("étoile", "star"),
    ("caméra", "camera CMOS sensor astrophotography"),
    ("capteur", "sensor CMOS focal field of view"),
    ("filtre", "filter narrowband OIII H-alpha"),
)


def _is_french_query(query: str) -> bool:
    """Detect whether `query` is written in French using accents or French function/astronomy words."""
    q_low = query.lower()
    if _FR_ACCENT_RE.search(q_low) or "est-ce" in q_low:
        return True
    cleaned = _FR_ELISION_RE.sub(" ", q_low)
    tokens = re.findall(r"[a-zà-ÿ]+(?:-[a-zà-ÿ]+)?", cleaned)
    if any(tok in _FR_INDICATOR_WORDS for tok in tokens):
        return True
    return bool(_FR_ELISION_RE.search(q_low))


def _strip_french_stopwords(text: str) -> str:
    """Remove French elisions (`l'`, `d'`, `qu'`) and French function words that pollute English BM25."""
    cleaned = _FR_ELISION_RE.sub(" ", text)
    cleaned = re.sub(r"\best-ce\b", " ", cleaned, flags=re.IGNORECASE)
    kept: list[str] = []
    for raw_tok in cleaned.split():
        bare = raw_tok.strip(".,;:!?()[]{}\"'«»").lower()
        if bare in _FR_STOPWORDS:
            continue
        kept.append(raw_tok)
    return " ".join(kept)


def _expand_bilingual_query(query: str) -> str:
    """Append English astronomy equivalents and strip French stopword noise on francophone queries."""
    q_low = query.lower().replace("’", "'")
    is_fr = _is_french_query(query)
    additions: list[str] = []
    for fr_term, en_equiv in _FR_TO_EN_ASTRO_MAP:
        if fr_term in q_low:
            additions.append(en_equiv)
    base = _strip_french_stopwords(query) if is_fr else query
    if not additions:
        return base or query
    dedup_additions = " ".join(dict.fromkeys(" ".join(additions).split()))
    return f"{base} {dedup_additions}".strip()


def _strip_chunk_header(text: str) -> str:
    """Remove synthetic `[DOC_ID | ...]\nSection: ...` prefix before MS-MARCO Cross-Encoder scoring."""
    stripped = _CHUNK_HEADER_RE.sub("", text, count=1).strip()
    return stripped or text


def _build_bm25_query_stopwords() -> list[str] | str:
    """Combine English BM25 stopwords with French stopwords so words like `de`/`la`/`du` never match M011."""
    if bm25s is not None:
        en_sw = getattr(getattr(bm25s, "stopwords", None), "STOPWORDS_EN", None)
        if en_sw:
            return list(set(en_sw) | _FR_STOPWORDS)
    return "en"


_BM25_QUERY_STOPWORDS = _build_bm25_query_stopwords()


def _configure_quiet_hf_hub() -> None:
    """Silence HuggingFace Hub unauthenticated warnings and weight-loading progress bars."""
    try:
        import huggingface_hub.utils.logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bars()
    except (ImportError, AttributeError):
        pass
    try:
        import transformers.utils.logging as tf_logging

        tf_logging.set_verbosity_error()
        tf_logging.disable_progress_bar()
    except (ImportError, AttributeError):
        pass


class HybridRetriever:
    """Two-stage hybrid retriever: (BM25 + Qdrant Dense -> RRF) followed by Cross-Encoder Reranking."""

    def __init__(
        self,
        bm25_dir: Path | None = None,
        qdrant_dir: Path | None = None,
        embedding_model_name: str | None = None,
        reranker_model_name: str | None = None,
    ) -> None:
        self.bm25_dir = bm25_dir or settings.bm25_path
        self.qdrant_dir = qdrant_dir or settings.qdrant_path
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        self.reranker_model_name = reranker_model_name or settings.reranker_model

        self._init_lock = threading.RLock()
        self._mps_lock = threading.Lock()
        self._device: str | None = None

        self._bm25: bm25s.BM25 | None = None
        self._chunks: list[Chunk] = []
        self._chunks_by_doc: dict[str, list[Chunk]] = {}
        self._encoder = None
        self._reranker = None
        self._qdrant_client: QdrantClient | None = None

    def __enter__(self) -> HybridRetriever:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying local Qdrant client if open."""
        key = str(self.qdrant_dir.resolve())
        with _QDRANT_CLIENTS_LOCK:
            client = _SHARED_QDRANT_CLIENTS.pop(key, None) or self._qdrant_client
            self._qdrant_client = None
        _safe_close_qdrant_client(client)

    def _get_device(self) -> str:
        if self._device is None:
            self._device = get_torch_device()
        return self._device

    def _get_qdrant_client(self) -> QdrantClient:
        key = str(self.qdrant_dir.resolve())
        with _QDRANT_CLIENTS_LOCK:
            existing = _SHARED_QDRANT_CLIENTS.get(key)
            if existing is None or getattr(getattr(existing, "_client", None), "closed", False):
                _SHARED_QDRANT_CLIENTS[key] = QdrantClient(path=str(self.qdrant_dir))
            self._qdrant_client = _SHARED_QDRANT_CLIENTS[key]
            return self._qdrant_client

    def _ensure_bm25_loaded(self) -> None:
        if self._bm25 is not None or (bm25s is None and self._chunks):
            return
        with self._init_lock:
            if self._bm25 is not None or (bm25s is None and self._chunks):
                return
            meta_path = self.bm25_dir / "chunks.jsonl"
            if not meta_path.exists() or meta_path.stat().st_size == 0:
                chunks_src = settings.data_dir / "processed" / "chunks.jsonl"
                if not chunks_src.exists() or chunks_src.stat().st_size == 0:
                    from astropilote.ingest.run_ingest import run_pipeline

                    run_pipeline(offline_fallback=True)
                if chunks_src.exists() and chunks_src.stat().st_size > 0:
                    from astropilote.index.build_index import build_bm25_index, load_chunks

                    build_bm25_index(load_chunks(chunks_src), self.bm25_dir)
                else:
                    raise FileNotFoundError(
                        f"BM25 index not found at {self.bm25_dir}. Run `uv run python -m astropilote.index.build_index`."
                    )

            with meta_path.open("r", encoding="utf-8") as f:
                chunks_local = [Chunk.model_validate_json(line) for line in f if line.strip()]
            chunks_by_doc_local: dict[str, list[Chunk]] = {}
            for ch in chunks_local:
                chunks_by_doc_local.setdefault(ch.doc_id, []).append(ch)

            bm25_local = None
            if bm25s is not None and (self.bm25_dir / "vocab.index.json").exists():
                bm25_local = bm25s.BM25.load(str(self.bm25_dir), mmap=False)

            # Assign `_chunks` and `_chunks_by_doc` BEFORE `_bm25` for lock-free reader safety
            self._chunks = chunks_local
            self._chunks_by_doc = chunks_by_doc_local
            self._bm25 = bm25_local

    def get_doc_chunks(self, doc_id: str) -> list[Chunk]:
        """Return all indexed chunks belonging to `doc_id` in O(1) lookup time."""
        self._ensure_bm25_loaded()
        return [ch.model_copy(deep=True) for ch in self._chunks_by_doc.get(doc_id, ())]

    def has_dense_index(self) -> bool:
        """Return True if the local Qdrant vector database exists on disk."""
        return (self.qdrant_dir / "collection" / QDRANT_COLLECTION).exists() or (
            self.qdrant_dir / "meta.json"
        ).exists()

    def _ensure_encoder_loaded(self):
        if self._encoder is not None:
            return self._encoder
        with self._init_lock:
            if self._encoder is None:
                _configure_quiet_hf_hub()
                from sentence_transformers import SentenceTransformer

                model_file = self.qdrant_dir / "model_name.txt"
                model_name = (
                    model_file.read_text(encoding="utf-8").strip()
                    if model_file.exists()
                    else self.embedding_model_name
                )
                device = self._get_device()
                if device == "mps":
                    with self._mps_lock:
                        try:
                            self._encoder = SentenceTransformer(
                                model_name, device=device, local_files_only=True
                            )
                        except (OSError, ValueError):
                            self._encoder = SentenceTransformer(model_name, device=device)
                else:
                    try:
                        self._encoder = SentenceTransformer(
                            model_name, device=device, local_files_only=True
                        )
                    except (OSError, ValueError):
                        self._encoder = SentenceTransformer(model_name, device=device)
        return self._encoder

    def _ensure_reranker_loaded(self):
        if self._reranker is not None:
            return self._reranker
        with self._init_lock:
            if self._reranker is None:
                try:
                    _configure_quiet_hf_hub()
                    from sentence_transformers import CrossEncoder

                    device = self._get_device()
                    if device == "mps":
                        with self._mps_lock:
                            try:
                                self._reranker = CrossEncoder(
                                    self.reranker_model_name, device=device, local_files_only=True
                                )
                            except (OSError, ValueError):
                                self._reranker = CrossEncoder(
                                    self.reranker_model_name, device=device
                                )
                    else:
                        try:
                            self._reranker = CrossEncoder(
                                self.reranker_model_name, device=device, local_files_only=True
                            )
                        except (OSError, ValueError):
                            self._reranker = CrossEncoder(self.reranker_model_name, device=device)
                except ImportError:
                    return None
        return self._reranker

    def search_bm25(
        self,
        query: str,
        top_k: int = 20,
        filters: MetadataFilter | None = None,
    ) -> list[Chunk]:
        """Retrieve top-k chunks by BM25 lexical score, applying optional metadata filters."""
        self._ensure_bm25_loaded()
        expanded_query = _strip_french_stopwords(_expand_bilingual_query(query))
        if not expanded_query.strip():
            expanded_query = query
        fetch_k = min(len(self._chunks), top_k * 5 if filters else top_k)
        if fetch_k <= 0:
            return []

        if self._bm25 is not None and bm25s is not None:
            q_tokens = bm25s.tokenize(
                [expanded_query],
                stopwords=_BM25_QUERY_STOPWORDS,
                show_progress=False,
            )
            results, scores = self._bm25.retrieve(q_tokens, k=fetch_k, show_progress=False)
            pairs = list(zip(results[0], scores[0], strict=True))
        else:
            q_terms = [
                t
                for t in re.findall(r"[a-z0-9]+", expanded_query.lower())
                if t not in _FR_STOPWORDS
            ]
            doc_tokens = [re.findall(r"[a-z0-9]+", ch.text.lower()) for ch in self._chunks]
            n_docs = max(len(doc_tokens), 1)
            avgdl = sum(len(toks) for toks in doc_tokens) / n_docs if doc_tokens else 1.0
            df: Counter[str] = Counter()
            for toks in doc_tokens:
                for term in set(toks):
                    df[term] += 1
            scored_pairs: list[tuple[int, float]] = []
            for idx, toks in enumerate(doc_tokens):
                tf = Counter(toks)
                dl = len(toks) or 1
                score = 0.0
                for term in q_terms:
                    if tf[term] == 0:
                        continue
                    idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                    score += idf * (tf[term] * 2.5) / (tf[term] + 1.5 * (0.25 + 0.75 * dl / avgdl))
                scored_pairs.append((idx, score))
            scored_pairs.sort(key=lambda item: item[1], reverse=True)
            pairs = scored_pairs[:fetch_k]

        hits: list[Chunk] = []
        for idx, score in pairs:
            if float(score) <= 0.0:
                break
            chunk = self._chunks[int(idx)]
            if filters and not filters.matches(chunk.metadata):
                continue
            ch_copy = chunk.model_copy(deep=True)
            ch_copy.score = round(float(score), 4)
            hits.append(ch_copy)
            if len(hits) >= top_k:
                break
        return hits

    def search_dense(
        self,
        query: str,
        top_k: int = 20,
        filters: MetadataFilter | None = None,
    ) -> list[Chunk]:
        """Retrieve top-k chunks by cosine similarity in local Qdrant with payload filtering."""
        if not self.has_dense_index():
            return []

        expanded_query = _expand_bilingual_query(query)
        encoder = self._ensure_encoder_loaded()
        if self._get_device() == "mps":
            with self._mps_lock:
                encoded = encoder.encode(
                    [expanded_query], normalize_embeddings=True, show_progress_bar=False
                )
        else:
            encoded = encoder.encode(
                [expanded_query], normalize_embeddings=True, show_progress_bar=False
            )
        q_vec = encoded[0].tolist()
        qdrant_filter = filters.to_qdrant_filter() if filters else None

        client = self._get_qdrant_client()
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=q_vec,
            query_filter=qdrant_filter,
            limit=top_k,
        )

        hits: list[Chunk] = []
        for pt in response.points:
            p = pt.payload or {}
            meta = ChunkMetadata(
                doc_id=p["doc_id"],
                catalog=p["catalog"],
                object_type=p["object_type"],
                constellation=p["constellation"],
                ra_deg=p.get("ra_deg"),
                dec_deg=p.get("dec_deg"),
                magnitude=p.get("magnitude"),
                surface_brightness=p.get("surface_brightness"),
                size_major_arcmin=p.get("size_major_arcmin"),
                size_minor_arcmin=p.get("size_minor_arcmin"),
                narrowband_friendly=bool(p.get("narrowband_friendly", False)),
                aliases=list(p.get("aliases", [])),
            )
            hits.append(
                Chunk(
                    chunk_id=p["chunk_id"],
                    doc_id=p["doc_id"],
                    title=p["title"],
                    section=p["section"],
                    text=p["text"],
                    score=round(float(pt.score), 4),
                    metadata=meta,
                )
            )
        return hits

    @staticmethod
    def _diversify_by_doc(chunks: list[Chunk], top_k: int, max_per_doc: int = 1) -> list[Chunk]:
        """Select up to `top_k` chunks while capping how many chunks come from a single `doc_id`."""
        selected: list[Chunk] = []
        overflow: list[Chunk] = []
        counts: dict[str, int] = {}
        for ch in chunks:
            c = counts.get(ch.doc_id, 0)
            if c < max_per_doc:
                selected.append(ch)
                counts[ch.doc_id] = c + 1
                if len(selected) >= top_k:
                    return selected
            else:
                overflow.append(ch)
        for ch in overflow:
            if len(selected) >= top_k:
                break
            selected.append(ch)
        return selected

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int = 5,
        max_chunks_per_doc: int = 1,
    ) -> list[Chunk]:
        """Score `(query, clean_passage)` pairs with MS-MARCO Cross-Encoder and sort directly by score."""
        if not candidates:
            return []
        reranker = self._ensure_reranker_loaded()
        if reranker is None:
            return self._diversify_by_doc(candidates, top_k=top_k, max_per_doc=max_chunks_per_doc)

        ce_query = _expand_bilingual_query(query) if _is_french_query(query) else query
        pairs = [(ce_query, _strip_chunk_header(c.text)) for c in candidates]
        if self._get_device() == "mps":
            with self._mps_lock:
                raw_scores = reranker.predict(pairs, show_progress_bar=False)
        else:
            raw_scores = reranker.predict(pairs, show_progress_bar=False)

        ce_logits = np.asarray(raw_scores, dtype=float)
        order = np.argsort(ce_logits)[::-1]

        ce_ranked: list[Chunk] = []
        for i in order:
            ch = candidates[int(i)].model_copy(deep=True)
            ch.score = round(float(ce_logits[int(i)]), 4)
            ce_ranked.append(ch)

        return self._diversify_by_doc(ce_ranked, top_k=top_k, max_per_doc=max_chunks_per_doc)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: Literal["hybrid", "bm25", "dense"] = "hybrid",
        use_reranker: bool | None = None,
        filters: MetadataFilter | None = None,
        max_chunks_per_doc: int = 1,
    ) -> list[Chunk]:
        """Execute full retrieval pipeline: BM25 / Dense / RRF Fusion + optional Reranking."""
        final_k = top_k or settings.final_top_k
        do_rerank = settings.use_reranker if use_reranker is None else use_reranker

        if mode == "bm25" or (mode == "hybrid" and not self.has_dense_index()):
            candidates = self.search_bm25(query, top_k=settings.bm25_top_k, filters=filters)
        elif mode == "dense":
            candidates = self.search_dense(query, top_k=settings.dense_top_k, filters=filters)
        else:
            bm25_hits = self.search_bm25(query, top_k=settings.bm25_top_k, filters=filters)
            dense_hits = self.search_dense(query, top_k=settings.dense_top_k, filters=filters)
            fused_all = reciprocal_rank_fusion(
                [bm25_hits, dense_hits],
                rrf_k=settings.rrf_k,
                top_k=settings.bm25_top_k + settings.dense_top_k,
            )
            candidates = self._diversify_by_doc(
                fused_all, top_k=max(settings.bm25_top_k, final_k), max_per_doc=2
            )

        if do_rerank and len(candidates) > 1:
            return self.rerank(
                query, candidates, top_k=final_k, max_chunks_per_doc=max_chunks_per_doc
            )
        return self._diversify_by_doc(candidates, top_k=final_k, max_per_doc=max_chunks_per_doc)
