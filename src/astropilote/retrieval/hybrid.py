"""Hybrid retrieval (BM25 + Qdrant Dense), Reciprocal Rank Fusion (RRF), and Cross-Encoder Reranking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np

try:
    import bm25s
except ImportError:  # pragma: no cover
    bm25s = None  # type: ignore[assignment]

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue, Range
except ImportError:  # pragma: no cover
    QdrantClient = Any  # type: ignore[assignment,misc]
    FieldCondition = Filter = MatchValue = Range = Any  # type: ignore[assignment,misc]

from astropilote.agent.schemas import Chunk, ChunkMetadata
from astropilote.config import settings
from astropilote.index.build_index import QDRANT_COLLECTION


@dataclass
class MetadataFilter:
    """Optional structured filters applied during catalog search."""

    catalog: Literal["Messier", "Caldwell", "Herschel", "Constellation"] | None = None
    object_type: str | None = None
    constellation: str | None = None
    max_magnitude: float | None = None
    min_size_arcmin: float | None = None
    narrowband_only: bool = False

    def matches(self, meta: ChunkMetadata) -> bool:
        """Evaluate whether a chunk's metadata satisfies all active filter criteria."""
        if self.catalog and meta.catalog.lower() != self.catalog.lower():
            return False
        if self.object_type and meta.object_type.lower() != self.object_type.lower():
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
            must.append(
                FieldCondition(key="object_type", match=MatchValue(value=self.object_type.lower()))
            )
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
        return Filter(must=must) if must else None


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

        self._bm25: bm25s.BM25 | None = None
        self._chunks: list[Chunk] = []
        self._encoder = None
        self._reranker = None
        self._qdrant_client: QdrantClient | None = None

    def close(self) -> None:
        """Close the underlying local Qdrant client if open."""
        key = str(self.qdrant_dir.resolve())
        client = _SHARED_QDRANT_CLIENTS.pop(key, None) or self._qdrant_client
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        self._qdrant_client = None

    def _get_qdrant_client(self) -> QdrantClient:
        key = str(self.qdrant_dir.resolve())
        if key not in _SHARED_QDRANT_CLIENTS:
            try:
                _SHARED_QDRANT_CLIENTS[key] = QdrantClient(path=str(self.qdrant_dir))
            except RuntimeError as exc:
                if "already accessed by another instance" not in str(exc):
                    raise
                import portalocker

                orig_lock = portalocker.lock
                portalocker.lock = lambda *_a, **_kw: None
                try:
                    _SHARED_QDRANT_CLIENTS[key] = QdrantClient(
                        path=str(self.qdrant_dir), force_disable_check_same_thread=True
                    )
                finally:
                    portalocker.lock = orig_lock
        self._qdrant_client = _SHARED_QDRANT_CLIENTS[key]
        return self._qdrant_client

    def _ensure_bm25_loaded(self) -> None:
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
                    f"BM25 index not found at {self.bm25_dir}. Run `python -m astropilote.index.build_index`."
                )
        if bm25s is not None and (self.bm25_dir / "vocab.index.json").exists():
            self._bm25 = bm25s.BM25.load(str(self.bm25_dir), mmap=False)
        with meta_path.open("r", encoding="utf-8") as f:
            self._chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]

    def has_dense_index(self) -> bool:
        """Return True if the local Qdrant vector database exists on disk."""
        return (self.qdrant_dir / "collection" / QDRANT_COLLECTION).exists() or (
            self.qdrant_dir / "meta.json"
        ).exists()

    def _ensure_encoder_loaded(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            from astropilote.index.build_index import get_torch_device

            model_file = self.qdrant_dir / "model_name.txt"
            model_name = (
                model_file.read_text(encoding="utf-8").strip()
                if model_file.exists()
                else self.embedding_model_name
            )
            self._encoder = SentenceTransformer(model_name, device=get_torch_device())
        return self._encoder

    def _ensure_reranker_loaded(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                from astropilote.index.build_index import get_torch_device

                self._reranker = CrossEncoder(self.reranker_model_name, device=get_torch_device())
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
        fetch_k = min(len(self._chunks), top_k * 5 if filters else top_k)

        if self._bm25 is not None and bm25s is not None:
            q_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
            results, scores = self._bm25.retrieve(q_tokens, k=fetch_k, show_progress=False)
            pairs = list(zip(results[0], scores[0], strict=True))
        else:
            q_terms = re.findall(r"[a-z0-9]+", query.lower())
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
            if float(score) <= 0.0 and len(hits) >= top_k:
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

        encoder = self._ensure_encoder_loaded()
        q_vec = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
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
        """Score `(query, chunk.text)` pairs with a Cross-Encoder and fuse with Stage-1 RRF rank."""
        if not candidates:
            return []
        reranker = self._ensure_reranker_loaded()
        if reranker is None:
            return self._diversify_by_doc(
                candidates, top_k=top_k, max_per_doc=max_chunks_per_doc
            )
        pairs = [(query, c.text) for c in candidates]
        ce_logits = np.asarray(reranker.predict(pairs, show_progress_bar=False), dtype=float)
        order = np.argsort(ce_logits)[::-1]

        ce_ranked: list[Chunk] = []
        for i in order:
            ch = candidates[int(i)].model_copy(deep=True)
            ch.score = round(float(ce_logits[int(i)]), 4)
            ce_ranked.append(ch)

        fused = reciprocal_rank_fusion(
            [candidates, ce_ranked],
            rrf_k=40,
            top_k=len(candidates),
        )
        return self._diversify_by_doc(fused, top_k=top_k, max_per_doc=max_chunks_per_doc)

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
