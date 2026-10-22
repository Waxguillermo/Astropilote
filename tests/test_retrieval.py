"""Unit tests for Reciprocal Rank Fusion (RRF), MetadataFilter, and BM25 retrieval."""

from pathlib import Path

from astropilote.agent.schemas import Chunk, ChunkMetadata
from astropilote.index.build_index import build_bm25_index
from astropilote.retrieval.hybrid import (
    HybridRetriever,
    MetadataFilter,
    reciprocal_rank_fusion,
)


def _make_chunk(cid: str, doc_id: str, text: str, const: str = "AND", nb: bool = False) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        title=f"Object {doc_id}",
        section="Overview",
        text=text,
        metadata=ChunkMetadata(
            doc_id=doc_id,
            catalog="Messier",
            object_type="galaxy" if not nb else "planetary_nebula",
            constellation=const,
            magnitude=5.0,
            size_major_arcmin=30.0,
            narrowband_friendly=nb,
        ),
    )


def test_reciprocal_rank_fusion() -> None:
    """Verify that RRF boosts chunks appearing near the top of both BM25 and Dense lists."""
    c1 = _make_chunk("M031#0", "M031", "Andromeda Galaxy spiral in Andromeda")
    c2 = _make_chunk("M027#0", "M027", "Dumbbell Nebula planetary nebula in Vulpecula", "VUL", True)
    c3 = _make_chunk("M013#0", "M013", "Hercules Globular Cluster in Hercules", "HER")

    # c2 is rank 2 in list A and rank 1 in list B -> highest combined RRF score
    fused = reciprocal_rank_fusion([[c1, c2], [c2, c3]], rrf_k=60, top_k=3)
    assert fused[0].chunk_id == "M027#0"
    assert len(fused) == 3


def test_bm25_index_and_metadata_filter(tmp_path: Path) -> None:
    """Verify building a temporary BM25 index and querying with metadata filters."""
    chunks = [
        _make_chunk(
            "M031#0",
            "M031",
            "[M031 | Andromeda Galaxy] Bright spiral galaxy in Andromeda visible to the naked eye.",
            const="AND",
            nb=False,
        ),
        _make_chunk(
            "M027#0",
            "M027",
            "[M027 | Dumbbell Nebula] Planetary nebula in Vulpecula ideal for narrowband OIII filter.",
            const="VUL",
            nb=True,
        ),
    ]
    bm25_dir = tmp_path / "bm25"
    build_bm25_index(chunks, bm25_dir)

    retriever = HybridRetriever(bm25_dir=bm25_dir, qdrant_dir=tmp_path / "qdrant")
    hits = retriever.search(
        "narrowband OIII planetary nebula", top_k=2, mode="bm25", use_reranker=False
    )
    assert len(hits) >= 1
    assert hits[0].doc_id == "M027"

    # Test narrowband_only filter
    nb_filter = MetadataFilter(narrowband_only=True)
    filtered = retriever.search(
        "galaxy or nebula", top_k=2, mode="bm25", use_reranker=False, filters=nb_filter
    )
    assert all(c.metadata.narrowband_friendly for c in filtered)
