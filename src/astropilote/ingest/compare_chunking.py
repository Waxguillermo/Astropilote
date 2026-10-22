"""Experimental ablation comparing 3 chunking strategies on the 506-document corpus.

Compares:
1. `fixed_no_header`: Naive fixed-size sliding window (~220 tokens) without prepended object header
2. `fixed_with_header`: Fixed-size sliding window (~220 tokens) with prepended object header
3. `section_with_header`: Section-aware semantic chunking (max 400 tokens) with prepended object header

Usage:
    uv run python -m astropilote.ingest.compare_chunking
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from astropilote.agent.schemas import Chunk
from astropilote.config import settings
from astropilote.index.build_index import build_bm25_index
from astropilote.ingest.catalog import build_curated_catalog
from astropilote.ingest.chunker import (
    chunk_target_fixed_size,
    chunk_target_sections,
    estimate_tokens,
)
from astropilote.ingest.cleaner import split_into_sections
from astropilote.ingest.wikipedia import build_fallback_extract
from astropilote.retrieval.eval_retrieval import evaluate_retrieval_mode
from astropilote.retrieval.hybrid import HybridRetriever


def _load_cached_sections(raw_dir: Path, targets) -> dict[str, list[tuple[str, str]]]:
    wiki_dir = raw_dir / "wikipedia"
    sections_by_id: dict[str, list[tuple[str, str]]] = {}
    for t in targets:
        cache_file = wiki_dir / f"{t.doc_id}.json"
        extract = ""
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            extract = data.get("extract", "")
        secs = split_into_sections(extract) if extract else []
        if not secs:
            secs = split_into_sections(build_fallback_extract(t))
        sections_by_id[t.doc_id] = secs
    return sections_by_id


def run_chunking_experiment() -> dict[str, dict]:
    """Build chunks under all 3 strategies and score them on `eval/golden.jsonl`."""
    raw_dir = settings.data_dir / "raw"
    targets = build_curated_catalog(raw_dir)
    sections_by_id = _load_cached_sections(raw_dir, targets)

    golden_path = Path("eval/golden.jsonl")
    with golden_path.open("r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    strategies = {
        "1_fixed_no_header": lambda t, secs: chunk_target_fixed_size(
            t, secs, max_chunk_tokens=220, overlap_words=30, include_header=False
        ),
        "2_fixed_with_header": lambda t, secs: chunk_target_fixed_size(
            t, secs, max_chunk_tokens=220, overlap_words=30, include_header=True
        ),
        "3_section_with_header": lambda t, secs: chunk_target_sections(
            t, secs, max_chunk_tokens=400, overlap_words=35, include_header=True
        ),
    }

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for name, fn in strategies.items():
            chunks: list[Chunk] = []
            for t in targets:
                chunks.extend(fn(t, sections_by_id[t.doc_id]))

            bm25_dir = tmp_root / name
            build_bm25_index(chunks, bm25_dir)
            retriever = HybridRetriever(bm25_dir=bm25_dir, qdrant_dir=tmp_root / "no_qdrant")
            metrics = evaluate_retrieval_mode(
                retriever=retriever,
                questions=questions,
                mode="bm25",
                use_reranker=False,
                top_k=5,
            )
            avg_tok = round(sum(estimate_tokens(c.text) for c in chunks) / max(1, len(chunks)), 1)
            results[name] = {
                "total_chunks": len(chunks),
                "avg_tokens_per_chunk": avg_tok,
                **metrics,
            }

    out_path = Path("eval/chunking_experiment.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_chunking_experiment()
