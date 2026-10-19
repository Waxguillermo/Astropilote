"""CLI entry point to collect, clean, and chunk the ~500-document AstroPilote corpus.

Usage:
    uv run python -m astropilote.ingest.run_ingest
    uv run python -m astropilote.ingest.run_ingest --limit 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from astropilote.agent.schemas import Chunk
from astropilote.config import settings
from astropilote.ingest.catalog import MAX_GUIDE_SECTIONS, CatalogTarget, build_curated_catalog
from astropilote.ingest.chunker import chunk_target_sections, estimate_tokens
from astropilote.ingest.cleaner import split_into_sections
from astropilote.ingest.wikipedia import (
    build_fallback_extract,
    fetch_wikipedia_corpus_async,
    is_synthetic_fallback,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Download OpenNGC + English Wikipedia corpus and build section chunks."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on total documents to ingest (for quick testing).",
    )
    parser.add_argument(
        "--herschel-limit",
        type=int,
        default=200,
        help="Number of bright Herschel/NGC targets to include (default: 200).",
    )
    parser.add_argument(
        "--no-astrophoto",
        action="store_true",
        help="Leave out the Astrophoto subset (famous nebulae/galaxies added in corpus v2).",
    )
    parser.add_argument(
        "--no-guides",
        action="store_true",
        help="Leave out the practical observing & astrophotography guides (corpus v2).",
    )
    parser.add_argument(
        "--offline-fallback",
        action="store_true",
        help="Skip Wikipedia HTTP calls and use synthesized OpenNGC extracts.",
    )
    return parser.parse_args()


def run_pipeline(
    data_dir: Path | None = None,
    limit: int | None = None,
    herschel_limit: int = 200,
    offline_fallback: bool = False,
    include_astrophoto: bool = True,
    include_guides: bool = True,
) -> tuple[int, int]:
    """Execute the full ingestion & chunking pipeline and write JSONL artifacts."""
    base_dir = data_dir or settings.data_dir
    raw_dir = base_dir / "raw"
    wiki_cache_dir = raw_dir / "wikipedia"
    proc_dir = base_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building curated catalog (OpenNGC + IAU Constellations + guides)...")
    targets: list[CatalogTarget] = build_curated_catalog(
        raw_dir,
        herschel_limit=herschel_limit,
        include_astrophoto=include_astrophoto,
        include_guides=include_guides,
    )
    if limit is not None:
        targets = targets[:limit]
    print(f"      Selected {len(targets)} canonical targets.")

    print("[2/4] Fetching English Wikipedia articles (with local cache in data/raw/wikipedia/)...")
    if offline_fallback:
        extracts_by_id: dict[str, tuple[str, str]] = {}
        for t in targets:
            cache_file = wiki_cache_dir / f"{t.doc_id}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                fallback = "" if t.catalog == "Guide" else build_fallback_extract(t)
                extracts_by_id[t.doc_id] = (
                    cached.get("title", t.title),
                    cached.get("extract") or fallback,
                )
            else:
                fallback = "" if t.catalog == "Guide" else build_fallback_extract(t)
                extracts_by_id[t.doc_id] = (t.title, fallback)
    else:
        extracts_by_id = asyncio.run(fetch_wikipedia_corpus_async(targets, wiki_cache_dir))

    print("[3/4] Cleaning articles and splitting into section-aware chunks...")
    docs_path = proc_dir / "documents.jsonl"
    chunks_path = proc_dir / "chunks.jsonl"
    docs_tmp = proc_dir / "documents.jsonl.tmp"
    chunks_tmp = proc_dir / "chunks.jsonl.tmp"
    all_chunks: list[Chunk] = []
    catalog_counts: Counter[str] = Counter()
    synthetic_ids: list[str] = []
    skipped_guides: list[str] = []

    with (
        docs_tmp.open("w", encoding="utf-8") as doc_f,
        chunks_tmp.open("w", encoding="utf-8") as chunk_f,
    ):
        for target in targets:
            resolved_title, raw_extract = extracts_by_id[target.doc_id]
            sections = split_into_sections(raw_extract)
            if target.catalog == "Guide":
                if not sections:
                    skipped_guides.append(target.doc_id)
                    continue
                sections = sections[:MAX_GUIDE_SECTIONS]
            elif not sections or is_synthetic_fallback(raw_extract):
                synthetic_ids.append(target.doc_id)
                if not sections:
                    sections = split_into_sections(build_fallback_extract(target))
            catalog_counts[target.catalog] += 1

            target_chunks = chunk_target_sections(
                target=target,
                sections=sections,
                max_chunk_tokens=settings.max_chunk_tokens,
            )
            all_chunks.extend(target_chunks)

            doc_record = {
                "doc_id": target.doc_id,
                "title": target.title,
                "wikipedia_title": resolved_title,
                "catalog": target.catalog,
                "metadata": target.to_chunk_metadata().model_dump(),
                "num_sections": len(sections),
                "num_chunks": len(target_chunks),
            }
            doc_f.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

            for ch in target_chunks:
                chunk_f.write(ch.model_dump_json() + "\n")

    docs_tmp.replace(docs_path)
    chunks_tmp.replace(chunks_path)

    avg_tokens = (
        round(sum(estimate_tokens(c.text) for c in all_chunks) / len(all_chunks), 1)
        if all_chunks
        else 0.0
    )
    total_docs = sum(catalog_counts.values())
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "total_documents": total_docs,
        "total_chunks": len(all_chunks),
        "avg_chunk_tokens": avg_tokens,
        "max_chunk_tokens": settings.max_chunk_tokens,
        "catalog_breakdown": dict(catalog_counts),
        # Documents whose text was synthesized from OpenNGC numbers (no usable Wikipedia article)
        "synthetic_fallback_count": len(synthetic_ids),
        "synthetic_fallback_doc_ids": synthetic_ids,
        # Guides whose Wikipedia article could not be fetched (never synthesized, just skipped)
        "skipped_guides": skipped_guides,
        "sources": {
            "wikipedia": "https://en.wikipedia.org/w/api.php (CC BY-SA 4.0)",
            "openngc": "https://github.com/mattiaverga/OpenNGC (CC-BY-SA-4.0)",
        },
    }
    manifest_path = proc_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[4/4] Done! Wrote {total_docs} documents and {len(all_chunks)} chunks "
        f"(avg ~{avg_tokens} tokens/chunk) to {proc_dir}/"
    )
    print(f"      Breakdown: {dict(catalog_counts)}")
    print(f"      Synthetic fallback texts: {len(synthetic_ids)} | Skipped guides: {skipped_guides}")
    return total_docs, len(all_chunks)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_pipeline(
        limit=args.limit,
        herschel_limit=args.herschel_limit,
        offline_fallback=args.offline_fallback,
        include_astrophoto=not args.no_astrophoto,
        include_guides=not args.no_guides,
    )


if __name__ == "__main__":
    main()
