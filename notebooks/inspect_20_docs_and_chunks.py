"""Inspect 20 random documents and 20 random chunks from `data/processed/`.

Implements the ENPC course guideline:
    "read your data: open 20 random documents and 20 random chunks before choosing a chunking strategy"

Usage:
    uv run python notebooks/inspect_20_docs_and_chunks.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def main(seed: int = 42) -> None:
    """Sample and display 20 random documents and 20 random chunks with token statistics."""
    rng = random.Random(seed)
    proc_dir = Path("data/processed")
    docs_path = proc_dir / "documents.jsonl"
    chunks_path = proc_dir / "chunks.jsonl"

    with docs_path.open("r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    sample_docs = rng.sample(docs, min(20, len(docs)))
    sample_chunks = rng.sample(chunks, min(20, len(chunks)))

    print(f"=== SAMPLED 20 RANDOM DOCUMENTS (out of {len(docs)}) ===")
    for i, d in enumerate(sample_docs, start=1):
        meta = d["metadata"]
        print(
            f"{i:02d}. [{d['doc_id']}] {d['title']} | Catalog: {d['catalog']} | "
            f"Type: {meta['object_type']} | Const: {meta['constellation']} | "
            f"Mag: {meta['magnitude']} | Sections: {d['num_sections']} | Chunks: {d['num_chunks']}"
        )

    print(f"\n=== SAMPLED 20 RANDOM CHUNKS (out of {len(chunks)}) ===")
    for i, c in enumerate(sample_chunks, start=1):
        words = len(c["text"].split())
        est_tokens = int(words * 1.33)
        preview = c["text"].replace("\n", " ")[:160]
        print(
            f"{i:02d}. [{c['chunk_id']}] (~{est_tokens} tokens) -> {preview}..."
        )


if __name__ == "__main__":
    main()
