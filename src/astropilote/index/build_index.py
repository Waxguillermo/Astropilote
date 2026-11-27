"""Build local BM25 (`bm25s`) and lightweight ONNX Dense Vector (`fastembed` + `float16` NumPy) indexes.

Usage:
    uv run python -m astropilote.index.build_index
    uv run python -m astropilote.index.build_index --bm25-only
    uv run python -m astropilote.index.build_index --embedding-model BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

try:
    import bm25s
except ImportError:  # pragma: no cover
    bm25s = None  # type: ignore[assignment]

from astropilote.agent.schemas import Chunk
from astropilote.config import settings
from astropilote.device import get_compute_device


def load_chunks(chunks_path: Path) -> list[Chunk]:
    """Load validated `Chunk` objects from `data/processed/chunks.jsonl`."""
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {chunks_path}. Run `uv run python -m astropilote.ingest.run_ingest` first."
        )
    chunks: list[Chunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    return chunks


def build_bm25_index(chunks: list[Chunk], bm25_dir: Path) -> None:
    """Build and persist a `bm25s` lexical index over `chunks` (with pure-Python fallback if bm25s is absent)."""
    t0 = time.perf_counter()
    if bm25_dir.exists():
        shutil.rmtree(bm25_dir)
    bm25_dir.mkdir(parents=True, exist_ok=True)

    if bm25s is not None:
        corpus_texts = [c.text for c in chunks]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(str(bm25_dir))

    meta_path = bm25_dir / "chunks.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(ch.model_dump_json() + "\n")

    elapsed = time.perf_counter() - t0
    print(f"[BM25] Indexed {len(chunks)} chunks into {bm25_dir}/ in {elapsed:.2f}s.")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 embeddings (safe for zero-norm rows)."""
    mat = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return mat / norms


def build_dense_index(
    chunks: list[Chunk],
    dense_dir: Path,
    embedding_model_name: str,
    batch_size: int = 64,
) -> None:
    """Encode chunks with ONNX `fastembed` and store L2-normalized `float16` vectors + aligned metadata.

    The output directory contains read-only files (`vectors.npy`, `chunks.jsonl`, `meta.json`) with
    no file locks, so multiple concurrent Uvicorn workers can load and query it safely.
    """
    from fastembed import TextEmbedding

    t0 = time.perf_counter()
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)

    device = get_compute_device()
    print(f"[Dense] Loading ONNX embedding model `{embedding_model_name}` (runtime=`{device}`)...")
    encoder = TextEmbedding(model_name=embedding_model_name)
    texts = [c.text for c in chunks]

    print(f"[Dense] Encoding {len(texts)} chunks (batch_size={batch_size})...")
    raw_vecs = list(encoder.embed(texts, batch_size=batch_size))
    embeddings = (
        _l2_normalize(np.vstack(raw_vecs)) if raw_vecs else np.empty((0, 0), dtype=np.float32)
    )
    dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0

    vectors_f16 = embeddings.astype(np.float16)
    np.save(dense_dir / "vectors.npy", vectors_f16)

    chunks_path = dense_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(ch.model_dump_json() + "\n")

    meta = {
        "model": embedding_model_name,
        "dim": dim,
        "count": len(chunks),
        "dtype": "float16",
        "backend": "fastembed-onnx",
    }
    (dense_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - t0
    size_mb = (dense_dir / "vectors.npy").stat().st_size / (1024 * 1024)
    print(
        f"[Dense] Indexed {len(chunks)} vectors (dim={dim}, {size_mb:.2f} MB float16) "
        f"into {dense_dir}/ in {elapsed:.2f}s."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for index construction."""
    parser = argparse.ArgumentParser(
        description="Build BM25 and lightweight ONNX dense indexes over processed chunks."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=settings.data_dir / "processed" / "chunks.jsonl",
        help="Path to input chunks.jsonl",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=settings.embedding_model,
        help=f"FastEmbed ONNX model ID (default: {settings.embedding_model})",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Build only the BM25 index (skip downloading/encoding dense embeddings).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    chunks = load_chunks(args.chunks)
    build_bm25_index(chunks, settings.bm25_path)
    if not args.bm25_only:
        build_dense_index(
            chunks=chunks,
            dense_dir=settings.dense_path,
            embedding_model_name=args.embedding_model,
        )


if __name__ == "__main__":
    main()
