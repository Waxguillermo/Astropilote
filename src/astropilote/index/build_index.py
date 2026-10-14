"""Build local BM25 (`bm25s`) and Dense Vector (`Qdrant` + `SentenceTransformers`) indexes.

Usage:
    uv run python -m astropilote.index.build_index
    uv run python -m astropilote.index.build_index --bm25-only
    uv run python -m astropilote.index.build_index --embedding-model BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

try:
    import bm25s
except ImportError:  # pragma: no cover
    bm25s = None  # type: ignore[assignment]

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:  # pragma: no cover
    QdrantClient = Distance = PointStruct = VectorParams = Any  # type: ignore[assignment,misc]

from astropilote.agent.schemas import Chunk
from astropilote.config import settings

QDRANT_COLLECTION = "astropilote_chunks"


def load_chunks(chunks_path: Path) -> list[Chunk]:
    """Load validated `Chunk` objects from `data/processed/chunks.jsonl`."""
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {chunks_path}. Run `python -m astropilote.ingest.run_ingest` first."
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

    # Persist the ordered chunk JSONL alongside the BM25 matrices for fast lookup
    meta_path = bm25_dir / "chunks.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(ch.model_dump_json() + "\n")

    elapsed = time.perf_counter() - t0
    print(f"[BM25] Indexed {len(chunks)} chunks into {bm25_dir}/ in {elapsed:.2f}s.")


def get_torch_device() -> str:
    """Detect best available PyTorch accelerator (Apple Silicon MPS, CUDA, or CPU)."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_dense_qdrant_index(
    chunks: list[Chunk],
    qdrant_dir: Path,
    embedding_model_name: str,
    batch_size: int = 64,
) -> None:
    """Encode chunks with `SentenceTransformer` and store vectors + metadata in local Qdrant."""
    from sentence_transformers import SentenceTransformer

    lock_file = qdrant_dir / ".lock"
    if lock_file.exists() and (qdrant_dir / "meta.json").exists():
        import portalocker

        try:
            with lock_file.open("r+") as lf:
                portalocker.lock(
                    lf, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING
                )
                portalocker.unlock(lf)
        except Exception:
            print(
                f"[Dense] Qdrant index at {qdrant_dir}/ is already built and currently open by the running server — keeping existing dense index."
            )
            return

    t0 = time.perf_counter()
    if qdrant_dir.exists():
        shutil.rmtree(qdrant_dir)
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    device = get_torch_device()
    print(f"[Dense] Loading embedding model `{embedding_model_name}` on device=`{device}`...")
    encoder = SentenceTransformer(embedding_model_name, device=device)
    texts = [c.text for c in chunks]

    print(f"[Dense] Encoding {len(texts)} chunks (batch_size={batch_size})...")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    dim = int(embeddings.shape[1])

    client = QdrantClient(path=str(qdrant_dir))
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points: list[PointStruct] = []
    for idx, (chunk, vec) in enumerate(zip(chunks, embeddings, strict=True)):
        payload = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "section": chunk.section,
            "text": chunk.text,
            "catalog": chunk.metadata.catalog,
            "object_type": chunk.metadata.object_type,
            "constellation": chunk.metadata.constellation,
            "ra_deg": chunk.metadata.ra_deg,
            "dec_deg": chunk.metadata.dec_deg,
            "magnitude": chunk.metadata.magnitude,
            "surface_brightness": chunk.metadata.surface_brightness,
            "size_major_arcmin": chunk.metadata.size_major_arcmin,
            "size_minor_arcmin": chunk.metadata.size_minor_arcmin,
            "narrowband_friendly": chunk.metadata.narrowband_friendly,
            "aliases": chunk.metadata.aliases,
        }
        points.append(PointStruct(id=idx, vector=vec.tolist(), payload=payload))

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    client.close()

    # Record which embedding model built this Qdrant store
    (qdrant_dir / "model_name.txt").write_text(embedding_model_name, encoding="utf-8")
    elapsed = time.perf_counter() - t0
    print(
        f"[Dense] Indexed {len(chunks)} vectors (dim={dim}) into Qdrant at {qdrant_dir}/ in {elapsed:.2f}s."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for index construction."""
    parser = argparse.ArgumentParser(
        description="Build BM25 and Qdrant dense indexes over processed chunks."
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
        help=f"HuggingFace embedding model ID (default: {settings.embedding_model})",
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
        build_dense_qdrant_index(
            chunks=chunks,
            qdrant_dir=settings.qdrant_path,
            embedding_model_name=args.embedding_model,
        )


if __name__ == "__main__":
    main()
