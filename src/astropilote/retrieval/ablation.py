"""Run 4-way retrieval ablation study over `eval/golden.jsonl`.

Compares:
1. `bm25_only` (`bm25s` lexical retrieval)
2. `dense_only` (`BAAI/bge-small-en-v1.5` + Qdrant cosine similarity)
3. `hybrid_rrf` (BM25 + Dense fused via Reciprocal Rank Fusion, k=60)
4. `hybrid_rrf_rerank` (Hybrid RRF + Cross-Encoder `ms-marco-MiniLM-L-6-v2` calibrated reranking)

Usage:
    uv run python -m astropilote.retrieval.ablation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from astropilote.retrieval.eval_retrieval import evaluate_retrieval_mode
from astropilote.retrieval.hybrid import HybridRetriever

__all__ = ["evaluate_retrieval_mode", "run_ablation"]


def run_ablation() -> dict[str, dict]:
    """Run all 4 retrieval configurations and save metrics to `eval/retrieval_ablation.json`."""
    golden_path = Path("eval/golden.jsonl")
    questions = [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    configs: list[tuple[str, Literal["bm25", "dense", "hybrid"], bool]] = [
        ("1_bm25_only", "bm25", False),
        ("2_dense_only", "dense", False),
        ("3_hybrid_rrf", "hybrid", False),
        ("4_hybrid_rrf_rerank", "hybrid", True),
    ]

    results: dict[str, dict] = {}
    with HybridRetriever() as retriever:
        # Warm up encoder & reranker once before timing
        retriever.search("Messier 31 Andromeda Galaxy", top_k=5, mode="hybrid", use_reranker=True)

        for name, mode, rerank in configs:
            raw_metrics = evaluate_retrieval_mode(
                retriever=retriever,
                questions=questions,
                mode=mode,
                use_reranker=rerank,
                top_k=5,
            )
            metrics = {
                "num_queries": raw_metrics["num_queries"],
                "recall_at_5": raw_metrics["recall_at_5"],
                "hit_rate_at_5": raw_metrics["hit_rate_at_5"],
                "mrr_at_5": raw_metrics["mrr_at_5"],
                "avg_latency_ms": raw_metrics["avg_latency_ms"],
            }
            results[name] = metrics
            print(
                f"[{name}] Recall@5={metrics['recall_at_5']:.4f} | "
                f"HitRate@5={metrics['hit_rate_at_5']:.4f} | "
                f"MRR@5={metrics['mrr_at_5']:.4f} | "
                f"avg_latency={metrics['avg_latency_ms']:.1f}ms"
            )

    out_path = Path("eval/retrieval_ablation.json")
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    run_ablation()
