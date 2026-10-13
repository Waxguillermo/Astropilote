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
import time
from pathlib import Path
from typing import Literal

from astropilote.retrieval.hybrid import HybridRetriever


def evaluate_retrieval_mode(
    retriever: HybridRetriever,
    questions: list[dict],
    mode: Literal["bm25", "dense", "hybrid"],
    use_reranker: bool,
    top_k: int = 5,
) -> dict:
    """Compute Recall@k, HitRate@k, MRR@k, and average latency for a retrieval configuration."""
    recalls: list[float] = []
    hits: list[float] = []
    mrrs: list[float] = []
    latencies_ms: list[float] = []

    for q in questions:
        expected = list(dict.fromkeys(q.get("gold_docs", [])))
        if not expected:
            continue

        t0 = time.perf_counter()
        chunks = retriever.search(
            query=q["question"],
            top_k=top_k,
            mode=mode,
            use_reranker=use_reranker,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        ret_ids = list(dict.fromkeys(c.doc_id for c in chunks))[:top_k]
        matched = sum(1 for eid in expected if eid in ret_ids)
        recalls.append(matched / len(expected))
        hits.append(1.0 if matched > 0 else 0.0)

        rr = 0.0
        for rank_idx, rid in enumerate(ret_ids, start=1):
            if rid in expected:
                rr = 1.0 / rank_idx
                break
        mrrs.append(rr)

    n = max(1, len(recalls))
    return {
        "num_queries": len(recalls),
        "recall_at_5": round(sum(recalls) / n, 4),
        "hit_rate_at_5": round(sum(hits) / n, 4),
        "mrr_at_5": round(sum(mrrs) / n, 4),
        "avg_latency_ms": round(sum(latencies_ms) / n, 2),
    }


def run_ablation() -> dict[str, dict]:
    """Run all 4 retrieval configurations and save metrics to `eval/retrieval_ablation.json`."""
    golden_path = Path("eval/golden.jsonl")
    questions = [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retriever = HybridRetriever()
    # Warm up encoder & reranker once before timing
    retriever.search("Messier 31 Andromeda Galaxy", top_k=5, mode="hybrid", use_reranker=True)

    configs: list[tuple[str, Literal["bm25", "dense", "hybrid"], bool]] = [
        ("1_bm25_only", "bm25", False),
        ("2_dense_only", "dense", False),
        ("3_hybrid_rrf", "hybrid", False),
        ("4_hybrid_rrf_rerank", "hybrid", True),
    ]

    results: dict[str, dict] = {}
    for name, mode, rerank in configs:
        metrics = evaluate_retrieval_mode(
            retriever=retriever,
            questions=questions,
            mode=mode,
            use_reranker=rerank,
            top_k=5,
        )
        results[name] = metrics
        print(
            f"[{name}] Recall@5={metrics['recall_at_5']:.4f} | "
            f"HitRate@5={metrics['hit_rate_at_5']:.4f} | "
            f"MRR@5={metrics['mrr_at_5']:.4f} | "
            f"avg_latency={metrics['avg_latency_ms']:.1f}ms"
        )

    retriever.close()
    out_path = Path("eval/retrieval_ablation.json")
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    run_ablation()
