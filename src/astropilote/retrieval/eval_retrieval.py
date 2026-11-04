"""Evaluate retrieval quality (Recall@5, HitRate@5, MRR@5, Latency) on `eval/golden.jsonl`.

Usage:
    uv run python -m astropilote.retrieval.eval_retrieval --mode bm25 --no-rerank
    uv run python -m astropilote.retrieval.eval_retrieval --mode hybrid
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

from astropilote.retrieval.hybrid import HybridRetriever


def evaluate_retrieval_mode(
    retriever: HybridRetriever,
    questions: list[dict],
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid",
    use_reranker: bool = False,
    top_k: int = 5,
) -> dict[str, float | int]:
    """Compute Recall@K, HitRate@K, MRR@K, and average latency on answerable golden questions."""
    answerable = [q for q in questions if q.get("gold_docs")]
    recalls: list[float] = []
    hits: list[float] = []
    rrs: list[float] = []
    latencies_ms: list[float] = []

    for q in answerable:
        gold_docs = list(dict.fromkeys(q["gold_docs"]))
        gold_set = set(gold_docs)
        t0 = time.perf_counter()
        chunks = retriever.search(
            query=q["question"],
            top_k=top_k,
            mode=mode,
            use_reranker=use_reranker,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        retrieved_doc_ids: list[str] = list(dict.fromkeys(c.doc_id for c in chunks))[:top_k]
        matched = gold_set.intersection(retrieved_doc_ids)

        recalls.append(len(matched) / len(gold_set))
        hits.append(1.0 if matched else 0.0)

        rr = 0.0
        for rank_idx, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in gold_set:
                rr = 1.0 / rank_idx
                break
        rrs.append(rr)

    n = max(1, len(answerable))
    recall_v = round(sum(recalls) / n, 4)
    hit_v = round(sum(hits) / n, 4)
    mrr_v = round(sum(rrs) / n, 4)
    lat_v = round(sum(latencies_ms) / n, 2)
    return {
        "num_queries": len(answerable),
        f"recall@{top_k}": recall_v,
        f"hit_rate@{top_k}": hit_v,
        f"mrr@{top_k}": mrr_v,
        f"recall_at_{top_k}": recall_v,
        f"hit_rate_at_{top_k}": hit_v,
        f"mrr_at_{top_k}": mrr_v,
        "avg_latency_ms": lat_v,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Score retrieval on eval/golden.jsonl.")
    parser.add_argument("--questions", type=Path, default=Path("eval/golden.jsonl"))
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    with args.questions.open("r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    with HybridRetriever() as retriever:
        metrics = evaluate_retrieval_mode(
            retriever=retriever,
            questions=questions,
            mode=args.mode,
            use_reranker=not args.no_rerank,
            top_k=args.top_k,
        )
    print(json.dumps({"mode": args.mode, "rerank": not args.no_rerank, **metrics}, indent=2))


if __name__ == "__main__":
    main()
