"""Mandatory CLI evaluation runner for AstroPilote.

Usage (required by Albert School × Mines Paris project instructions):
    uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from astropilote.agent.graph import run_agent
from astropilote.config import settings


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run AstroPilote evaluation over a golden JSONL question set."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("eval/golden.jsonl"),
        help="Path to input golden.jsonl file (default: eval/golden.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval"),
        help="Output directory where results.jsonl will be written (default: eval/)",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["api", "local", "offline"],
        default=None,
        help="LLM backend mode: 'api' (Anthropic/OpenRouter), 'local' (Ollama/SmolLM2 MPS), or 'offline'. Defaults to 'api' if ANTHROPIC_API_KEY is set.",
    )
    return parser.parse_args()


def _score_judge_correctness(
    pred_answer: str,
    gold_answer: str,
    cited: list[str],
    gold_docs: list[str],
    stop_reason: str,
) -> float:
    """Compute correctness score [0.0, 1.0] combining source overlap, inline citation fidelity, and key-fact synthesis."""
    if not gold_docs:
        # Unanswerable / trap question: correct iff agent abstained or refused with zero sources
        return 1.0 if (stop_reason in ("abstained", "refused") and len(cited) == 0) else 0.0

    if stop_reason in ("abstained", "refused"):
        return 0.0

    gold_set = set(gold_docs)
    cite_overlap = len(gold_set.intersection(cited)) / len(gold_set)

    # Check inline citation presence in generated text (e.g. [M031] or M31)
    pred_upper = pred_answer.upper()
    inline_hits = 0
    for gd in gold_docs:
        short_id = gd.replace("M0", "M").replace("M00", "M").replace("C0", "C").replace("C00", "C")
        if f"[{gd.upper()}]" in pred_upper or gd.upper() in pred_upper or short_id.upper() in pred_upper:
            inline_hits += 1
    inline_cov = inline_hits / len(gold_docs)

    # Check key terms from reference answer
    stop = {"the", "and", "for", "that", "with", "from", "are", "is", "in", "of", "to", "a", "an"}
    gold_words = {
        w.strip(".,()[];:").lower()
        for w in gold_answer.split()
        if len(w.strip(".,()[];:")) >= 3
    } - stop
    pred_low = pred_answer.lower()
    matched_words = sum(1 for w in gold_words if w in pred_low)
    lexical_cov = matched_words / max(1, len(gold_words))

    score = 0.40 * cite_overlap + 0.25 * inline_cov + 0.35 * min(1.0, lexical_cov * 1.45)
    if "surfbr:" in pred_low or "narrowband: no" in pred_low:
        score = max(0.0, score - 0.15)

    # Penalize degenerate LLM repetition loops (e.g. repeated 6-grams 5+ times or character loops)
    words_seq = pred_low.split()
    if len(words_seq) >= 24:
        sixgrams = [tuple(words_seq[i : i + 6]) for i in range(len(words_seq) - 5)]
        max_repeat = max((sixgrams.count(sg) for sg in set(sixgrams)), default=1)
        if max_repeat >= 5 or "00000000" in pred_low:
            score *= 0.25

    return round(min(1.0, score), 4)


def main() -> None:
    """Execute the evaluation pipeline and write `eval/results.jsonl` and `eval/summary_metrics.json`."""
    args = parse_args()
    from astropilote.llm import _refresh_env_settings

    _refresh_env_settings()
    chosen_mode = args.llm_mode
    if chosen_mode is None:
        chosen_mode = "api" if settings.anthropic_api_key.strip() else "offline"

    if chosen_mode == "offline":
        settings.eval_offline_mode = True
        llm_mode_arg = None
    else:
        settings.eval_offline_mode = False
        llm_mode_arg = chosen_mode  # type: ignore[assignment]

    if args.out.suffix == ".jsonl":
        results_path = args.out
        out_dir = args.out.parent
    else:
        out_dir = args.out
        results_path = out_dir / "results.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.questions.open("r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(
        f"[AstroPilote Eval] Running {len(questions)} questions from {args.questions} (mode={chosen_mode})..."
    )

    recalls: list[float] = []
    mrrs: list[float] = []
    prudences: list[float] = []
    correctness_scores: list[float] = []
    latencies: list[float] = []
    tokens_in_list: list[int] = []
    tokens_out_list: list[int] = []
    costs_usd: list[float] = []

    with results_path.open("w", encoding="utf-8") as out_f:
        for q in questions:
            qid = q["id"]
            question_text = q["question"]
            gold_docs: list[str] = q.get("gold_docs", [])
            gold_answer: str = q.get("answer", "")

            _, _, row = run_agent(question_text, question_id=qid, llm_mode=llm_mode_arg)
            out_f.write(row.model_dump_json() + "\n")

            latencies.append(row.latency_s)
            tokens_in_list.append(row.tokens_in)
            tokens_out_list.append(row.tokens_out)
            costs_usd.append(row.cost_usd)

            if gold_docs:
                gold_set = set(gold_docs)
                matched = gold_set.intersection(row.retrieved_ids)
                recalls.append(len(matched) / len(gold_set))
                rr = 0.0
                for rank_idx, doc_id in enumerate(row.retrieved_ids, start=1):
                    if doc_id in gold_set:
                        rr = 1.0 / rank_idx
                        break
                mrrs.append(rr)
            else:
                prudences.append(
                    1.0
                    if (row.stop_reason in ("abstained", "refused") and not row.cited_sources)
                    else 0.0
                )

            c_score = _score_judge_correctness(
                pred_answer=row.answer,
                gold_answer=gold_answer,
                cited=row.cited_sources,
                gold_docs=gold_docs,
                stop_reason=row.stop_reason,
            )
            correctness_scores.append(c_score)

    sorted_lat = sorted(latencies)
    p95_idx = min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))
    summary = {
        "total_questions": len(questions),
        "answerable_questions": len(recalls),
        "unanswerable_or_trap_questions": len(prudences),
        "retrieval_recall": round(sum(recalls) / max(1, len(recalls)), 4),
        "retrieval_mrr": round(sum(mrrs) / max(1, len(mrrs)), 4),
        "prudence_abstention_refusal_accuracy": round(
            sum(prudences) / max(1, len(prudences)), 4
        ),
        "judged_correctness": round(
            sum(correctness_scores) / max(1, len(correctness_scores)), 4
        ),
        "latency_median_s": round(statistics.median(latencies), 4) if latencies else 0.0,
        "latency_p95_s": round(sorted_lat[p95_idx], 4) if sorted_lat else 0.0,
        "avg_tokens_in": int(round(sum(tokens_in_list) / max(1, len(tokens_in_list)))),
        "avg_tokens_out": int(round(sum(tokens_out_list) / max(1, len(tokens_out_list)))),
        "total_cost_usd": round(sum(costs_usd), 6),
    }

    # A custom results file (e.g. `--out eval/results_extension.jsonl`) gets its own summary
    # (`summary_results_extension.json`) so it never overwrites the main golden-set metrics.
    if results_path.stem not in ("results", "results_local"):
        summary_path = out_dir / f"summary_{results_path.stem}.json"
        summary_with_meta = {"questions": str(args.questions), "mode": chosen_mode, **summary}
        summary_path.write_text(json.dumps(summary_with_meta, indent=2) + "\n", encoding="utf-8")
        from astropilote.retrieval.hybrid import _close_all_shared_qdrant_clients

        _close_all_shared_qdrant_clients()
        print(f"[AstroPilote Eval] Wrote {results_path} and {summary_path}")
        print(json.dumps(summary, indent=2))
        return

    is_local_run = chosen_mode == "local" or "local" in results_path.stem.lower()
    if is_local_run:
        summary_with_meta = {
            "model": "HuggingFaceTB/SmolLM2-360M-Instruct (Apple Silicon MPS)",
            "mode": "local",
            **summary,
        }
        summary_path = out_dir / "summary_metrics_local.json"
        summary_path.write_text(json.dumps(summary_with_meta, indent=2) + "\n", encoding="utf-8")
    else:
        summary_path = out_dir / "summary_metrics.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Keep llm_comparison.json in sync whenever both API and Local summaries are present
    api_sum_path = out_dir / "summary_metrics.json"
    loc_sum_path = out_dir / "summary_metrics_local.json"
    if api_sum_path.exists() and loc_sum_path.exists():
        try:
            api_s = json.loads(api_sum_path.read_text(encoding="utf-8"))
            loc_s = json.loads(loc_sum_path.read_text(encoding="utf-8"))
            comp = {
                "benchmark": (
                    f"eval/golden.jsonl ({api_s['total_questions']} astronomy queries: "
                    f"{api_s['answerable_questions']} answerable, "
                    f"{api_s['unanswerable_or_trap_questions']} traps/unanswerable)"
                ),
                "models": {
                    "api": {
                        "model_name": settings.anthropic_model,
                        "provider": "Anthropic API (OpenRouter)",
                        "retrieval_recall": api_s["retrieval_recall"],
                        "retrieval_mrr": api_s["retrieval_mrr"],
                        "prudence_accuracy": api_s["prudence_abstention_refusal_accuracy"],
                        "judged_correctness": api_s["judged_correctness"],
                        "latency_median_s": api_s["latency_median_s"],
                        "latency_p95_s": api_s["latency_p95_s"],
                        "avg_tokens_in": api_s["avg_tokens_in"],
                        "avg_tokens_out": api_s["avg_tokens_out"],
                        "total_cost_usd": api_s["total_cost_usd"],
                        "cost_per_query_usd": round(
                            api_s["total_cost_usd"] / max(1, api_s["total_questions"]), 6
                        ),
                    },
                    "local": {
                        "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
                        "provider": "Local Open-Weight (Apple Silicon MPS FP16)",
                        "retrieval_recall": loc_s["retrieval_recall"],
                        "retrieval_mrr": loc_s["retrieval_mrr"],
                        "prudence_accuracy": loc_s["prudence_abstention_refusal_accuracy"],
                        "judged_correctness": loc_s["judged_correctness"],
                        "latency_median_s": loc_s["latency_median_s"],
                        "latency_p95_s": loc_s["latency_p95_s"],
                        "avg_tokens_in": loc_s["avg_tokens_in"],
                        "avg_tokens_out": loc_s["avg_tokens_out"],
                        "total_cost_usd": loc_s["total_cost_usd"],
                        "cost_per_query_usd": 0.0,
                    },
                },
            }
            (out_dir / "llm_comparison.json").write_text(
                json.dumps(comp, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

    from astropilote.retrieval.hybrid import _close_all_shared_qdrant_clients

    _close_all_shared_qdrant_clients()

    print(f"[AstroPilote Eval] Wrote per-question results to {results_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
