"""Mandatory CLI evaluation runner for AstroPilote.

Usage (required by ENPC project instructions):
    uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

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

    summary_path = out_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[AstroPilote Eval] Wrote per-question results to {results_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
