"""Mandatory CLI evaluation runner for AstroPilote.

Usage (required by Albert School × Mines Paris project instructions):
    uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from astropilote.agent.graph import run_agent
from astropilote.config import settings
from astropilote.device import get_torch_device

REFERENCE_DATE = "2026-10-15"
DEFAULT_REFERENCE_DATE = REFERENCE_DATE

_GOLD_STARTS_NEG_RE = re.compile(
    r"^\s*(?:no\b|not\s+really\b|this\s+premise\s+is\s+wrong\b|non\b)",
    re.IGNORECASE,
)
_GOLD_NEG_MARKERS_RE = re.compile(
    r"(?:"
    r"never\s+rises|"
    r"stays\s+below\s+the\s+horizon|"
    r"cannot\s+be\s+observed|"
    r"cannot\s+observe|"
    r"cannot\s+see|"
    r"cannot\s+photograph|"
    r"not\s+observable|"
    r"unobservable|"
    r"far\s+below\s+the\s+20\s+degree\s+minimum|"
    r"does\s+not\s+fit|"
    r"will\s+not\s+fit|"
    r"does\s+not\s+record|"
    r"not\s+available\s+in\s+the\s+corpus|"
    r"little\s+to\s+no\s+true\s+astronomical\s+darkness|"
    r"decline\s+to\s+give\s+cloud\s+cover"
    r")",
    re.IGNORECASE,
)
_GOLD_HORIZON_IMPOSSIBLE_RE = re.compile(
    r"(?:never\s+rises|stays\s+below\s+the\s+horizon|cannot\s+be\s+observed|unobservable)",
    re.IGNORECASE,
)
_GOLD_FRAMING_IMPOSSIBLE_RE = re.compile(
    r"(?:does\s+not\s+fit|will\s+not\s+fit|mosaic\s+or\s+a\s+much\s+shorter\s+focal\s+length|smaller\s+than\s+\S+\s+roughly\s+\d+)",
    re.IGNORECASE,
)
_GOLD_STARTS_POS_RE = re.compile(r"^\s*(?:yes|oui)\b", re.IGNORECASE)

_PRED_NEG_TOKENS_RE = re.compile(
    r"\b(?:"
    r"no|not|never|cannot|can't|won't|doesn't|isn't|aren't|don't|"
    r"unobservable|impossible|unavailable|unknown|unrecorded|"
    r"decline|wrong|false|incorrect|mosaic|"
    r"non|pas|jamais|aucun|aucune|inobservable|indisponible|faux|erron[eé]|mosa[iï]que"
    r")\b|"
    r"(?:"
    r"below\s+the\s+horizon|under\s+the\s+horizon|too\s+low|too\s+large|too\s+big|smaller\s+than|"
    r"sous\s+l['’]horizon|trop\s+bas|trop\s+grand"
    r")",
    re.IGNORECASE,
)
_PRED_HORIZON_NEG_RE = re.compile(
    r"(?:"
    r"never\s+rises|below\s+the\s+horizon|under\s+the\s+horizon|"
    r"cannot\s+(?:be\s+)?(?:observ|see|photograph|imag)|can't\s+(?:be\s+)?(?:observ|see|photograph|imag)|"
    r"not\s+(?:visible|observable|accessible)|isn't\s+(?:visible|observable|accessible)|"
    r"unobservable|impossible|-\d+(?:\.\d+)?\s*°|"
    r"ne\s+se\s+l[eè]ve\s+jamais|sous\s+l['’]horizon|inobservable|"
    r"\bno\b\s*[:,\-—]|\bnon\b\s*[:,\-—]"
    r")",
    re.IGNORECASE,
)
_PRED_FRAMING_NEG_RE = re.compile(
    r"(?:"
    r"does\s+not\s+fit|doesn't\s+fit|will\s+not\s+fit|won't\s+fit|cannot\s+fit|can't\s+fit|"
    r"not\s+fit|too\s+large|too\s+big|smaller\s+than|mosaic|shorter\s+focal\s+length|"
    r"covers\s+only|only\s+covers|ne\s+rentre\s+pas|trop\s+grand|mosa[iï]que|"
    r"\bno\b\s*[:,\-—]|\bnon\b\s*[:,\-—]"
    r")",
    re.IGNORECASE,
)
_PRED_STARTS_POS_RE = re.compile(
    r"^\s*\**(?:yes|oui|absolutely|certainly|of\s+course)\b",
    re.IGNORECASE,
)
_PRED_CORRECTIVE_CLAUSE_RE = re.compile(
    r"(?:"
    r"never\s+rises|below\s+the\s+horizon|cannot\s+(?:be\s+)?(?:observ|see|photograph)|"
    r"unobservable|does\s+not\s+fit|doesn't\s+fit|will\s+not\s+fit|won't\s+fit|"
    r"smaller\s+than|mosaic|premise\s+is\s+wrong|not\s+in\s+orion|far\s+below\s+the\s+20|"
    r"sous\s+l['’]horizon|inobservable|ne\s+rentre\s+pas|mosa[iï]que"
    r")",
    re.IGNORECASE,
)
_PRED_STARTS_NEG_RE = re.compile(
    r"^\s*\**(?:no\b|non\b|not\s+really\b|unfortunately,?\s+(?:you\s+)?cannot\b)",
    re.IGNORECASE,
)
_PRED_POS_AFFIRM_RE = re.compile(
    r"(?:"
    r"\byes\b|\boui\b|observable!|\bis\s+observable\b|\bare\s+observable\b|"
    r"\bwell[\s-]+positioned\b|\bwell[\s-]+placed\b|\bideally\s+placed\b|"
    r"\bexceeds\s+\d+|\babove\s+[2-8]\d\s*(?:°|degrees)\b|\bcan\s+(?:be\s+)?(?:imaged|photographed|observed)\b"
    r")",
    re.IGNORECASE,
)
_PRED_STRONG_IMPOSSIBLE_RE = re.compile(
    r"(?:"
    r"never\s+rises|stays\s+below\s+the\s+horizon|cannot\s+be\s+observed|"
    r"cannot\s+observe|cannot\s+photograph|unobservable|inobservable|sous\s+l['’]horizon"
    r")",
    re.IGNORECASE,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        help="Output directory or JSONL file path where results will be written (default: eval/)",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["api", "local", "offline"],
        default=None,
        help="LLM backend mode: 'api' (Anthropic/OpenRouter), 'local' (Ollama), or 'offline'. Defaults to 'api' if ANTHROPIC_API_KEY is set.",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default=REFERENCE_DATE,
        help=f"ISO reference date YYYY-MM-DD anchored for evaluation (default: {REFERENCE_DATE})",
    )
    parser.add_argument(
        "--strict-corpus-guard",
        action="store_true",
        dest="strict_corpus_guard",
        default=True,
        help="Enable strict corpus scope guard during evaluation (default: True).",
    )
    parser.add_argument(
        "--no-strict-corpus-guard",
        action="store_false",
        dest="strict_corpus_guard",
        help="Disable strict corpus scope guard during evaluation.",
    )
    return parser.parse_args(argv)


def _has_polarity_contradiction(pred_answer: str, gold_answer: str) -> bool:
    """Return True when `pred_answer` reverses the observability/framing/factual polarity of `gold_answer`."""
    if not gold_answer.strip() or not pred_answer.strip():
        return False

    gold_starts_neg = bool(_GOLD_STARTS_NEG_RE.search(gold_answer))
    gold_has_neg = gold_starts_neg or bool(_GOLD_NEG_MARKERS_RE.search(gold_answer))

    if gold_has_neg:
        if not _PRED_NEG_TOKENS_RE.search(pred_answer):
            return True
        if (
            gold_starts_neg
            and _PRED_STARTS_POS_RE.search(pred_answer)
            and not _PRED_CORRECTIVE_CLAUSE_RE.search(pred_answer)
        ):
            return True
        if _GOLD_HORIZON_IMPOSSIBLE_RE.search(gold_answer) and not _PRED_HORIZON_NEG_RE.search(
            pred_answer
        ):
            return True
        if _GOLD_FRAMING_IMPOSSIBLE_RE.search(gold_answer) and not _PRED_FRAMING_NEG_RE.search(
            pred_answer
        ):
            return True

    if _GOLD_STARTS_POS_RE.search(gold_answer):
        has_pos_affirm = bool(_PRED_POS_AFFIRM_RE.search(pred_answer))
        if _PRED_STARTS_NEG_RE.search(pred_answer) and not has_pos_affirm:
            return True
        if _PRED_STRONG_IMPOSSIBLE_RE.search(pred_answer) and not has_pos_affirm:
            return True

    return False


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
        if (
            f"[{gd.upper()}]" in pred_upper
            or gd.upper() in pred_upper
            or short_id.upper() in pred_upper
        ):
            inline_hits += 1
    inline_cov = inline_hits / len(gold_docs)

    # Check key terms from reference answer
    stop = {"the", "and", "for", "that", "with", "from", "are", "is", "in", "of", "to", "a", "an"}
    gold_words = {
        w.strip(".,()[];:").lower() for w in gold_answer.split() if len(w.strip(".,()[];:")) >= 3
    } - stop
    pred_low = pred_answer.lower()
    matched_words = sum(1 for w in gold_words if w in pred_low)
    lexical_cov = matched_words / max(1, len(gold_words))

    polarity_contradiction = _has_polarity_contradiction(pred_answer, gold_answer)
    if polarity_contradiction:
        lexical_cov = min(lexical_cov, 0.15)

    score = 0.40 * cite_overlap + 0.25 * inline_cov + 0.35 * min(1.0, lexical_cov * 1.45)
    if polarity_contradiction:
        score = min(score, 0.15)

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


def _resolve_output_paths(out_arg: Path, chosen_mode: str) -> tuple[Path, Path, Path]:
    """Resolve `(out_dir, results_path, summary_path)` while isolating offline/local runs from API outputs."""
    if out_arg.suffix == ".jsonl":
        out_dir = out_arg.parent
        if chosen_mode == "offline":
            if out_arg.stem in ("results", "results_local", "results_offline"):
                return (
                    out_dir,
                    out_dir / "results_offline.jsonl",
                    out_dir / "summary_metrics_offline.json",
                )
            return out_dir, out_arg, out_dir / f"summary_{out_arg.stem}.json"

        if out_arg.stem not in ("results", "results_local"):
            return out_dir, out_arg, out_dir / f"summary_{out_arg.stem}.json"
        if chosen_mode == "local" or "local" in out_arg.stem.lower():
            results_path = out_dir / "results_local.jsonl" if chosen_mode == "local" else out_arg
            return out_dir, results_path, out_dir / "summary_metrics_local.json"
        return out_dir, out_arg, out_dir / "summary_metrics.json"

    out_dir = out_arg
    if chosen_mode == "offline":
        return (
            out_dir,
            out_dir / "results_offline.jsonl",
            out_dir / "summary_metrics_offline.json",
        )
    if chosen_mode == "local":
        return (
            out_dir,
            out_dir / "results_local.jsonl",
            out_dir / "summary_metrics_local.json",
        )
    return out_dir, out_dir / "results.jsonl", out_dir / "summary_metrics.json"


def main() -> None:
    """Execute the evaluation pipeline and write per-question JSONL and summary JSON metrics."""
    args = parse_args()
    from astropilote.llm import _refresh_env_settings

    _refresh_env_settings()
    settings.reference_date = args.reference_date
    settings.strict_corpus_guard = args.strict_corpus_guard

    chosen_mode = args.llm_mode
    if chosen_mode is None:
        chosen_mode = "api" if settings.anthropic_api_key.strip() else "offline"

    if chosen_mode == "offline":
        settings.eval_offline_mode = True
        llm_mode_arg = None
    else:
        settings.eval_offline_mode = False
        llm_mode_arg = chosen_mode  # type: ignore[assignment]

    out_dir, results_path, summary_path = _resolve_output_paths(args.out, chosen_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.questions.open("r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(
        f"[AstroPilote Eval] Running {len(questions)} questions from {args.questions} "
        f"(mode={chosen_mode}, reference_date={args.reference_date}, strict_corpus_guard={args.strict_corpus_guard})..."
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

            _, _, row = run_agent(
                question_text,
                question_id=qid,
                llm_mode=llm_mode_arg,
                strict_corpus_guard=args.strict_corpus_guard,
            )
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
        "prudence_abstention_refusal_accuracy": round(sum(prudences) / max(1, len(prudences)), 4),
        "judged_correctness": round(sum(correctness_scores) / max(1, len(correctness_scores)), 4),
        "latency_median_s": round(statistics.median(latencies), 4) if latencies else 0.0,
        "latency_p95_s": round(sorted_lat[p95_idx], 4) if sorted_lat else 0.0,
        "avg_tokens_in": int(round(sum(tokens_in_list) / max(1, len(tokens_in_list)))),
        "avg_tokens_out": int(round(sum(tokens_out_list) / max(1, len(tokens_out_list)))),
        "total_cost_usd": round(sum(costs_usd), 6),
    }

    # Custom JSONL stems or offline mode write isolated summaries without touching API/Local comparison
    if chosen_mode == "offline":
        if summary_path.name == "summary_metrics_offline.json":
            summary_with_meta = {
                "mode": "offline",
                "reference_date": args.reference_date,
                **summary,
            }
        else:
            summary_with_meta = {
                "questions": str(args.questions),
                "mode": "offline",
                "reference_date": args.reference_date,
                **summary,
            }
        summary_path.write_text(json.dumps(summary_with_meta, indent=2) + "\n", encoding="utf-8")
        from astropilote.retrieval.hybrid import _close_all_shared_qdrant_clients

        _close_all_shared_qdrant_clients()
        print(f"[AstroPilote Eval] Wrote {results_path} and {summary_path}")
        print(json.dumps(summary, indent=2))
        return

    if summary_path.name not in ("summary_metrics.json", "summary_metrics_local.json"):
        summary_with_meta = {"questions": str(args.questions), "mode": chosen_mode, **summary}
        summary_path.write_text(json.dumps(summary_with_meta, indent=2) + "\n", encoding="utf-8")
        from astropilote.retrieval.hybrid import _close_all_shared_qdrant_clients

        _close_all_shared_qdrant_clients()
        print(f"[AstroPilote Eval] Wrote {results_path} and {summary_path}")
        print(json.dumps(summary, indent=2))
        return

    local_device = get_torch_device()
    if summary_path.name == "summary_metrics_local.json":
        summary_with_meta = {
            "model": f"ollama/{settings.ollama_model} ({local_device})",
            "mode": "local",
            **summary,
        }
        summary_path.write_text(json.dumps(summary_with_meta, indent=2) + "\n", encoding="utf-8")
    else:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Keep llm_comparison.json in sync whenever both API and Local summaries are present
    api_sum_path = out_dir / "summary_metrics.json"
    loc_sum_path = out_dir / "summary_metrics_local.json"
    if api_sum_path.exists() and loc_sum_path.exists():
        try:
            api_s = json.loads(api_sum_path.read_text(encoding="utf-8"))
            loc_s = json.loads(loc_sum_path.read_text(encoding="utf-8"))
            raw_loc_model = str(loc_s.get("model") or "")
            loc_model_name = (
                f"ollama/{settings.ollama_model}"
                if not raw_loc_model or "smollm2" in raw_loc_model.lower()
                else raw_loc_model
            )
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
                        "model_name": loc_model_name,
                        "provider": f"Local Ollama ({local_device})",
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
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass

    from astropilote.retrieval.hybrid import _close_all_shared_qdrant_clients

    _close_all_shared_qdrant_clients()

    print(f"[AstroPilote Eval] Wrote per-question results to {results_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
