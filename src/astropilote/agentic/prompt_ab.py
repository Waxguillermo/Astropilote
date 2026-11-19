"""Paired A/B comparison of two agent task policies on the same scenarios.

    # run both variants k times, then compare
    uv run python -m astropilote.agentic.prompt_ab --scenarios eval/scenarios_redteam.jsonl \
        --split test --repeats 3
    # compare result files that already exist
    uv run python -m astropilote.agentic.prompt_ab --from-results eval/ab/results_*.jsonl

Each variant runs on identical scenarios with the same model. A scenario counts as passed for
a variant when it passes in a majority of its repeats. The paired test is an exact McNemar test
on the discordant scenarios (b: only A passes, c: only B passes):

    p = min(1, 2 * sum_{i <= min(b, c)} C(b + c, i) / 2^(b + c))

With n = 30 scenarios, B needs at least 6 discordant wins and no loss to reach p < 0.05.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from astropilote.agentic.prompt_opt.scoring import weighted_tokens
from astropilote.agentic.prompts import (
    BASELINE,
    PROJECT_ROOT,
    PromptPolicyError,
    PromptVariant,
    resolve_variant,
)

_OVER_REFUSAL_STATUSES = frozenset({"needs_clarification", "no_action"})
_BENIGN_KINDS = frozenset({"normal"})


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts `b` and `c`."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def _load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row.setdefault("results_file", str(path))
                rows.append(row)
    return rows


def _over_refused(row: dict[str, Any]) -> bool:
    """A benign task the agent declined (clarification or no-op) instead of doing it."""
    statuses = row.get("response_statuses") or []
    return (
        row.get("category") in _BENIGN_KINDS
        and not row.get("pass")
        and any(status in _OVER_REFUSAL_STATUSES for status in statuses)
    )


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0.0


def _variant_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(r.get("latency_s", 0.0)) for r in rows]
    by_group: dict[str, dict[str, list[bool]]] = {
        "category": defaultdict(list),
        "family": defaultdict(list),
    }
    for row in rows:
        by_group["category"][str(row.get("category"))].append(bool(row.get("pass")))
        if row.get("family"):
            by_group["family"][str(row["family"])].append(bool(row.get("pass")))
    benign = [r for r in rows if r.get("category") in _BENIGN_KINDS]
    return {
        "rows": len(rows),
        "repeats": len({r.get("repeat", 0) for r in rows}),
        "pass_rate": round(sum(bool(r.get("pass")) for r in rows) / max(1, len(rows)), 4),
        "pass_rate_by_category": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_group["category"].items())
        },
        "pass_rate_by_family": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_group["family"].items())
        },
        "forbidden_attempt_rows": sum(bool(r.get("forbidden_attempts")) for r in rows),
        "blocked_writes": sum(len(r.get("blocked_writes", [])) for r in rows),
        "over_refusal_rate": round(sum(_over_refused(r) for r in benign) / max(1, len(benign)), 4),
        "total_cost_usd": round(sum(float(r.get("cost_usd", 0.0)) for r in rows), 6),
        "tokens_in": sum(int(r.get("tokens_in", 0)) for r in rows),
        "tokens_out": sum(int(r.get("tokens_out", 0)) for r in rows),
        "steps_mean": round(statistics.mean(float(r.get("steps", 0)) for r in rows), 3)
        if rows
        else 0.0,
        "weighted_tokens_mean": round(statistics.mean(weighted_tokens(r) for r in rows))
        if rows
        else 0,
        "latency_median_s": round(statistics.median(latencies), 4) if latencies else 0.0,
        "latency_p95_s": round(_p95(latencies), 4),
    }


def _majority_pass(rows: list[dict[str, Any]]) -> dict[str, bool]:
    votes: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        votes[str(row["id"])].append(bool(row.get("pass")))
    return {sid: sum(v) * 2 > len(v) for sid, v in votes.items()}


def compare(rows: list[dict[str, Any]], *, a: str, b: str) -> dict[str, Any]:
    """Paired comparison of variants `a` (reference) and `b` from graded result rows."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("prompt_variant", "baseline"))].append(row)
    missing = [name for name in (a, b) if name not in groups]
    if missing:
        raise ValueError(f"no result rows for variant(s) {missing}; found {sorted(groups)}")

    warnings: list[str] = []
    shas = {name: {r.get("prompt_sha256") for r in groups[name]} for name in (a, b)}
    if shas[a] & shas[b]:
        warnings.append("both variants share a prompt SHA-256: they ran the same prompt")
    for name in (a, b):
        if len(shas[name]) > 1:
            warnings.append(
                f"variant {name!r} mixes several prompts: {sorted(map(str, shas[name]))}"
            )

    pass_a, pass_b = _majority_pass(groups[a]), _majority_pass(groups[b])
    paired = sorted(set(pass_a) & set(pass_b))
    unpaired = sorted(set(pass_a) ^ set(pass_b))
    if unpaired:
        warnings.append(
            f"scenarios run under one variant only (excluded from the test): {unpaired}"
        )
    only_a = [sid for sid in paired if pass_a[sid] and not pass_b[sid]]
    only_b = [sid for sid in paired if pass_b[sid] and not pass_a[sid]]

    stats = {name: _variant_stats(groups[name]) for name in (a, b)}
    return {
        "generated_on": date.today().isoformat(),
        "reference": a,
        "candidate": b,
        "prompt_sha256": {name: sorted(map(str, shas[name])) for name in (a, b)},
        "variants": stats,
        "paired_scenarios": len(paired),
        "majority_pass": {a: sum(pass_a[s] for s in paired), b: sum(pass_b[s] for s in paired)},
        "regressions": only_a,
        "improvements": only_b,
        "mcnemar_p": round(mcnemar_exact(len(only_a), len(only_b)), 6),
        "deltas": {
            "pass_rate": round(stats[b]["pass_rate"] - stats[a]["pass_rate"], 4),
            "over_refusal_rate": round(
                stats[b]["over_refusal_rate"] - stats[a]["over_refusal_rate"], 4
            ),
            "forbidden_attempt_rows": stats[b]["forbidden_attempt_rows"]
            - stats[a]["forbidden_attempt_rows"],
            "total_cost_usd": round(stats[b]["total_cost_usd"] - stats[a]["total_cost_usd"], 6),
            "steps_mean": round(stats[b]["steps_mean"] - stats[a]["steps_mean"], 3),
            "weighted_tokens_mean": stats[b]["weighted_tokens_mean"]
            - stats[a]["weighted_tokens_mean"],
            "latency_median_s": round(
                stats[b]["latency_median_s"] - stats[a]["latency_median_s"], 4
            ),
        },
        "warnings": warnings,
    }


def format_report(report: dict[str, Any]) -> str:
    a, b = report["reference"], report["candidate"]
    sa, sb = report["variants"][a], report["variants"][b]
    lines = [
        f"{'metric':<26}{a:>14}{b:>14}",
        f"{'pass rate':<26}{sa['pass_rate']:>14.3f}{sb['pass_rate']:>14.3f}",
        f"{'over-refusal rate':<26}{sa['over_refusal_rate']:>14.3f}{sb['over_refusal_rate']:>14.3f}",
        f"{'forbidden-attempt rows':<26}{sa['forbidden_attempt_rows']:>14}{sb['forbidden_attempt_rows']:>14}",
        f"{'blocked writes':<26}{sa['blocked_writes']:>14}{sb['blocked_writes']:>14}",
        f"{'cost (USD)':<26}{sa['total_cost_usd']:>14.4f}{sb['total_cost_usd']:>14.4f}",
        f"{'steps / scenario':<26}{sa['steps_mean']:>14.2f}{sb['steps_mean']:>14.2f}",
        f"{'weighted tokens / scen.':<26}{sa['weighted_tokens_mean']:>14}{sb['weighted_tokens_mean']:>14}",
        f"{'latency median (s)':<26}{sa['latency_median_s']:>14.2f}{sb['latency_median_s']:>14.2f}",
        "",
        f"paired scenarios: {report['paired_scenarios']}  improvements: {report['improvements']}  "
        f"regressions: {report['regressions']}  McNemar exact p = {report['mcnemar_p']:.4f}",
    ]
    families = sorted(set(sa["pass_rate_by_family"]) | set(sb["pass_rate_by_family"]))
    for family in families:
        lines.append(
            f"  family {family:<20}{sa['pass_rate_by_family'].get(family, float('nan')):>10.3f}"
            f"{sb['pass_rate_by_family'].get(family, float('nan')):>10.3f}"
        )
    lines += [f"WARNING: {w}" for w in report["warnings"]]
    return "\n".join(lines)


def _run_variants(
    scenario_files: list[Path],
    variants: list[PromptVariant],
    *,
    split: str | None,
    repeats: int,
    out_dir: Path,
) -> list[Path]:
    from astropilote.agentic.scenario_eval import evaluate_scenarios
    from astropilote.config import settings
    from astropilote.llm import _refresh_env_settings, check_llm_status

    _refresh_env_settings()
    settings.llm_backend_fallback = False
    status = check_llm_status()
    if not status.get("anthropic_ready"):
        raise SystemExit("`ANTHROPIC_API_KEY` is not configured: the A/B runs cannot start.")
    model = str(status["anthropic_model"])
    # The MCP agent plans with ROUTER_MODEL: that is the model whose policy is compared.
    slug = str(status["router_model"]).rsplit("/", 1)[-1]
    backend = {
        "model": model,
        "router_model": str(status["router_model"]),
        "generated_on": date.today().isoformat(),
    }

    from astropilote.agentic.scenario_eval import load_scenarios

    written: list[Path] = []
    for scenarios in scenario_files:
        # `--split` selects within partitioned files (red team); an unpartitioned file such as
        # the 30-scenario benchmark is never seen by GEPA and runs whole, as a regression set.
        labelled = any("split" in item for item in load_scenarios(scenarios))
        file_split = split if labelled else None
        for variant in variants:
            for k in range(repeats):
                results = out_dir / f"results_{slug}_{scenarios.stem}_{variant.name}_r{k}.jsonl"
                evaluate_scenarios(
                    scenarios,
                    results,
                    llm_mode="api",
                    backend=backend,
                    variant=variant,
                    split=file_split,
                    repeat_index=k,
                )
                written.append(results)
    return written


def _display_path(path: Path) -> str:
    """Repository-relative path for shared A/B reports (no local home directory)."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paired A/B comparison of agent task policies.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenarios", type=Path, nargs="+", help="Scenario JSONL file(s) to run.")
    source.add_argument("--from-results", type=Path, nargs="+", help="Existing result JSONL files.")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--candidate-prompt",
        type=Path,
        default=None,
        help="Policy file for variant B (default: the optimized policy).",
    )
    parser.add_argument("--reference", default="baseline", help="Variant A name in result rows.")
    parser.add_argument("--candidate", default=None, help="Variant B name (default: inferred).")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval" / "ab")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    from astropilote.tracing import setup_tracing

    setup_tracing()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.scenarios:
        try:
            # The reference is always the shipped baseline, whatever AGENT_POLICY_PATH says.
            reference = BASELINE
            candidate = resolve_variant(
                optimized=args.candidate_prompt is None, path=args.candidate_prompt
            )
        except PromptPolicyError as exc:
            parser.error(str(exc))
        if args.split is None and any("redteam" in p.name for p in args.scenarios):
            print("[A/B] note: no --split given, train/val scenarios seen by GEPA are included.")
        files = _run_variants(
            args.scenarios,
            [reference, candidate],
            split=args.split,
            repeats=args.repeats,
            out_dir=args.out_dir,
        )
        a, b = reference.name, candidate.name
    else:
        files = args.from_results
        a, b = args.reference, args.candidate or "optimized"

    report = compare(_load_rows(files), a=a, b=b)
    report["results_files"] = [_display_path(p) for p in files]
    report_path = args.out_dir / f"report_{a}_vs_{b}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(format_report(report))
    print(f"[A/B] Wrote {report_path}")


if __name__ == "__main__":
    main()
