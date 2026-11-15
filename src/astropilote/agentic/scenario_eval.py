"""Isolated, state-graded Agentic Workflows scenario harness for AstroPilote.

Run through the existing ``eval/run_eval.py --scenarios`` entrypoint after installing
Phase 5. Each scenario gets *fresh* databases and real MCP stdio calls; only
trusted benchmark metadata can approve an interrupted cancellation.

No LLM is needed for this deterministic, conservative conversational router;
measured token use and LLM cost for these scenarios are therefore exactly zero.
"""

from __future__ import annotations

import copy
import json
import shutil
import statistics
import tempfile
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

CATEGORIES = frozenset({"normal", "ambiguous", "impossible", "adversarial"})
READ_TOOLS = frozenset({"get_observatory_state", "list_telescopes", "list_slots", "get_booking"})
WRITE_TOOLS = frozenset({"create_booking", "cancel_booking"})
INJECTION_TOOLS = frozenset({"list_telescopes", "list_slots", "get_booking"})


def load_scenarios(file_path: str | Path) -> list[dict[str, Any]]:
    """Validate a bounded JSONL dataset before any backend execution begins."""
    path = Path(file_path)
    scenarios = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_num, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_num}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_num}: scenario must be an object")
            scenario_id = item.get("id")
            if not isinstance(scenario_id, str) or not scenario_id or scenario_id in ids:
                raise ValueError(f"Line {line_num}: missing or duplicate scenario id")
            ids.add(scenario_id)
            if item.get("category") not in CATEGORIES:
                raise ValueError(f"Line {line_num}: unsupported category")
            turns = item.get("turns")
            if not isinstance(turns, list) or not 1 <= len(turns) <= 5:
                raise ValueError(f"Line {line_num}: turns must contain 1..5 items")
            for turn in turns:
                if (not isinstance(turn, dict) or not isinstance(turn.get("text"), str)
                        or not turn["text"].strip() or len(turn["text"]) > 4000):
                    raise ValueError(f"Line {line_num}: invalid turn text")
                if not isinstance(turn.get("user_id"), str) or not turn["user_id"]:
                    raise ValueError(f"Line {line_num}: trusted user_id required")
                if "reference_date" in turn:
                    date.fromisoformat(turn["reference_date"])
                if turn.get("operator_decision") not in (None, "approve", "reject"):
                    raise ValueError(f"Line {line_num}: invalid operator_decision")
            limit = item.get("max_steps", 5)
            if type(limit) is not int or not 1 <= limit <= 20:
                raise ValueError(f"Line {line_num}: max_steps must be 1..20")
            for field in ("forbid_calls", "expect_tools"):
                val = item.get(field, [])
                if not isinstance(val, list) or not all(isinstance(t, str) for t in val):
                    raise ValueError(f"Line {line_num}: {field} must be a list of names")
            for field in ("expect_responses", "expect_state", "expect_memory"):
                val = item.get(field, {} if field != "expect_responses" else [])
                if not isinstance(val, (dict if field != "expect_responses" else list)):
                    raise ValueError(f"Line {line_num}: malformed {field}")
            injections = item.get("inject_tool_response", [])
            if not isinstance(injections, list):
                raise ValueError(f"Line {line_num}: invalid injection list")
            for injection in injections:
                if (not isinstance(injection, dict)
                        or injection.get("tool") not in INJECTION_TOOLS
                        or not isinstance(injection.get("text"), str)
                        or not 0 < len(injection["text"]) <= 1000):
                    raise ValueError(f"Line {line_num}: invalid read-only injection")
            scenarios.append(item)
    if not scenarios:
        raise ValueError("The scenario file is empty")
    return scenarios


class TracedGateway:
    """Real MCP call wrapper. Inject only *untrusted read-result text* for tests.

    Never intercepts a write to invent its success, never issues HITL tokens,
    and never puts a secret approval_token in traces.
    """

    def __init__(self, delegate: Any, injections: list[dict[str, str]] | None = None):
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []
        self.injections = injections or []
        self.injections_applied: list[str] = []

    def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        trace = {
            "server": server,
            "tool": tool,
            "args": {key: value for key, value in args.items() if key != "approval_token"},
            "status": "ok",
        }
        self.calls.append(trace)
        try:
            payload = self.delegate.call(server, tool, args)
        except Exception:
            trace["status"] = "error"
            raise
        for injected in self.injections:
            if injected["tool"] != tool or tool not in INJECTION_TOOLS:
                continue
            response = copy.deepcopy(payload)
            if tool == "list_telescopes" and isinstance(response, list) and response:
                response[0]["name"] = str(response[0]["name"]) + " | " + injected["text"]
            elif tool == "list_slots" and isinstance(response, list) and response:
                response[0]["untrusted_note"] = injected["text"]
            elif tool == "get_booking" and isinstance(response, dict):
                response["untrusted_note"] = injected["text"]
            else:
                continue
            payload = response
            self.injections_applied.append(tool)
        return payload


def _subset_matches(actual: Any, expected: Any, path: str, problems: list[str]) -> None:
    """Assert structural subset, matching DB rows by stable id, not list order."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            problems.append(f"{path}: expected object")
            return
        for key, expected_value in expected.items():
            if key not in actual:
                problems.append(f"{path}.{key}: missing")
            else:
                _subset_matches(actual[key], expected_value, f"{path}.{key}", problems)
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            problems.append(f"{path}: expected list")
            return
        for i, value in enumerate(expected):
            if isinstance(value, dict) and "id" in value:
                matching = next((row for row in actual if isinstance(row, dict)
                                 and row.get("id") == value["id"]), None)
                if matching is None:
                    problems.append(f"{path}: missing row id={value['id']}")
                else:
                    _subset_matches(matching, value, f"{path}[{value['id']}]", problems)
            elif i >= len(actual):
                problems.append(f"{path}[{i}]: missing")
            else:
                _subset_matches(actual[i], value, f"{path}[{i}]", problems)
    elif type(actual) is not type(expected) or actual != expected:
        problems.append(f"{path}: expected {expected!r}, got {actual!r}")


def _expected_response_matches(actual: dict[str, Any], expected: dict[str, Any], issues: list[str], index: int) -> None:
    expected_fields = {key: value for key, value in expected.items() if key != "data_ids"}
    _subset_matches(actual, expected_fields, f"response[{index}]", issues)
    if "data_ids" in expected:
        rows = actual.get("data", [])
        ids = [row.get("id") for row in rows] if isinstance(rows, list) else None
        if ids != expected["data_ids"]:
            issues.append(f"response[{index}].data_ids: expected {expected['data_ids']}, got {ids}")


def run_one(scenario: dict[str, Any], *, seed_path: Path) -> dict[str, Any]:
    """Run the actual user-facing controller with isolated SQLite databases."""
    from astropilote.agentic.memory import UserMemory
    from astropilote.agentic.memory_conversation import MemoryConversationalAgent
    from astropilote.agentic.workflow import ObservatoryWorkflow, StdioMCPGateway
    from astropilote.backend.store import ObservatoryStore

    started = time.perf_counter()
    scenario_id = scenario["id"]
    errors: list[str] = []
    responses: list[dict[str, Any]] = []
    before_approval_ok = True

    with tempfile.TemporaryDirectory(prefix="astropilote-eval-") as tmp:
        directory = Path(tmp)
        db_path = directory / "observatory.sqlite3"
        checkpoint_path = directory / "checkpoints.sqlite3"
        memory_path = directory / "memory.sqlite3"
        backend = ObservatoryStore(db_path)
        original_state = backend.reset_backend(seed_path)
        memory = UserMemory(memory_path)
        gateway = TracedGateway(StdioMCPGateway(db_path), scenario.get("inject_tool_response"))

        def make_workflow() -> ObservatoryWorkflow:
            return ObservatoryWorkflow(db_path=db_path, checkpoint_path=checkpoint_path, gateway=gateway)

        # This is an action-only evaluator. Any fall-through to the historical
        # LLM astronomy assistant is an unexpected route, never a live API call.
        def no_astronomy(_: str) -> dict[str, Any]:
            return {"status": "unexpected_astronomy_route", "message": "Outside the action benchmark"}

        controller = MemoryConversationalAgent(
            db_path=db_path,
            checkpoint_path=checkpoint_path,
            memory=memory,
            gateway=gateway,
            workflow_factory=make_workflow,
            astronomy_runner=no_astronomy,
        )
        try:
            for position, turn in enumerate(scenario["turns"]):
                ref_date = date.fromisoformat(turn["reference_date"]) if turn.get("reference_date") else date(2026, 11, 14)
                thread_id = f"eval-{scenario_id}-{position}"
                old_state = backend.get_state()
                old_calls = len(gateway.calls)
                response = controller.handle(
                    turn["text"], user_id=turn["user_id"], thread_id=thread_id,
                    reference_date=ref_date, max_steps=scenario.get("max_steps", 5),
                )
                decision = turn.get("operator_decision")
                if decision is not None:
                    if response.get("status") != "awaiting_human_approval":
                        errors.append(f"turn {position}: operator decision without interrupt")
                    else:
                        # Fail closed: state must be unchanged before approval.
                        state_safe = backend.get_state() == old_state
                        tools_before = [call["tool"] for call in gateway.calls[old_calls:]]
                        if not state_safe or "cancel_booking" in tools_before:
                            before_approval_ok = False
                            errors.append(f"turn {position}: write before human decision")
                        with make_workflow() as workflow:
                            resumed = workflow.resume(
                                thread_id=thread_id, approved=(decision == "approve")
                            )
                        response = {**response, **resumed, "route": response.get("route"),
                                    "intent": response.get("intent"), "operator_decision": decision}
                if response.get("status") == "unexpected_astronomy_route":
                    errors.append(f"turn {position}: unexpected LLM fallback")
                if response.get("status") == "awaiting_human_approval":
                    if backend.get_state() != old_state:
                        before_approval_ok = False
                        errors.append(f"turn {position}: change while awaiting approval")
                responses.append(response)
        except Exception as exc:
            # Do not leak API credentials or raw exception text into results.
            errors.append(f"execution error: {type(exc).__name__}")

        final_state = backend.get_state()
        if scenario.get("expect_unchanged", False) and final_state != original_state:
            errors.append("backend changed although expect_unchanged=true")
        _subset_matches(final_state, scenario.get("expect_state", {}), "backend_state", errors)
        if "expect_booking_count" in scenario:
            expected_count = scenario["expect_booking_count"]
            if len(final_state["bookings"]) != expected_count:
                errors.append(f"expected {expected_count} bookings, got {len(final_state['bookings'])}")
        if len(responses) != len(scenario.get("expect_responses", [])):
            errors.append("number of responses differs from expect_responses")
        else:
            for i, expected in enumerate(scenario["expect_responses"]):
                _expected_response_matches(responses[i], expected, errors, i)
        for user, preferred in scenario.get("expect_memory", {}).items():
            actual_preferred = memory.preferred_telescope(user)
            if actual_preferred != preferred:
                errors.append(f"memory[{user}]: expected {preferred!r}, got {actual_preferred!r}")
        tools = [call["tool"] for call in gateway.calls]
        unexpected = sorted(set(tools).intersection(scenario.get("forbid_calls", [])))
        if unexpected:
            errors.append(f"forbidden tools invoked: {unexpected}")
        if "expect_tools" in scenario and tools != scenario["expect_tools"]:
            errors.append(f"expected tools {scenario['expect_tools']}, got {tools}")
        if len(gateway.injections_applied) != len(scenario.get("inject_tool_response", [])):
            errors.append("tool-result injection fixture was not exercised")
        steps = sum(int(result.get("steps", 0)) for result in responses)
        for i, result in enumerate(responses):
            if int(result.get("steps", 0)) > scenario.get("max_steps", 5):
                errors.append(f"turn {i}: max_steps exceeded")
        duration = round(time.perf_counter() - started, 4)
        # Only summary fields used in results, never secrets or injected raw text.
        last_status = responses[-1].get("status") if responses else "error"
        return {
            "id": scenario_id,
            "category": scenario["category"],
            "pass": not errors,
            "reason": "all_assertions_passed" if not errors else "; ".join(errors),
            "expect_unchanged": bool(scenario.get("expect_unchanged", False)),
            "preapproval_unchanged": before_approval_ok,
            "final_state": final_state,
            "tool_calls": gateway.calls,
            "steps": steps,
            "max_steps": scenario.get("max_steps", 5),
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_s": duration,
            "cost_usd": 0.0,
            "stop_reason": last_status,
            "response_statuses": [r.get("status") for r in responses],
            "injections_applied": gateway.injections_applied,
            "memory_effect": [{
                "memory_applied": r.get("memory_applied", False),
                "recommended_slot_id": r.get("recommended_slot_id"),
                "preferred_telescope_id": r.get("preferred_telescope_id"),
            } for r in responses],
        }


def evaluate_scenarios(scenarios_path: Path, out_arg: Path) -> dict[str, Any]:
    """Run, grade and persist each scenario; return a compact summary."""
    scenarios = load_scenarios(scenarios_path)
    seed = Path(__file__).resolve().parents[3] / "seeds" / "observatory_demo.json"
    if not seed.is_file():
        raise FileNotFoundError(f"Missing seed: {seed}")
    out_dir = out_arg.parent if out_arg.suffix == ".jsonl" else out_arg
    results_path = out_arg if out_arg.suffix == ".jsonl" else out_dir / "results.jsonl"
    summary_path = out_dir / "summary_agentic.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The original Generative AI runner also uses results.jsonl. Preserve its
    # pre-existing golden-question output before writing the mandated filename.
    if results_path.is_file():
        try:
            first_row = next((json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()), {})
        except (ValueError, OSError):
            first_row = {}
        if "category" not in first_row:
            backup = out_dir / "results_generative_ai_backup.jsonl"
            if not backup.exists():
                shutil.copy2(results_path, backup)

    outcomes: list[dict[str, Any]] = []
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out_dir,
                                     prefix=".agentic-results-", suffix=".jsonl", delete=False) as tmp_file:
        temporary_results = Path(tmp_file.name)
        for scenario in scenarios:
            try:
                record = run_one(scenario, seed_path=seed)
            except Exception as exc:
                record = {"id": scenario["id"], "category": scenario["category"], "pass": False,
                          "reason": f"runner_error:{type(exc).__name__}", "stop_reason": "error",
                          "final_state": None, "tool_calls": [], "steps": 0,
                          "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0, "cost_usd": 0.0}
            outcomes.append(record)
            tmp_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"[Agentic Eval] {scenario['id']} {scenario['category']}: "
                  f"{'PASS' if record['pass'] else 'FAIL'} - {record['reason']}")
    temporary_results.replace(results_path)

    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted(CATEGORIES):
        records = [r for r in outcomes if r["category"] == category]
        if records:
            passed = sum(bool(r["pass"]) for r in records)
            by_category[category] = {"total": len(records), "passed": passed,
                                     "pass_rate": round(passed / len(records), 4)}
    durations = [item["latency_s"] for item in outcomes]
    summary = {
        "benchmark": "AstroPilote Agentic Workflows state-based scenarios",
        "scenarios_path": str(scenarios_path),
        "total": len(outcomes),
        "passed": sum(bool(item["pass"]) for item in outcomes),
        "failed": sum(not item["pass"] for item in outcomes),
        "pass_rate": round(sum(bool(item["pass"]) for item in outcomes) / len(outcomes), 4),
        "categories": by_category,
        "stop_reasons": dict(Counter(item.get("stop_reason", "error") for item in outcomes)),
        "latency_median_s": round(statistics.median(durations), 4),
        "latency_p95_s": round(sorted(durations)[min(len(durations) - 1, int(len(durations) * 0.95))], 4),
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "total_cost_usd": 0.0,
        "measurement_note": "Deterministic routes use no LLM; tokens and LLM cost are zero, not estimates.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Agentic Eval] {summary['passed']}/{summary['total']} passed -> {results_path}, {summary_path}")
    return summary
