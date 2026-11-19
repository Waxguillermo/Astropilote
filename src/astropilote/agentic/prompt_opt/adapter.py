"""GEPA adapter: a candidate is the agent's task policy, a rollout is one graded scenario.

`evaluate` runs `scenario_eval.run_one` (isolated SQLite + real stdio MCP servers) with the
candidate composed behind the frozen security preamble. The prompt is passed explicitly to
each run, never through `settings`, so concurrent rollouts of different candidates are safe.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from gepa import EvaluationBatch
from openinference.instrumentation import using_metadata

from astropilote.agentic.prompt_opt.scoring import (
    ZERO,
    Reference,
    ScenarioScores,
    reflection_example,
    score_record,
)
from astropilote.agentic.prompts import (
    BASELINE_POLICY,
    PromptPolicyError,
    compose_system_prompt,
    prompt_sha256,
)
from astropilote.agentic.scenario_eval import run_one

logger = logging.getLogger(__name__)

POLICY = "policy"


def policy_brevity(policy: str) -> float:
    """Shorter policies are re-sent at every step: baseline length = 0.5, empty = 1."""
    return min(1.0, max(0.0, 1.0 - 0.5 * len(policy.strip()) / len(BASELINE_POLICY.strip())))


RunOne = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class Rollout:
    """One scenario run under one candidate (GEPA output and trajectory alike)."""

    scenario: dict[str, Any]
    record: dict[str, Any]
    scores: ScenarioScores
    reference: Reference | None = None


def _failed_record(scenario: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"id": scenario["id"], "pass": False, "reason": reason, "tool_calls": []}


class ScenarioAdapter:
    """`gepa.GEPAAdapter` over the AstroPilote scenario harness (single component `policy`)."""

    # GEPA proposes new texts with its default reflective proposer and `reflection_lm`.
    propose_new_texts = None

    def __init__(
        self,
        *,
        llm_mode: Literal["api", "local"] = "api",
        workers: int = 3,
        run: RunOne = run_one,
        references: Mapping[str, Reference] | None = None,
    ) -> None:
        """`references`: baseline-policy cost per scenario id (`scoring.load_references`);
        without one a scenario's token and step objectives stay neutral."""
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.llm_mode = llm_mode
        self.workers = workers
        self._run = run
        self.references = dict(references or {})
        self._lock = threading.Lock()
        self.rollouts = 0
        self.rollout_cost_usd = 0.0

    def _rollout(self, scenario: dict[str, Any], prompt: str, sha: str) -> Rollout:
        with using_metadata({"gepa_candidate_sha256": sha, "scenario_id": scenario["id"]}):
            try:
                record = self._run(
                    copy.deepcopy(scenario), llm_mode=self.llm_mode, system_prompt=prompt
                )
            except Exception as exc:
                # One crashed rollout must not abort a paid optimization run: score it 0.
                logger.exception("Rollout %s crashed under candidate %s", scenario["id"], sha)
                record = _failed_record(scenario, f"runner_error:{type(exc).__name__}")
        with self._lock:
            self.rollouts += 1
            self.rollout_cost_usd += float(record.get("cost_usd", 0.0))
        reference = self.references.get(scenario["id"])
        return Rollout(scenario, record, score_record(scenario, record, reference), reference)

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Rollout, Rollout]:
        try:
            prompt = compose_system_prompt(candidate[POLICY])
        except PromptPolicyError as exc:
            # Never executed: an invalid policy (too long, forged frame tags) scores zero.
            logger.warning("Rejected candidate policy: %s", exc)
            rollouts = [Rollout(s, _failed_record(s, f"invalid_policy:{exc}"), ZERO) for s in batch]
            brevity = 0.0
        else:
            brevity = policy_brevity(candidate[POLICY])
            sha = prompt_sha256(prompt)
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                rollouts = list(pool.map(lambda s: self._rollout(s, prompt, sha), batch))
        return EvaluationBatch(
            outputs=rollouts,
            scores=[r.scores.scalar for r in rollouts],
            trajectories=rollouts if capture_traces else None,
            objective_scores=[
                {**r.scores.as_objectives(), "policy_brevity": brevity} for r in rollouts
            ],
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Rollout, Rollout],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        trajectories = eval_batch.trajectories or []
        examples = [
            reflection_example(r.scenario, r.record, r.scores, r.reference) for r in trajectories
        ]
        return {component: examples for component in components_to_update}


__all__ = ["POLICY", "Rollout", "ScenarioAdapter", "policy_brevity"]
