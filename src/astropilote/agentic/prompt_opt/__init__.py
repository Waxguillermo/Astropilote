"""Reflective prompt optimization (GEPA) of the LLM agent's task policy.

`scoring` turns one graded scenario record into per-objective scores and textual feedback;
`adapter` plugs `scenario_eval.run_one` into GEPA; `optimize` is the CLI. GEPA (Agrawal et al.
2025, arXiv:2507.19457) is an optional dependency: `uv sync --extra opt`.
"""
