"""Idempotent, minimal patch adding --scenarios to the existing historical CLI.

Run from repository root: python scripts/apply_agentic_eval.py
Do not replace the whole legacy runner: its --questions path is preserved.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "eval" / "run_eval.py"
PARSER_ANCHOR = '    parser.add_argument(\n        "--questions",'
MAIN_ANCHOR = '    args = parse_args()\n    from astropilote.llm import _refresh_env_settings'
PARSER_REPLACEMENT = '''    parser.add_argument(
        "--scenarios",
        type=Path,
        default=None,
        help="Run isolated Agentic Workflows state-based JSONL scenarios instead of legacy questions.",
    )
'''
MAIN_REPLACEMENT = '''    args = parse_args()
    if args.scenarios is not None:
        from astropilote.agentic.scenario_eval import evaluate_scenarios

        summary = evaluate_scenarios(args.scenarios, args.out)
        if summary["failed"]:
            raise SystemExit(1)
        return

    from astropilote.llm import _refresh_env_settings'''


def main() -> None:
    contents = RUNNER.read_text(encoding="utf-8")
    if "--scenarios" in contents and "astropilote.agentic.scenario_eval" in contents:
        print("Already patched: no changes.")
        return
    if contents.count(PARSER_ANCHOR) != 1 or contents.count(MAIN_ANCHOR) != 1:
        raise RuntimeError("The historical CLI has changed. Refusing to apply an unsafe patch.")
    updated = contents.replace(PARSER_ANCHOR, PARSER_REPLACEMENT + PARSER_ANCHOR)
    updated = updated.replace(MAIN_ANCHOR, MAIN_REPLACEMENT)
    ast.parse(updated, filename=str(RUNNER))
    RUNNER.write_text(updated, encoding="utf-8")
    print("Updated eval/run_eval.py: added --scenarios; --questions path preserved.")


if __name__ == "__main__":
    main()
