"""Local trusted operator CLI for starting and resuming MCP-backed workflows.

Do not expose ``resume --approve`` on an unauthenticated HTTP endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from astropilote.agentic.workflow import ObservatoryWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="AstroPilote LangGraph + MCP + HITL demo")
    parser.add_argument("--db", default="data/backend/observatory.sqlite3")
    parser.add_argument("--checkpoints", default="data/backend/agentic_checkpoints.sqlite3")
    commands = parser.add_subparsers(dest="operation", required=True)
    start = commands.add_parser("start")
    start.add_argument("--thread", required=True)
    start.add_argument("--action", choices=["book", "cancel"], required=True)
    start.add_argument("--user", required=True)
    start.add_argument("--slot")
    start.add_argument("--booking")
    start.add_argument("--place", help="Optional sky geocoding via the second MCP server")
    start.add_argument("--max-steps", type=int, default=5)
    resume = commands.add_parser("resume")
    resume.add_argument("--thread", required=True)
    choice = resume.add_mutually_exclusive_group(required=True)
    choice.add_argument("--approve", action="store_true")
    choice.add_argument("--reject", action="store_true")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--thread", required=True)
    args = parser.parse_args()
    db_path = Path(args.db)
    if not db_path.is_file():
        parser.error("Backend not found. Run: uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json")
    with ObservatoryWorkflow(db_path=db_path, checkpoint_path=args.checkpoints) as wf:
        if args.operation == "start":
            output = wf.start(thread_id=args.thread, action=args.action, user_id=args.user,
                              slot_id=args.slot, booking_id=args.booking,
                              place_name=args.place, max_steps=args.max_steps)
        elif args.operation == "resume":
            output = wf.resume(thread_id=args.thread, approved=args.approve)
        else:
            output = wf.inspect(thread_id=args.thread)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
