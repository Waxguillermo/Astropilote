"""Prove two separate real stdio MCP processes serve read-only requests.

Usage: uv run python -m astropilote.mcp.smoke_test
Use --db path/to.sqlite3 to inspect a specific database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


async def probe(module: str, tool: str, arguments: dict, env: dict[str, str]) -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        env=env,
    )
    async with Client(server_params, read_timeout_seconds=20) as client:
        tools = await client.list_tools()
        print(f"{module}: {len(tools.tools)} tools: {', '.join(t.name for t in tools.tools)}")
        result = await client.call_tool(tool, arguments)
        print(
            json.dumps(
                {
                    "tool": tool,
                    "is_error": result.is_error,
                    "structured": result.structured_content,
                },
                ensure_ascii=False,
            )
        )
        if result.is_error:
            raise RuntimeError(f"Smoke test failed: {module}.{tool}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/backend/observatory.sqlite3")
    args = parser.parse_args()
    db_path = str(Path(args.db).resolve())
    if not Path(db_path).is_file():
        raise SystemExit(
            "Initialize your backend first: uv run python -m astropilote.backend.store --seed seeds/observatory_demo.json"
        )
    env = {**os.environ, "ASTROPILOTE_BACKEND_DB": db_path, "EVAL_OFFLINE_MODE": "1"}
    await probe("astropilote.mcp.sky_server", "geocode", {"place_name": "Fontainebleau"}, env)
    await probe("astropilote.mcp.observatory_server", "list_slots", {"available_only": True}, env)


if __name__ == "__main__":
    asyncio.run(main())
