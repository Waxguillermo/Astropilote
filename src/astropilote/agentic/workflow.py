"""Checkpointed LangGraph workflow using real stdio MCP servers and human approval.

Provides a deterministic, typed action workflow for telescope slot bookings and
cancellations alongside the LLM ReAct planner (`llm_agent.py`), while leaving
the read-only astronomy assistant graph untouched.

Security boundaries:
* only the host ``resume`` command can supply an approval decision;
* ``interrupt()`` is checkpointed BEFORE the dangerous write;
* the short-lived cancellation capability is created inside ``_write`` only,
  never serialized into graph state or returned to the model;
* the Observatory MCP server independently consumes and validates it.
This demo CLI assumes local, trusted OS access. Add authenticated operator
identity and role checks before exposing resume over a web endpoint.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import sqlite3
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from mcp import Client, StdioServerParameters

from astropilote.backend.approvals import CancellationApprovals
from astropilote.backend.store import ObservatoryStore

Action = Literal["book", "cancel"]
MAX_STEPS = 5


class ActionState(TypedDict, total=False):
    request: dict[str, str]
    steps: int
    max_steps: int
    tool_trace: list[dict[str, Any]]
    observed: dict[str, Any]
    output: dict[str, Any]
    approved: bool
    stop_reason: str
    error: dict[str, Any]


class ToolFailure(Exception):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__(str(payload.get("error", {}).get("code", "TOOL_ERROR")))


class StdioMCPGateway:
    """Persistent-session stdio MCP client running separate subprocesses for `sky` and `observatory`."""

    def __init__(self, db_path: str | Path, *, offline_sky: bool = True) -> None:
        self.db_path = str(Path(db_path).resolve())
        self.offline_sky = offline_sky
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._queues: dict[str, asyncio.Queue[Any]] = {}
        self._tasks: dict[str, Future[Any]] = {}
        self._closed = False
        atexit.register(self.close)

    def __enter__(self) -> StdioMCPGateway:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and self._thread is not None and self._thread.is_alive():
            return self._loop
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_run, name="astro-mcp-gateway", daemon=True)
        thread.start()
        ready.wait(timeout=5.0)
        self._loop = loop
        self._thread = thread
        return loop

    def _ensure_server(self, server: str) -> tuple[asyncio.AbstractEventLoop, asyncio.Queue[Any]]:
        module = {
            "sky": "astropilote.mcp.sky_server",
            "observatory": "astropilote.mcp.observatory_server",
        }.get(server)
        if module is None:
            raise ValueError("Unrecognized MCP server")
        loop = self._ensure_loop()
        existing_q = self._queues.get(server)
        existing_task = self._tasks.get(server)
        if existing_q is not None and existing_task is not None and not existing_task.done():
            return loop, existing_q

        env = {**os.environ, "ASTROPILOTE_BACKEND_DB": self.db_path}
        if self.offline_sky:
            env["EVAL_OFFLINE_MODE"] = "true"
        params = StdioServerParameters(command=sys.executable, args=["-m", module], env=env)
        ready_fut: Future[bool] = Future()
        queue_holder: dict[str, asyncio.Queue[Any]] = {}

        async def _worker() -> None:
            req_q: asyncio.Queue[Any] = asyncio.Queue()
            queue_holder["q"] = req_q
            try:
                async with Client(params, read_timeout_seconds=20) as client:
                    await client.list_tools()
                    if not ready_fut.done():
                        ready_fut.set_result(True)
                    while True:
                        item = await req_q.get()
                        if item is None:
                            break
                        tool_name, tool_args, call_fut = item
                        try:
                            res = await client.call_tool(tool_name, tool_args)
                            call_fut.set_result(res)
                        except BaseException as exc:
                            call_fut.set_exception(exc)
            except BaseException as exc:
                if not ready_fut.done():
                    ready_fut.set_exception(exc)

        worker_fut = asyncio.run_coroutine_threadsafe(_worker(), loop)
        ready_fut.result(timeout=20.0)
        req_q = queue_holder["q"]
        self._queues[server] = req_q
        self._tasks[server] = worker_fut
        return loop, req_q

    def call(self, server: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("StdioMCPGateway is closed")
            loop, req_q = self._ensure_server(server)
            call_fut: Future[Any] = Future()
            loop.call_soon_threadsafe(req_q.put_nowait, (tool, args, call_fut))
            result = call_fut.result(timeout=25.0)

        payload = result.structured_content
        if not isinstance(payload, dict):
            raise ToolFailure({"ok": False, "error": {"code": "BAD_MCP_RESULT"}})
        if result.is_error or payload.get("ok") is not True:
            raise ToolFailure(payload)
        return payload["data"]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            if loop is not None and loop.is_running():
                for srv, req_q in list(self._queues.items()):
                    loop.call_soon_threadsafe(req_q.put_nowait, None)
                    task_fut = self._tasks.get(srv)
                    if task_fut is not None:
                        try:
                            task_fut.result(timeout=3.0)
                        except (TimeoutError, OSError, RuntimeError) as _close_err:
                            _ = _close_err
                loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            if loop is not None and not loop.is_running():
                try:
                    loop.close()
                except OSError:
                    pass
            self._queues.clear()
            self._tasks.clear()
            self._loop = None
            self._thread = None


class ObservatoryWorkflow:
    """Reusable sync wrapper; new instances resume the same persisted thread."""

    def __init__(
        self,
        *,
        db_path: str | Path = "data/backend/observatory.sqlite3",
        checkpoint_path: str | Path = "data/backend/agentic_checkpoints.sqlite3",
        gateway: Any | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = ObservatoryStore(self.db_path)
        self.approvals = CancellationApprovals(self.db_path)
        self.gateway = gateway if gateway is not None else StdioMCPGateway(self.db_path)
        self._connection = sqlite3.connect(self.checkpoint_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._connection)
        self.graph = self._compile_graph()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ObservatoryWorkflow:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be nonempty")
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _tick(state: ActionState) -> tuple[dict[str, Any], bool]:
        next_step = int(state.get("steps", 0)) + 1
        if next_step > int(state.get("max_steps", MAX_STEPS)):
            return {"stop_reason": "max_steps", "error": {"code": "BUDGET_EXCEEDED"}}, False
        return {"steps": next_step}, True

    @staticmethod
    def _stop(state: ActionState) -> str:
        return "end" if state.get("stop_reason") else "next"

    @staticmethod
    def _record(
        state: ActionState, tool: str, status: str, args: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Minimal trace: never record approval_token (not even its hash)."""
        allowed = {k: v for k, v in args.items() if k != "approval_token"}
        return [*state.get("tool_trace", []), {"tool": tool, "status": status, "args": allowed}]

    def _validate(self, state: ActionState) -> dict[str, Any]:
        patch, ok = self._tick(state)
        if not ok:
            return patch
        req = state.get("request")
        if not isinstance(req, dict) or req.get("action") not in ("book", "cancel"):
            return {**patch, "stop_reason": "invalid_request", "error": {"code": "INVALID_ACTION"}}
        for field in ("user_id", "slot_id" if req["action"] == "book" else "booking_id"):
            if not isinstance(req.get(field), str) or not req[field].strip():
                return {
                    **patch,
                    "stop_reason": "invalid_request",
                    "error": {"code": "MISSING_FIELD", "field": field},
                }
        return patch

    def _read(self, state: ActionState) -> dict[str, Any]:
        patch, ok = self._tick(state)
        if not ok:
            return patch
        req = state["request"]
        name = "list_slots" if req["action"] == "book" else "get_booking"
        args: dict[str, Any] = (
            {"available_only": False} if name == "list_slots" else {"booking_id": req["booking_id"]}
        )
        trace = list(state.get("tool_trace", []))
        current_tool = name
        current_args = args
        try:
            response = self.gateway.call("observatory", name, args)
            trace = self._record({"tool_trace": trace}, name, "ok", args)
            if req["action"] == "book":
                target = next((slot for slot in response if slot["id"] == req["slot_id"]), None)
                if target is None or not target["available"]:
                    raise ToolFailure({"error": {"code": "SLOT_UNAVAILABLE"}})
                # Optional astronomy step: inspect the requested observation site via
                # the second MCP server BEFORE making a booking. No model-supplied
                # text is executed as instructions.
                if req.get("place_name"):
                    current_tool = "geocode"
                    current_args = {"place_name": req["place_name"]}
                    site = self.gateway.call("sky", "geocode", current_args)
                    trace = self._record({"tool_trace": trace}, "geocode", "ok", current_args)
                    if not site.get("found"):
                        raise ToolFailure({"error": {"code": "LOCATION_NOT_FOUND"}})
                    target = {**target, "requested_site": site}
            else:
                target = response
                if target["status"] != "confirmed":
                    raise ToolFailure({"error": {"code": "BOOKING_NOT_ACTIVE"}})
                if target["user_id"] != req["user_id"]:
                    raise ToolFailure({"error": {"code": "BOOKING_FORBIDDEN"}})
            patch["observed"] = target
            patch["tool_trace"] = trace
        except ToolFailure as exc:
            # Domain errors (e.g. unavailable slot) need not be attributed to a
            # failed transport call if the read itself succeeded.
            patch["tool_trace"] = trace
            patch["error"] = exc.payload.get("error", {"code": "TOOL_ERROR"})
            patch["stop_reason"] = "impossible"
        except Exception:
            patch["tool_trace"] = self._record(
                {"tool_trace": trace}, current_tool, "error", current_args
            )
            patch["error"] = {"code": "MCP_UNAVAILABLE"}
            patch["stop_reason"] = "tool_error"
        return patch

    def _human_approval(self, state: ActionState) -> dict[str, Any]:
        patch, ok = self._tick(state)
        if not ok:
            return patch
        req = state["request"]
        # No side effects before interrupt; node restarts from the top on resume.
        decision = interrupt(
            {
                "action": "cancel_booking",
                "booking_id": req["booking_id"],
                "user_id": req["user_id"],
                "paid": bool(state["observed"].get("paid")),
                "risk": "Irreversible cancellation and simulated refund",
                "question": "Do you approve this cancellation?",
            }
        )
        if not isinstance(decision, dict) or decision.get("approved") is not True:
            return {**patch, "approved": False, "stop_reason": "human_rejected"}
        return {**patch, "approved": True}

    def _write(self, state: ActionState) -> dict[str, Any]:
        patch, ok = self._tick(state)
        if not ok:
            return patch
        req = state["request"]
        if req["action"] == "cancel" and state.get("approved") is not True:
            return {
                **patch,
                "stop_reason": "approval_missing",
                "error": {"code": "APPROVAL_REQUIRED"},
            }
        if req["action"] == "cancel":
            tool = "cancel_booking"
            public_args = {"booking_id": req["booking_id"], "user_id": req["user_id"]}
            # Capability is generated only after the approved interrupt; do NOT
            # put token in ActionState, trace or model-visible tool outputs.
            token = self.approvals.issue(booking_id=req["booking_id"], user_id=req["user_id"])
            arguments = {**public_args, "approval_token": token}
        else:
            tool = "create_booking"
            public_args = {"slot_id": req["slot_id"], "user_id": req["user_id"]}
            arguments = public_args
        try:
            response = self.gateway.call("observatory", tool, arguments)
            return {
                **patch,
                "output": response,
                "tool_trace": self._record(state, tool, "ok", public_args),
                "stop_reason": "completed",
            }
        except ToolFailure as exc:
            return {
                **patch,
                "error": exc.payload.get("error", {"code": "TOOL_ERROR"}),
                "tool_trace": self._record(state, tool, "error", public_args),
                "stop_reason": "tool_error",
            }
        except Exception:
            return {
                **patch,
                "error": {"code": "MCP_UNAVAILABLE"},
                "tool_trace": self._record(state, tool, "error", public_args),
                "stop_reason": "tool_error",
            }

    def _compile_graph(self) -> Any:
        graph = StateGraph(ActionState)
        graph.add_node("validate", self._validate)
        graph.add_node("read", self._read)
        graph.add_node("approve", self._human_approval)
        graph.add_node("write", self._write)
        graph.add_edge(START, "validate")
        graph.add_conditional_edges("validate", self._stop, {"end": END, "next": "read"})
        graph.add_conditional_edges(
            "read", self._route_read, {"end": END, "approve": "approve", "write": "write"}
        )
        graph.add_conditional_edges("approve", self._stop, {"end": END, "next": "write"})
        graph.add_edge("write", END)
        return graph.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _route_read(state: ActionState) -> str:
        if state.get("stop_reason"):
            return "end"
        return "approve" if state["request"]["action"] == "cancel" else "write"

    def start(
        self,
        *,
        thread_id: str,
        action: Action,
        user_id: str,
        slot_id: str | None = None,
        booking_id: str | None = None,
        place_name: str | None = None,
        max_steps: int = MAX_STEPS,
    ) -> dict[str, Any]:
        if not 1 <= max_steps <= 20:
            raise ValueError("max_steps must be 1..20")
        config = self._config(thread_id)
        snapshot = self.graph.get_state(config)
        if snapshot.values or snapshot.next:
            raise ValueError("thread_id already exists; use a fresh thread ID")
        request = {"action": action, "user_id": user_id}
        if slot_id is not None:
            request["slot_id"] = slot_id
        if booking_id is not None:
            request["booking_id"] = booking_id
        if place_name is not None:
            request["place_name"] = place_name
        result = self.graph.invoke(
            {
                "request": request,
                "steps": 0,
                "max_steps": max_steps,
                "tool_trace": [],
                "approved": False,
            },
            config,
        )
        return self._summary(result, config)

    def resume(self, *, thread_id: str, approved: bool) -> dict[str, Any]:
        if type(approved) is not bool:
            raise ValueError("approved must be an explicit boolean from the trusted host")
        config = self._config(thread_id)
        snapshot = self.graph.get_state(config)
        if "approve" not in snapshot.next:
            raise ValueError("No human approval interrupt is pending on this thread")
        result = self.graph.invoke(Command(resume={"approved": bool(approved)}), config)
        return self._summary(result, config)

    def inspect(self, *, thread_id: str) -> dict[str, Any]:
        config = self._config(thread_id)
        snapshot = self.graph.get_state(config)
        if not snapshot.values and not snapshot.next:
            raise ValueError("Unknown thread")
        return self._summary(dict(snapshot.values), config)

    def _summary(self, result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.graph.get_state(config)
        values = dict(snapshot.values)
        pending = "approve" in snapshot.next
        return {
            "thread_id": config["configurable"]["thread_id"],
            "status": "awaiting_human_approval"
            if pending
            else values.get("stop_reason", "unknown"),
            "steps": values.get("steps", 0),
            "tool_trace": values.get("tool_trace", []),
            "output": values.get("output"),
            "error": values.get("error"),
            "pending_approval": pending,
            "pending_booking_id": (values.get("request") or {}).get("booking_id")
            if pending
            else None,
            "backend_state": self.backend.get_state(),
        }
