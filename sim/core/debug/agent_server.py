"""MCP server exposing the Debugger to an agentic LLM client.

Tools forward to `session.debugger`, which the `main_agent.py` loop
re-binds for each fresh Simulator instance. When no run is bound
(between simulations), forwarding tools return an error pointing the
caller at `restart_simulation`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from .agent_runner import AgentSession


_SERVER_INSTRUCTIONS = (
    "Interactive debugger for the cg-sim compute graph simulator.\n"
    "\n"
    "Typical session:\n"
    "  1. Call `list_breakpoints` to see available BREAK_* flags.\n"
    "  2. Call `toggle_breakpoint` to enable the ones you care about.\n"
    "  3. Call `start_simulation`. It blocks until the first breakpoint\n"
    "     fires (or the run finishes) and returns the new state.\n"
    "  4. Use `execute` to inspect/mutate state (e.g. `trace.node_map`,\n"
    "     `job.BREAK_AT_JOB_RETIRED = True`, `debug.record({...})`).\n"
    "  5. Call `continue_simulation`. It blocks until the next state\n"
    "     change and returns it. Repeat 4-5 until the response reports\n"
    "     `simulation_finished=True`.\n"
    "\n"
    "After a run finishes, call `restart_simulation` to spin up a fresh\n"
    "Simulator (optionally with a new input path) and repeat from step 1,\n"
    "or call `shutdown` to end the process.\n"
    "\n"
    "`current_state` is a cheap re-read of the current state and is only\n"
    "needed if `start_simulation`/`continue_simulation` returns with\n"
    "`timed_out=True` (long-running stage)."
)


_NO_BOUND_DEBUGGER = {
    "ok": False,
    "error": "No simulator is currently bound. Call `restart_simulation` first.",
}


def build_mcp_server(session: "AgentSession") -> FastMCP:
    """Construct a FastMCP server backed by `session`."""
    server = FastMCP(name="cg-sim-debugger", instructions=_SERVER_INSTRUCTIONS)

    @server.tool(
        description=(
            "Start the simulator and block until it reaches its first "
            "breakpoint or finishes. Returns the resulting state in the "
            "same response — no polling needed. On a long-running stage "
            "the call returns `timed_out=True` so the caller can re-check "
            "via `current_state` without releasing the worker again."
        )
    )
    def start_simulation() -> dict[str, Any]:
        if session.debugger is None:
            return dict(_NO_BOUND_DEBUGGER)
        return session.debugger.agent_start_simulation()

    @server.tool(
        description=(
            "Return the current debugger state: whether the simulator is "
            "parked at a breakpoint, the breakpoint name, available "
            "variables and types, tip text, and whether the simulation "
            "has finished."
        )
    )
    def current_state() -> dict[str, Any]:
        if session.debugger is None:
            return dict(_NO_BOUND_DEBUGGER)
        return session.debugger.agent_current_state()

    @server.tool(
        description="Return a mapping of all BREAK_* flags to their On/Off status."
    )
    def list_breakpoints() -> dict[str, bool] | dict[str, Any]:
        if session.debugger is None:
            return dict(_NO_BOUND_DEBUGGER)
        return session.debugger.agent_list_breakpoints()

    @server.tool(
        description=(
            "Toggle a BREAK_* flag on the Debugger. Argument is the flag "
            "name (e.g. 'BREAK_AFTER_COMPILE_STAGE'). Returns the new value."
        )
    )
    def toggle_breakpoint(name: str) -> dict[str, Any]:
        if session.debugger is None:
            return dict(_NO_BOUND_DEBUGGER)
        return session.debugger.agent_toggle_breakpoint(name)

    @server.tool(
        description=(
            "Execute Python `code` against the parked breakpoint's "
            "namespace. The namespace persists across calls within a "
            "single breakpoint and exposes the variables advertised by "
            "`current_state`, plus `debug` (the Debugger). A bare last "
            "expression has its value echoed in `output`, like the REPL. "
            "Errors are returned as tracebacks rather than raised. Only "
            "callable when `at_breakpoint=True`."
        )
    )
    def execute(code: str) -> dict[str, Any]:
        if session.debugger is None:
            return dict(_NO_BOUND_DEBUGGER)
        return session.debugger.agent_execute(code)

    @server.tool(
        description=(
            "Release the worker and block until the next breakpoint or "
            "the simulation finishes. Returns the resulting state in the "
            "same response (including `simulation_finished=True` when "
            "the run is complete). On timeout, returns `timed_out=True`. "
            "Only callable when `at_breakpoint=True`."
        )
    )
    def continue_simulation() -> dict[str, Any]:
        if session.debugger is None:
            return dict(_NO_BOUND_DEBUGGER)
        return session.debugger.agent_continue()

    @server.tool(
        description=(
            "Tear down the just-finished simulator and build a fresh one. "
            "Only callable after `simulation_finished=True` (or before "
            "the first run). Optionally accepts `input_path` to switch "
            "the YAML config for the next run. Blocks until the new "
            "Simulator is constructed and ready to accept breakpoint "
            "toggles; the response carries the same state shape as "
            "`current_state` (at_breakpoint=False, simulation_finished=False)."
        )
    )
    def restart_simulation(input_path: str | None = None) -> dict[str, Any]:
        dbg = session.debugger
        if dbg is not None and not dbg._simulation_finished:
            return {
                "ok": False,
                "error": (
                    "Current simulation is not finished. Drive it to "
                    "completion via `continue_simulation` first, or call "
                    "`shutdown` to abort."
                ),
            }
        if input_path:
            session.next_input_path = input_path
        session.restart_complete.clear()
        session.request("restart")
        reached = session.restart_complete.wait(timeout=120.0)
        if not reached:
            return {
                "ok": False,
                "error": "Simulator construction timed out.",
                "timed_out": True,
            }
        if session.debugger is None:
            return {
                "ok": False,
                "error": "Simulator construction failed; see process stderr.",
            }
        state = session.debugger.agent_current_state()
        return {"ok": True, "input_path": session.next_input_path, **state}

    @server.tool(
        description=(
            "End the agent session. The main_agent.py process exits after "
            "this returns. Call after `simulation_finished=True` for a "
            "clean shutdown."
        )
    )
    def shutdown() -> dict[str, Any]:
        # Release any parked simulator so the run completes naturally
        # before the main loop tears it down.
        dbg = session.debugger
        if dbg is not None and dbg._at_breakpoint:
            dbg._continue_event.set()
        session.request("shutdown")
        return {"ok": True}

    return server
