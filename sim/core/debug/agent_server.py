"""MCP server exposing the Debugger to an agentic LLM client.

The server is transport-agnostic; `agent_runner.run_agent_mode` wires it
to a stdio transport whose real stdout fd is reserved separately from
the simulator's `print()` traffic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from .debug import Debugger


_SERVER_INSTRUCTIONS = (
    "Interactive debugger for the cg-sim compute graph simulator.\n"
    "\n"
    "Typical session:\n"
    "  1. Call `list_breakpoints` to see available BREAK_* flags.\n"
    "  2. Call `toggle_breakpoint` to enable the ones you care about.\n"
    "  3. Call `start_simulation` to release the simulator.\n"
    "  4. Poll `current_state` until `at_breakpoint=True`.\n"
    "  5. Use `execute` to inspect/mutate state (e.g. `trace.node_map`,\n"
    "     `job.BREAK_AT_JOB_RETIRED = True`, `debug.record({...})`).\n"
    "  6. Call `continue_simulation` to resume; repeat 4-6 until\n"
    "     `current_state` reports `simulation_finished=True`."
)


def build_mcp_server(debugger: "Debugger") -> FastMCP:
    """Construct a FastMCP server bound to `debugger`."""
    server = FastMCP(name="cg-sim-debugger", instructions=_SERVER_INSTRUCTIONS)

    @server.tool(
        description=(
            "Start the simulator. Must be called once after configuring "
            "breakpoint flags. Returns immediately; use `current_state` "
            "to track progress."
        )
    )
    def start_simulation() -> dict[str, Any]:
        return debugger.agent_start_simulation()

    @server.tool(
        description=(
            "Return the current debugger state: whether the simulator is "
            "parked at a breakpoint, the breakpoint name, available "
            "variables and types, tip text, and whether the simulation "
            "has finished."
        )
    )
    def current_state() -> dict[str, Any]:
        return debugger.agent_current_state()

    @server.tool(
        description="Return a mapping of all BREAK_* flags to their On/Off status."
    )
    def list_breakpoints() -> dict[str, bool]:
        return debugger.agent_list_breakpoints()

    @server.tool(
        description=(
            "Toggle a BREAK_* flag on the Debugger. Argument is the flag "
            "name (e.g. 'BREAK_AFTER_COMPILE_STAGE'). Returns the new value."
        )
    )
    def toggle_breakpoint(name: str) -> dict[str, Any]:
        return debugger.agent_toggle_breakpoint(name)

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
        return debugger.agent_execute(code)

    @server.tool(
        description=(
            "Release the worker thread so the simulator resumes. Only "
            "callable when `at_breakpoint=True`."
        )
    )
    def continue_simulation() -> dict[str, Any]:
        return debugger.agent_continue()

    return server
