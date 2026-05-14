"""Daemon-thread MCP server for agent-mode breakpoint control.

`start_agent_server` is invoked once from `main_agent.py` at process
start so the daemon outlives any single Simulator instance. The
session handle it receives is re-bound to a fresh Debugger at the
start of each run, so every breakpoint site — including ones that
fire from `Simulator.__init__` before `run()` — finds an active MCP
server waiting on the other side of the suspension event.

The simulator's `print()` traffic would corrupt the JSON-RPC stream if
both shared fd 1. We duplicate the real stdin/stdout fds for MCP, then
redirect fd 1 to fd 2 (stderr) so every simulator `print()` lands on
stderr — keeping the protocol stream clean without touching engine code.
"""

from __future__ import annotations

import os
import sys
import threading
from io import TextIOWrapper
from typing import TYPE_CHECKING, Optional

import anyio
from mcp.server.stdio import stdio_server

if TYPE_CHECKING:
    from .debug import Debugger


class AgentSession:
    """Shared state between the long-lived MCP daemon and the
    `main_agent.py` simulator loop.

    The daemon owns one of these. Each simulator run binds its
    `Debugger` via `bind()`, and on completion the main loop waits on
    `action_event` for the agent to request either a `restart` (build
    a fresh Simulator with `next_input_path`) or a `shutdown`.
    """

    def __init__(self, default_input_path: str):
        self.default_input_path = default_input_path
        self.next_input_path: str = default_input_path
        self.debugger: Optional["Debugger"] = None
        self.action_event = threading.Event()
        self.next_action: Optional[str] = None  # "restart" | "shutdown"
        self.restart_complete = threading.Event()

    def bind(self, debugger: "Debugger") -> None:
        self.debugger = debugger

    def unbind(self) -> None:
        self.debugger = None

    def request(self, action: str) -> None:
        """Signal the main loop's next move. Idempotent within one cycle."""
        self.next_action = action
        self.action_event.set()


def start_agent_server(session: AgentSession) -> threading.Thread:
    """Spawn the MCP server on a daemon thread and return it."""
    from .agent_server import build_mcp_server

    real_stdout_fd = os.dup(1)
    real_stdin_fd = os.dup(0)
    os.dup2(2, 1)

    real_stdout = open(real_stdout_fd, "wb", buffering=0)
    real_stdin = open(real_stdin_fd, "rb", buffering=0)

    server = build_mcp_server(session)

    async def _run_server() -> None:
        stdin_async = anyio.wrap_file(
            TextIOWrapper(real_stdin, encoding="utf-8", errors="replace")
        )
        stdout_async = anyio.wrap_file(
            TextIOWrapper(real_stdout, encoding="utf-8")
        )
        async with stdio_server(stdin_async, stdout_async) as (read_stream, write_stream):
            await server._mcp_server.run(
                read_stream,
                write_stream,
                server._mcp_server.create_initialization_options(),
            )

    def _thread_target() -> None:
        try:
            anyio.run(_run_server)
        except BaseException:
            import traceback as _tb
            _tb.print_exc(file=sys.stderr)
        finally:
            # Client disconnected — wake the current run so the simulator
            # finishes naturally, then push the main loop toward shutdown.
            dbg = session.debugger
            if dbg is not None:
                dbg._start_event.set()
                dbg._continue_event.set()
                dbg._state_changed_event.set()
            session.request("shutdown")

    thread = threading.Thread(
        target=_thread_target, name="cg-sim-mcp-server", daemon=True
    )
    thread.start()
    return thread
