"""Daemon-thread MCP server for agent-mode breakpoint control.

`start_agent_server` is invoked from `Debugger.welcome_prompt_agent`
early in simulator construction so that *every* breakpoint site —
including those that fire from `Simulator.__init__` before `run()` —
finds an active MCP server waiting on the other side of the suspension
event.

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
from typing import TYPE_CHECKING

import anyio
from mcp.server.stdio import stdio_server

from .agent_server import build_mcp_server

if TYPE_CHECKING:
    from .debug import Debugger


def start_agent_server(debugger: "Debugger") -> threading.Thread:
    """Spawn the MCP server on a daemon thread and return it.

    The caller (welcome_prompt_agent) should subsequently block on
    `debugger._start_event` so the main thread doesn't race ahead of the
    agent's flag-configuration step.
    """
    real_stdout_fd = os.dup(1)
    real_stdin_fd = os.dup(0)
    os.dup2(2, 1)

    real_stdout = open(real_stdout_fd, "wb", buffering=0)
    real_stdin = open(real_stdin_fd, "rb", buffering=0)

    server = build_mcp_server(debugger)

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
            # If the client disconnects, release any waiters so the
            # simulator doesn't deadlock — it'll just run uninterrupted
            # to completion.
            debugger._start_event.set()
            debugger._continue_event.set()

    thread = threading.Thread(
        target=_thread_target, name="cg-sim-mcp-server", daemon=True
    )
    thread.start()
    return thread
