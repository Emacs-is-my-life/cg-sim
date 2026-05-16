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

Lifecycle synchronization: `AgentSession` exposes one `threading.Condition`
plus an explicit `Phase` enum. The main loop (in `main_agent.py`) and
the FastMCP tool handlers cooperate through `transition_to`/`wait_until`
under the cv. This replaces an earlier pair of raw `threading.Event`s
(`action_event` + `restart_complete`) whose clear/set protocol could not
atomically express "wait until the *next* construction completes" — a
lost-edge race that caused `restart_simulation` called before the first
run to silently no-op. `Debugger`'s own three Events (`_continue_event`,
`_start_event`, `_state_changed_event`) coordinate breakpoint park/resume
on the worker thread and remain untouched here; they live on a different
axis (high-frequency, per-Debugger) and are guarded only by their own
internal locks.
"""

from __future__ import annotations

import enum
import os
import sys
import threading
from io import TextIOWrapper
from typing import TYPE_CHECKING, Any, Callable, Optional

import anyio
from mcp.server.stdio import stdio_server

if TYPE_CHECKING:
    from .debug import Debugger


class Phase(enum.Enum):
    """Lifecycle state of the simulator owned by the main thread.

    Transitions (all under `AgentSession.cv`):

        WAITING_FOR_CONFIG ──restart(input_path=…)──> CONSTRUCTING
             │
             └─shutdown─> SHUTDOWN

        CONSTRUCTING ──ok──> READY ──start──> RUNNING ──finish──> FINISHED
             │                  │                                     │
             │                  └──restart──> CONSTRUCTING             │
             │                  └──shutdown─> SHUTDOWN                 │
             │                                                         │
             ├─fail──> CONSTRUCT_FAILED ──restart──> CONSTRUCTING      │
             │                          └──shutdown─> SHUTDOWN         │
             │                                                         │
             └─shutdown (daemon-disconnect race)─> SHUTDOWN  <──restart/shutdown

    `WAITING_FOR_CONFIG` is the initial phase only when `main_agent.py`
    was launched without `-i`: no Simulator can be built until the agent
    calls `restart_simulation(input_path=...)`. With `-i`, the initial
    phase is `CONSTRUCTING` and the loop builds the default Simulator
    before serving tools, matching the historical behavior.
    """
    WAITING_FOR_CONFIG = "waiting_for_config"
    CONSTRUCTING = "constructing"
    READY = "ready"
    RUNNING = "running"
    FINISHED = "finished"
    CONSTRUCT_FAILED = "construct_failed"
    SHUTDOWN = "shutdown"


class AgentSession:
    """Shared state between the long-lived MCP daemon and the
    `main_agent.py` simulator loop.

    State invariants (held under `self.cv`):
      - `debugger is not None` iff `phase in {READY, RUNNING, FINISHED}`.
      - `next_input_path` is consulted by `_construct_and_bind` whenever
        the main loop is about to enter `CONSTRUCTING`; tools update it
        under `cv` before requesting a transition. It is `None` only
        when the server was started without `-i` and the agent has not
        yet supplied a path via `restart_simulation(input_path=...)`.
      - `next_overrides` is consulted alongside `next_input_path` for the
        same purpose — it carries Hydra-style overrides (e.g.
        `["scheduler.args.prefetch_window=8"]`) for the next
        construction. Initialized from `main_agent.py`'s unparsed CLI
        args so initial overrides flow through; thereafter mutated only
        by `restart_simulation(overrides=...)`.
      - `phase == SHUTDOWN` is terminal — `transition_to(SHUTDOWN)` is
        legal from every other phase, including `CONSTRUCTING` (used by
        the daemon-disconnect path so a hung initial bind doesn't trap
        the main thread).
    """

    def __init__(
        self,
        default_input_path: str | None = None,
        default_overrides: list[str] | None = None,
    ):
        self.default_input_path = default_input_path
        self.next_input_path: str | None = default_input_path
        self.next_overrides: list[str] = list(default_overrides or [])
        self.debugger: Optional["Debugger"] = None
        self.cv = threading.Condition()
        self.phase: Phase = (
            Phase.CONSTRUCTING if default_input_path is not None
            else Phase.WAITING_FOR_CONFIG
        )

    # -- Synchronization helpers (caller MUST hold or acquire `cv`) -----

    def transition_to(self, new_phase: Phase, **attrs: Any) -> None:
        """Atomically set `phase` (plus any attribute kwargs) and notify
        all waiters. `SHUTDOWN` is absorbing — once set, no other phase
        overwrites it (except the redundant SHUTDOWN-to-SHUTDOWN no-op).
        Attribute kwargs are still applied so the caller's bookkeeping
        completes, but the phase remains SHUTDOWN. Re-entrant: acquires
        `cv` if not held."""
        with self.cv:
            for k, v in attrs.items():
                setattr(self, k, v)
            if self.phase != Phase.SHUTDOWN:
                self.phase = new_phase
            self.cv.notify_all()

    def wait_until(
        self,
        predicate: Callable[[], bool],
        timeout: float | None = None,
    ) -> bool:
        """Block until `predicate()` is true. Returns True if the
        predicate became true, False on timeout. The predicate is
        re-evaluated under `cv` on every `notify_all` — that's the
        property that closes the lost-edge race a bare `Event` cannot."""
        with self.cv:
            return self.cv.wait_for(predicate, timeout=timeout)

    # -- Convenience views (cheap snapshots; no lock needed for reads) --

    def is_terminal(self) -> bool:
        return self.phase == Phase.SHUTDOWN


def start_agent_server(session: AgentSession) -> threading.Thread:
    """Spawn the MCP server on a daemon thread and return it."""
    from .agent_server import build_mcp_server

    # Mark this process as agent-mode so `Debugger.welcome_prompt`
    # short-circuits — its `input()` would race the MCP server for the
    # original fd 0 (both ends see it after the dup below).
    os.environ["CG_SIM_AGENT_MODE"] = "1"

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
            # Client disconnected. Wake any worker parked at a Debugger
            # event so the current run completes naturally, then transition
            # the session to SHUTDOWN — legal from every phase, including
            # CONSTRUCTING, so a hung initial bind doesn't trap the main
            # thread waiting on the cv.
            dbg = session.debugger
            if dbg is not None:
                dbg._start_event.set()
                dbg._continue_event.set()
                dbg._state_changed_event.set()
            session.transition_to(Phase.SHUTDOWN)

    thread = threading.Thread(
        target=_thread_target, name="cg-sim-mcp-server", daemon=True
    )
    thread.start()
    return thread
