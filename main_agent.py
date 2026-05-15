"""Agent-mode entry point for cg-sim.

Boots a long-lived MCP server on stdio, then drives a phase machine
that constructs / runs / tears down a Simulator on demand. The MCP
daemon outlives any single Simulator, so an agent can repeatedly call
`restart_simulation` (optionally with a new input YAML) without
restarting the process.

Per-run state — `Log` writer thread, `Debugger`, scheduler/HW objects —
is created inside `Simulator.__init__` exactly the same way `main.py`
does it, and torn down (`log.stop()`) before the next run begins.

Synchronization: the loop runs against an `AgentSession.Phase` machine
guarded by a single `threading.Condition` (see `sim/core/debug/agent_runner.py`).
Tool calls in the daemon thread (`agent_server.py`) drive phase transitions
under the same cv; the main loop reacts via `session.wait_until(...)`.
The old design used a pair of raw `threading.Event`s; the cv replaces
them to give atomic "decide-and-transition" semantics, closing a
lost-edge race when `restart_simulation` was called as the very first
tool call (before the initial `_construct_and_bind` had completed).
"""

import argparse
import sys
import traceback

from sim.core import Simulator
from sim.core.debug import AgentSession, start_agent_server
from sim.core.debug.agent_runner import Phase


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args
    return args


def _construct(session: AgentSession) -> Simulator | None:
    """Build a fresh Simulator using `session.next_input_path`.

    Returns the Simulator on success, or `None` if construction raised
    (e.g. malformed input). On failure, prints the traceback to stderr;
    the main loop transitions the session to `CONSTRUCT_FAILED` so any
    waiting tool gets an actionable error.
    """
    try:
        sim = Simulator(session.next_input_path)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return None
    # Mark agent-mode and apply env breakpoints BEFORE binding so an
    # agent that wins a race can never see a Debugger with mode="human"
    # or unconfigured BREAK_* flags.
    sim.debugger.mode = "agent"
    sim.debugger.BREAK_ON_ABORT = True
    sim.debugger.BREAK_ON_EXCEPTION = True
    sim.debugger.apply_env_breakpoints()
    return sim


def _teardown(sim: Simulator | None) -> None:
    """Stop the per-run log writer thread. Session unbinding happens
    via `transition_to(...)` in the main loop, not here."""
    if sim is None or sim.log is None:
        return
    try:
        sim.log.stop()
    except BaseException:
        traceback.print_exc(file=sys.stderr)


def main() -> int:
    args = _parse_args()
    if args.input is None:
        raise Exception("No simulation input file is supplied.")

    session = AgentSession(args.input)

    # Start the MCP daemon once, before any Simulator is constructed.
    # This dups stdio so subsequent simulator `print()` calls land on
    # stderr instead of corrupting the JSON-RPC stream.
    start_agent_server(session)

    while True:
        # --- CONSTRUCTING phase ---------------------------------------
        # session.phase is already CONSTRUCTING here (initial state, or
        # set by a tool that requested restart).
        if session.phase == Phase.SHUTDOWN:
            # Daemon disconnected during the previous loop's wait;
            # honor it.
            return 0

        sim = _construct(session)
        if sim is None:
            # Construction failed. Park in CONSTRUCT_FAILED until the
            # agent restarts (with a different path, presumably) or
            # shuts down.
            session.transition_to(Phase.CONSTRUCT_FAILED, debugger=None)
            session.wait_until(
                lambda: session.phase in (Phase.CONSTRUCTING, Phase.SHUTDOWN))
            if session.phase == Phase.SHUTDOWN:
                return 1
            continue  # back to CONSTRUCTING with possibly-updated input_path

        # --- READY phase ----------------------------------------------
        # Bind and notify; agent can now toggle breakpoints, start, or
        # request another restart.
        session.transition_to(Phase.READY, debugger=sim.debugger)

        # Wait for the agent to: start the run, request another restart
        # (legitimate "before the first run" path), or shut down.
        session.wait_until(lambda: session.phase != Phase.READY)

        if session.phase == Phase.SHUTDOWN:
            _teardown(sim)
            return 0
        if session.phase == Phase.CONSTRUCTING:
            # Pre-first-run restart — the bug case, now correct.
            _teardown(sim)
            session.transition_to(Phase.CONSTRUCTING, debugger=None)
            continue

        # --- RUNNING phase --------------------------------------------
        # The start_simulation tool already set _start_event before
        # transitioning to RUNNING, so sim.run() doesn't block at the
        # old _start_event.wait() seam (that wait has been removed).
        assert session.phase == Phase.RUNNING
        try:
            sim.run()
        except SystemExit:
            # sim.run() turns Ctrl-C into SystemExit(130); honor it.
            _teardown(sim)
            session.transition_to(Phase.SHUTDOWN, debugger=None)
            return 130
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            # Wake any agent waiter so it doesn't hang on the dead run.
            sim.debugger._simulation_finished = True
            sim.debugger._state_changed_event.set()

        # --- FINISHED phase -------------------------------------------
        # transition_to is "absorbing" for SHUTDOWN, so a shutdown
        # requested mid-run survives this transition and is honored by
        # the next wait_until below.
        session.transition_to(Phase.FINISHED)
        session.wait_until(
            lambda: session.phase in (Phase.CONSTRUCTING, Phase.SHUTDOWN))

        _teardown(sim)
        if session.phase == Phase.SHUTDOWN:
            return 0
        # phase == CONSTRUCTING — restart_simulation requested.
        session.transition_to(Phase.CONSTRUCTING, debugger=None)


if __name__ == "__main__":
    sys.exit(main())
