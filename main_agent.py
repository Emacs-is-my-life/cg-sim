"""Agent-mode entry point for cg-sim.

Boots a long-lived MCP server on stdio, then drives a loop that
constructs a fresh `Simulator` for each requested run. The MCP daemon
outlives any single Simulator instance, so an agent can run the
simulator repeatedly via `restart_simulation` (optionally with a new
input YAML) without restarting the process.

Per-run state — `Log` writer thread, `Debugger` events, scheduler/HW
objects — is created inside `Simulator.__init__` exactly the same way
`main.py` does it, and torn down (`log.stop()`) before the next run
begins, so each iteration is clean and idempotent.
"""

import argparse
import sys
import traceback

from sim.core import Simulator
from sim.core.debug import AgentSession, start_agent_server


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args
    return args


def _construct_and_bind(session: AgentSession) -> Simulator | None:
    """Build a fresh Simulator and bind it to the session.

    Returns the Simulator on success, or `None` if construction raised
    (e.g. malformed input). On failure, prints the traceback to stderr
    and leaves `session.debugger` cleared so the next agent-issued
    `restart_simulation` can try a different input path.
    """
    try:
        sim = Simulator(session.next_input_path)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        session.unbind()
        return None

    sim.debugger.mode = "agent"
    sim.debugger.apply_env_breakpoints()
    session.bind(sim.debugger)
    return sim


def _await_next_action(session: AgentSession) -> str:
    """Block until the agent requests `restart` or `shutdown`, return it."""
    session.action_event.wait()
    action = session.next_action or "shutdown"
    session.action_event.clear()
    session.next_action = None
    return action


def _teardown(session: AgentSession, sim: Simulator | None) -> None:
    """Detach the session and stop the per-run log writer thread."""
    session.unbind()
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
        sim = _construct_and_bind(session)
        # Unblock any restart_simulation tool call waiting on the new
        # instance. On the first iteration this is a no-op (no waiter).
        session.restart_complete.set()

        if sim is None:
            # Construction failed; wait for the agent's next move.
            action = _await_next_action(session)
            if action == "shutdown":
                return 1
            continue  # action == "restart": try again with potentially-new input

        # Block until the agent calls `start_simulation`, then run.
        sim.debugger._start_event.wait()
        try:
            sim.run()
        except SystemExit:
            # sim.run() turns Ctrl-C into SystemExit(130); honor it.
            _teardown(session, sim)
            return 130
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            # Wake any agent waiter so it doesn't hang on the dead run.
            sim.debugger._simulation_finished = True
            sim.debugger._state_changed_event.set()

        # Wait for the agent to choose `restart` or `shutdown`.
        action = _await_next_action(session)
        _teardown(session, sim)
        if action == "shutdown":
            return 0
        # action == "restart": loop, building a fresh Simulator.


if __name__ == "__main__":
    sys.exit(main())
