from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import sys
import textwrap
import threading
import traceback
from IPython.terminal.embed import InteractiveShellEmbed
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from enum import Enum, auto

from sim.core.sim_object import SimObject
from sim.core.log import Log, TrackID, Level

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from sim.core.engine import Engine
    from sim.core.system import System


_shell = InteractiveShellEmbed()


_RUNTIME_BREAK_TIP = (
    "Set BREAK_AT_* flags on any Node to break during the runtime stage.\n"
    "Example:\n"
    "  node.BREAK_AT_JOB_SUBMITTED  = True   # break when its Job is submitted\n"
    "  node.BREAK_AT_JOB_HEAD       = True   # break when its Job reaches the head of job_waiting\n"
    "  node.BREAK_AT_JOB_DISPATCHED = True   # break when its Job is dispatched to job_running\n"
    "  node.BREAK_AT_JOB_RETIRED    = True   # break when its Job is retired\n"
    "(Equivalent flags can also be set directly on Job objects.)"
)


@dataclass
class _Symbol:
    name: str
    _type: str
    description: str


def symbolPrint(variables: list[_Symbol]) -> None:
    """
    Pretty-print a list of `_Symbol` as an ASCII table with width-aligned
    columns. The `description` cell wraps across multiple lines when it
    exceeds the available width.
    """
    if not variables:
        return

    headers = ("Variable", "Type", "Description")
    name_w = max(len(headers[0]), *(len(v.name) for v in variables))
    type_w = max(len(headers[1]), *(len(v._type) for v in variables))

    # Borders + padding consume: "| " + " | " + " | " + " |" = 10 cols.
    term_w = shutil.get_terminal_size((100, 20)).columns
    desc_w = max(20, term_w - name_w - type_w - 10)

    sep = f"|{'-' * (name_w + 2)}+{'-' * (type_w + 2)}+{'-' * (desc_w + 2)}|"

    def _wrap_desc(desc: str) -> list[str]:
        lines: list[str] = []
        for paragraph in desc.splitlines() or [""]:
            lines.extend(textwrap.wrap(paragraph, width=desc_w) or [""])
        return lines

    def _row(name: str, _type: str, desc: str) -> str:
        desc_lines = _wrap_desc(desc)
        out: list[str] = []
        for i, dl in enumerate(desc_lines):
            n = name if i == 0 else ""
            t = _type if i == 0 else ""
            out.append(f"| {n:<{name_w}} | {t:<{type_w}} | {dl:<{desc_w}} |")
        return "\n".join(out)

    print(f"| {headers[0]:<{name_w}} | {headers[1]:<{type_w}} | {headers[2]:<{desc_w}} |")
    print(sep)
    for v in variables:
        print(_row(v.name, v._type, v.description))
    return


class Debugger(SimObject):
    """
    Debugging tool
    """

    def __init__(self, obj_id: int, name: str, log: Log):
        super().__init__(obj_id, name, log)
        self._log = log
        self.engine: Engine | None = None

        # Create subtracks for logging
        self._log.record(Log.subtrack(TrackID.Engine, self.id, "Debug"))

        # Debugging flags
        self.BREAK_BEFORE_COMPILE_STAGE: bool = False
        self.BREAK_AFTER_COMPILE_STAGE: bool = False
        self.BREAK_AFTER_LAYOUT_STAGE: bool = False
        self.BREAK_IN_RUNTIME_STAGE: bool = False
        self.BREAK_AFTER_RUNTIME_STAGE: bool = False
        # Safety-net: defaults On. Fires at each engine abort site (deadlock,
        # invalid Job submission) so the agent can inspect why & state before
        # the run tears down. Toggle off for fail-fast batch/CI runs.
        self.BREAK_ON_ABORT: bool = True
        # Hard-failure counterpart: defaults On. Fires when an uncaught
        # exception propagates out of `engine.run()` so the agent can inspect
        # the failing frame instead of chasing a stderr traceback after the
        # fact. Toggle off if you want exceptions to propagate silently.
        self.BREAK_ON_EXCEPTION: bool = True

        # Free-form scratchpad dict, mirroring the `.args` convention on
        # hardware/scheduler classes. Use it to stash notes, intermediate
        # values, working hypotheses — anything you want to carry across
        # breakpoints within one run. Lives for the lifetime of this
        # Debugger; cleared on `restart_simulation`. For records that
        # survive the run, use `record()` instead.
        self.args: dict[str, Any] = {}

        # User-supplied predicate `(engine: Engine, sys: System) -> bool`,
        # evaluated each runtime-loop iteration after retiring jobs and
        # before updating progress. Returning True fires
        # `break_in_runtime_stage[LAMBDA]`. Lets agents/humans express
        # arbitrary stop conditions without writing a new BREAK_* flag.
        # Auto-cleared if the predicate raises (so a buggy lambda can't
        # crash the run on every tick).
        self.break_lambda: Callable[[Engine, System], bool] | None = None

        # Active breakpoint context (set on entry, consumed by `help()`).
        self._current_breakpoint: str | None = None
        self._current_variables: list[_Symbol] = []
        self._current_tip: str | None = None

        # Mode: "human" drops into IPython; "agent" parks on an Event and
        # exposes the breakpoint state to an MCP server. Set by
        # `run_agent_mode()` before `engine.run()` is invoked.
        self.mode: str = "human"

        # Agent-mode suspension state. `start_simulation` and
        # `continue_simulation` are *blocking* RPCs: each releases the
        # worker via its event, then waits on `_state_changed_event` —
        # which the worker sets when it reaches the next breakpoint or
        # finishes — and returns the new state in the same response.
        # This eliminates client-side polling.
        self._continue_event = threading.Event()
        self._start_event = threading.Event()
        self._state_changed_event = threading.Event()
        self._wait_timeout: float = 50.0
        self._at_breakpoint: bool = False
        self._simulation_finished: bool = False
        self._exec_namespace: dict[str, Any] = {}
        return

    def log_counters(self) -> dict[str, Any] | None:
        return

    def log_states(self) -> dict[str, Any] | None:
        return

    def record(self, args: dict) -> None:
        try:
            json.dumps(args)
        except (TypeError, ValueError) as exc:
            print(f"Cannot log: argument is not JSON-serializable ({exc}).")
            return

        timestamp = 0 if self.engine is None else self.engine.timestamp_now
        self._log.record(Log.engine(self.id, "DEBUG_MSG", timestamp, args))
        return

    @property
    def log_path(self) -> Path:
        """Absolute path to the simulation log file.

        Where `record()` writes and where the final SIMULATION_RESULT
        / per-track events end up. Useful for post-mortem inspection
        after `simulation_finished=True`.
        """
        return self._log.result_path.resolve()

    def _break_lambda(self, engine, system) -> None:
        """Evaluate `self.break_lambda(engine, system)`; if it returns
        True, fire `break_in_runtime_stage[LAMBDA]`. Called once per
        runtime-loop iteration by Engine._runtime (after retiring jobs,
        before updating progress).

        Auto-clears `self.break_lambda` on any exception so a buggy
        predicate can't crash the run on every tick — the user/agent
        can set it again after fixing.

        The runtime-stage locals expected at the breakpoint (`debug`,
        `engine`, `timestamp_now`, `job_waiting`, `job_running`, `hw`,
        `trace`) are bound in this frame so `_enter_breakpoint`'s
        2-frames-up lookup picks them up. No `job` is bound: a lambda
        stop has no triggering Job.
        """
        if self.break_lambda is None:
            return
        try:
            should_break = self.break_lambda(engine, system)
        except BaseException:
            traceback.print_exc()
            print("[Debugger] break_lambda raised; cleared. Set it again to re-enable.")
            self.break_lambda = None
            return
        if should_break is not True:
            return

        # Bind the runtime-stage namespace for the breakpoint REPL.
        debug = self
        timestamp_now = engine.timestamp_now
        job_waiting = engine.job_waiting
        job_running = engine.job_running
        hw = system.hw
        trace = system.trace
        self.break_in_runtime_stage("LAMBDA")
        return

    def apply_env_breakpoints(self) -> None:
        """Pre-enable any BREAK_* flags listed in `CG_SIM_BREAKPOINTS`.

        Called per simulator construction by `main_agent.py` so each
        run honors the env var with a freshly-built Debugger.
        """
        raw = os.environ.get("CG_SIM_BREAKPOINTS", "")
        for name in (s.strip() for s in raw.split(",") if s.strip()):
            if name.startswith("BREAK_") and isinstance(getattr(self, name, None), bool):
                setattr(self, name, True)
        return

    def welcome_prompt(self) -> None:
        """
        Asks user to register breakpoints for debugging.

        Prints an Org-style table of available breakpoints (the
        `BREAK_*` flags) with their current On/Off status, then runs
        a REPL loop: each integer input toggles the matching breakpoint to
        On and re-renders the table. Returns when every breakpoint is On
        or when the user enters 'c'.

        No-op under agent mode: `main_agent.py` sets `CG_SIM_AGENT_MODE`
        before construction, and `input()` here would race the MCP server
        for stdin bytes (both ends inherit fd 0 from the dup in
        `start_agent_server`). Agents toggle flags via `toggle_breakpoint`
        instead.
        """
        if os.environ.get("CG_SIM_AGENT_MODE"):
            return

        breakpoints = [
            name for name, value in vars(self).items()
            if name.startswith("BREAK_") and isinstance(value, bool)
        ]
        if not breakpoints:
            return

        num_w = max(len("#"), len(str(len(breakpoints) - 1)))
        name_w = max(len("Breakpoint"), *(len(b) for b in breakpoints))
        stat_w = max(len("Status"), len("Off"))
        sep = f"|{'-' * (num_w + 2)}+{'-' * (name_w + 2)}+{'-' * (stat_w + 2)}|"
        table_h = len(breakpoints) + 2  # header + separator + rows

        def _print_table() -> None:
            print(f"| {'#':<{num_w}} | {'Breakpoint':<{name_w}} | {'Status':<{stat_w}} |")
            print(sep)
            for i, b in enumerate(breakpoints):
                status = "On" if getattr(self, b) else "Off"
                print(f"| {i:<{num_w}} | {b:<{name_w}} | {status:<{stat_w}} |")

        def _erase(n: int) -> None:
            if n > 0:
                # ESC[F = cursor up one line; ESC[2K = clear entire line.
                sys.stdout.write("\033[F\033[2K" * n)
                sys.stdout.flush()

        def _finish() -> None:
            # ESC[2J = erase entire screen; ESC[H = cursor to top-left.
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            print("Breakpoints registered. Starting simulator...")

        while True:
            _print_table()
            if all(getattr(self, b) for b in breakpoints):
                _finish()
                return

            # Inner loop: keep prompting until input is valid or 'c'. Each
            # `input()` consumes one screen line; each error message adds
            # one more — track both so we can erase everything on success.
            extra = 0
            while True:
                user_input = input("Enter breakpoint # to toggle ('c' to continue): ").strip()
                extra += 1
                if user_input.lower() == "c":
                    _finish()
                    return
                try:
                    idx = int(user_input)
                    if 0 <= idx < len(breakpoints):
                        break
                    print(f"Out of range: {idx}")
                except ValueError:
                    print(f"Invalid input: {user_input!r}")
                extra += 1

            setattr(self, breakpoints[idx], not getattr(self, breakpoints[idx]))
            _erase(table_h + extra)
        return

    def _print_banner(self, name: str) -> None:
        """
        Print the common header shown at every breakpoint, followed by an
        Org-style table of available REPL commands.
        """
        print()
        bar = "=" * max(60, len(name) + 20)
        print(bar)
        print(f"[BREAKPOINT] {name}")
        print(bar)
        print()

        commands = [
            ("debug.help()", "Show breakpoint context"),
            ("debug.record(dict)", "Leave information in simulation log file, in Engine -> Debug track."),
            ("debug.args", "Free-form scratchpad dict for ad-hoc notes/findings during this run."),
            ("debug.break_lambda", "Set to fn(engine, sys)->bool; fires break_in_runtime_stage[LAMBDA] when True."),
            ("debug.log_path", "Path to the simulation log file (where debug.record() writes)."),
            ("exit()", "Continue simulator execution"),
        ]
        cmd_w = max(len("Command"), *(len(c) for c, _ in commands))
        desc_w = max(len("Description"), *(len(d) for _, d in commands))
        sep = f"|{'-' * (cmd_w + 2)}+{'-' * (desc_w + 2)}|"
        print(f"| {'Command':<{cmd_w}} | {'Description':<{desc_w}} |")
        print(sep)
        for cmd, desc in commands:
            print(f"| {cmd:<{cmd_w}} | {desc:<{desc_w}} |")
        print()
        return

    def help(self) -> None:
        """
        Re-print the entry message (banner + variable table + optional
        tip) for the currently-active breakpoint. Intended to be called
        from the REPL.
        """
        if self._current_breakpoint is None:
            print("No active breakpoint.")
            return
        self._print_banner(self._current_breakpoint)
        symbolPrint(self._current_variables)
        if self._current_tip:
            print()
            print("Tip:")
            print(self._current_tip)
        return

    def _enter_breakpoint(self, name: str, variables: list[_Symbol], tip: str | None = None) -> None:
        """
        Register `name`/`variables`/`tip` as the active breakpoint and
        dispatch to the human (IPython) or agent (MCP) handoff depending
        on `self.mode`.
        """
        self._current_breakpoint = name
        self._current_variables = variables
        self._current_tip = tip

        # Frames: 0=_enter_breakpoint, 1=break_after_*, 2=actual caller.
        caller_frame = sys._getframe(2)

        if self.mode == "agent":
            self._handoff_to_agent(caller_frame)
        else:
            self.help()
            self._handoff_to_human(caller_frame)

        # Clear active breakpoint context.
        self._current_breakpoint = None
        self._current_variables = []
        self._current_tip = None
        self._exec_namespace = {}

        if self.mode == "human":
            # ESC[2J = erase entire screen; ESC[H = move cursor to top-left.
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            print(f"[{name}] Resuming simulator execution...")
        return

    def _handoff_to_human(self, caller_frame) -> None:
        """Drop into the embedded IPython shell scoped to caller_frame."""
        _shell.mainloop(
            local_ns=caller_frame.f_locals,
            module=sys.modules.get(caller_frame.f_globals.get("__name__")),
        )
        return

    def _handoff_to_agent(self, caller_frame) -> None:
        """
        Park the worker thread until the MCP server signals `continue`.
        The breakpoint's locals are snapshotted into `_exec_namespace`
        so agent-side `execute(code)` calls share a persistent namespace
        across one breakpoint. `debug` is exposed so the agent can use
        `debug.record(...)` from inside `execute`.
        """
        ns = dict(caller_frame.f_locals)
        ns.update(caller_frame.f_globals)
        ns["debug"] = self
        self._exec_namespace = ns
        self._continue_event.clear()
        self._at_breakpoint = True
        self._state_changed_event.set()
        try:
            self._continue_event.wait()
        finally:
            self._at_breakpoint = False
        return

    # -------- Agent-mode helpers (called by MCP tools) -----------------

    def agent_current_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the current breakpoint."""
        return {
            "at_breakpoint": self._at_breakpoint,
            "breakpoint": self._current_breakpoint,
            "variables": [
                {"name": v.name, "type": v._type, "description": v.description}
                for v in self._current_variables
            ],
            "tip": self._current_tip,
            "simulation_finished": self._simulation_finished,
        }

    def agent_list_breakpoints(self) -> dict[str, bool]:
        """Return a {flag_name: enabled} mapping for all BREAK_* flags."""
        return {
            name: getattr(self, name)
            for name in vars(self)
            if name.startswith("BREAK_") and isinstance(getattr(self, name), bool)
        }

    def agent_toggle_breakpoint(self, name: str) -> dict[str, Any]:
        """Flip a BREAK_* flag. Returns the new value or an error."""
        if not name.startswith("BREAK_") or not isinstance(getattr(self, name, None), bool):
            return {"ok": False, "error": f"No such breakpoint flag: {name!r}"}
        new_value = not getattr(self, name)
        setattr(self, name, new_value)
        return {"ok": True, "name": name, "new_value": new_value}

    def agent_execute(self, code: str) -> dict[str, Any]:
        """
        Execute `code` against the persistent breakpoint namespace.
        Echoes the value of a bare last expression like the Python REPL,
        captures stdout, and returns tracebacks as strings rather than
        raising. No-op if not currently parked at a breakpoint.
        """
        if not self._at_breakpoint:
            return {"ok": False, "error": "Not at a breakpoint."}

        stdout_buf = io.StringIO()
        try:
            # Mirror Python REPL semantics: if the final top-level statement
            # is a bare expression, evaluate it and echo its repr (unless
            # None). Everything before runs as `exec`. Both halves share one
            # namespace dict so assignments in the head are visible to the
            # trailing expression.
            tree = ast.parse(code, "<mcp>", "exec")
            with contextlib.redirect_stdout(stdout_buf):
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    head = ast.Module(body=tree.body[:-1], type_ignores=[])
                    tail = ast.Expression(body=tree.body[-1].value)
                    exec(compile(head, "<mcp>", "exec"), self._exec_namespace)
                    value = eval(compile(tail, "<mcp>", "eval"), self._exec_namespace)
                    if value is not None:
                        print(repr(value))
                else:
                    exec(compile(tree, "<mcp>", "exec"), self._exec_namespace)
            return {"ok": True, "output": stdout_buf.getvalue(), "error": None}
        except BaseException:
            return {
                "ok": False,
                "output": stdout_buf.getvalue(),
                "error": traceback.format_exc(),
            }

    def _wait_for_state_change(self) -> dict[str, Any]:
        """Block until the worker reaches a breakpoint or finishes.

        Returns the same fields as `agent_current_state` plus `ok` and
        `timed_out`. On timeout, the caller can read `current_state` to
        decide whether to keep waiting.
        """
        reached = self._state_changed_event.wait(timeout=self._wait_timeout)
        return {"ok": True, "timed_out": not reached, **self.agent_current_state()}

    def agent_continue(self) -> dict[str, Any]:
        """Release the worker, then block until the next observable state.

        Returns when the simulator reaches its next breakpoint or runs
        to completion, with the resulting state in the response — so the
        agent never has to poll.
        """
        if not self._at_breakpoint:
            return {"ok": False, "error": "Not at a breakpoint."}
        self._state_changed_event.clear()
        self._continue_event.set()
        return self._wait_for_state_change()

    def agent_start_simulation(self) -> dict[str, Any]:
        """Release the simulator from pre-run park; block until first state.

        Like `agent_continue`, this is a blocking RPC: it returns once
        the simulator hits its first enabled breakpoint or finishes.
        """
        if self._start_event.is_set():
            return {"ok": False, "error": "Simulation already started."}
        self._state_changed_event.clear()
        self._start_event.set()
        return self._wait_for_state_change()

    def notify_simulation_finished(self) -> None:
        """Mark the run as cleanly finished and wake any waiter.

        Called by `Simulator.run()` after `engine.run()` returns. In
        human mode this is a no-op; in agent mode the daemon stays
        alive across runs, so no stdio-flush sleep is needed here.
        """
        if self.mode != "agent":
            return
        self._simulation_finished = True
        self._state_changed_event.set()

    def break_before_compile_stage(self) -> None:
        self._enter_breakpoint("break_before_compile_stage", [
            _Symbol("debug", "Debugger", "Debugger — debug.record(dict) to log, debug.help() to re-print, debug.args dict for notes, debug.break_lambda for custom-predicate runtime stop, debug.log_path for log file path."),
            _Symbol("engine", "Engine", "Engine — exposes engine.sched (scheduler), engine.sys, engine.signal_abort, engine.job_stats, and internal state."),
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor"),
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
        ])
        return

    def break_after_compile_stage(self) -> None:
        self._enter_breakpoint("break_after_compile_stage", [
            _Symbol("debug", "Debugger", "Debugger — debug.record(dict) to log, debug.help() to re-print, debug.args dict for notes, debug.break_lambda for custom-predicate runtime stop, debug.log_path for log file path."),
            _Symbol("engine", "Engine", "Engine — exposes engine.sched (scheduler), engine.sys, engine.signal_abort, engine.job_stats, and internal state."),
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor"),
        ], tip=_RUNTIME_BREAK_TIP)
        return

    def break_after_layout_stage(self, hw) -> None:
        self._enter_breakpoint("break_after_layout_stage", [
            _Symbol("debug", "Debugger", "Debugger — debug.record(dict) to log, debug.help() to re-print, debug.args dict for notes, debug.break_lambda for custom-predicate runtime stop, debug.log_path for log file path."),
            _Symbol("engine", "Engine", "Engine — exposes engine.sched (scheduler), engine.sys, engine.signal_abort, engine.job_stats, and internal state."),
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor"),
        ], tip=_RUNTIME_BREAK_TIP)
        return

    def break_in_runtime_stage(self, bp_name: str) -> None:
        # `LAMBDA` fires from _break_lambda, where no triggering job exists.
        # Per-Job breakpoints (JOB_*) bind `job` in the caller's frame.
        variables = [
            _Symbol("debug", "Debugger", "Debugger — debug.record(dict) to log, debug.help() to re-print, debug.args dict for notes, debug.break_lambda for custom-predicate runtime stop, debug.log_path for log file path."),
            _Symbol("engine", "Engine", "Engine — exposes engine.sched (scheduler), engine.sys, engine.signal_abort, engine.job_stats, and internal state."),
            _Symbol("timestamp_now", "float", "Current simulator time"),
        ]
        if bp_name != "LAMBDA":
            variables.append(
                _Symbol("job", "BaseJob", "Job triggered this breakpoint")
            )
        variables.extend([
            _Symbol("job_waiting", "list[BaseJob]", "Queue of jobs waiting to be dispatched"),
            _Symbol("job_running", "list[BaseJob]", "Queue of currently running jobs"),
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor"),
        ])
        self._enter_breakpoint(f"break_in_runtime_stage[{bp_name}]", variables)
        return

    def break_on_abort(self) -> None:
        """Fired by `Engine._log_abort` for *any* abort path — engine-
        internal deadlocks, scheduler-issued `sys.abort(...)`, job
        assertion failures, mutation invariant breaks. Logging happens
        first; this breakpoint fires after it but before `signal_abort`
        is set, so the agent can inspect `abort_args` and the live
        runtime state.

        Caller's frame (`_log_abort`) must bind: `debug`, `engine`,
        `abort_args`, `timestamp_now`, `job_waiting`, `job_running`,
        `hw`, `trace`. `_enter_breakpoint`'s 2-frames-up lookup reads
        them from there.

        No discriminator in the breakpoint name — `abort_args['msg']`
        and `abort_args['from']` describe WHY; the agent reads them.
        """
        self._enter_breakpoint("break_on_abort", [
            _Symbol("debug", "Debugger", "Debugger — debug.record(dict) to log, debug.help() to re-print, debug.args dict for notes, debug.break_lambda for custom-predicate runtime stop, debug.log_path for log file path."),
            _Symbol("engine", "Engine", "Engine — exposes engine.sched (scheduler), engine.sys, engine.signal_abort, engine.job_stats, and internal state."),
            _Symbol("abort_args", "dict", "The args dict logged by _log_abort. `msg` and `from` describe the abort; specific aborts (e.g. runtime deadlock) may add `job` details."),
            _Symbol("abort_stack", "list[FrameInfo]", "Full call chain at the abort site (via inspect.stack()). abort_stack[0]=_log_abort itself; [1]=its direct caller (e.g. Engine._layout, System.abort, or your scheduler's runtime). Each entry: .filename, .lineno, .function, .frame.f_locals — read .frame.f_locals to inspect that frame's variables."),
            _Symbol("timestamp_now", "float", "Simulator time when the abort was raised."),
            _Symbol("job_waiting", "list[BaseJob]", "Queue of jobs waiting to be dispatched (head is usually the stuck job for deadlocks)"),
            _Symbol("job_running", "list[BaseJob]", "Queue of currently running jobs (empty at deadlock)"),
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor"),
        ])
        return

    def break_on_exception(self, exc_type_name: str) -> None:
        """Fired by `Simulator.run`'s outermost handler when an uncaught
        exception propagates out of `engine.run()`. Hard-failure
        counterpart of `break_on_abort` — same shape, different source:
        the failing frames are no longer on the live Python stack
        (they've unwound during exception propagation), so the call
        chain comes from `exception.__traceback__` instead of
        `inspect.stack()`.

        Caller's frame must bind: `debug`, `engine`, `exception`,
        `exception_origin` (dict), `exception_stack` (list of
        traceback objects walked from `exception.__traceback__`).

        Breakpoint name carries the exception type for at-a-glance
        triage from `current_state.breakpoint` (no `execute` needed):
        `break_on_exception[KeyError]`, etc.
        """
        self._enter_breakpoint(f"break_on_exception[{exc_type_name}]", [
            _Symbol("debug", "Debugger", "Debugger — debug.record(dict) to log, debug.help() to re-print, debug.args dict for notes, debug.break_lambda for custom-predicate runtime stop, debug.log_path for log file path."),
            _Symbol("engine", "Engine", "Engine — exposes engine.sched (scheduler), engine.sys, engine.signal_abort, engine.job_stats, and internal state."),
            _Symbol("exception", "BaseException", "The caught exception object. Read `str(exception)`, `exception.args`, `exception.__cause__` for further detail."),
            _Symbol("exception_origin", "dict", "Where the exception was raised: {'file': str, 'line': int, 'function': str}. Same info as the deepest traceback frame — surfaced top-level for at-a-glance triage."),
            _Symbol("exception_stack", "list[traceback]", "Full traceback chain walked from exception.__traceback__. [0]=outermost frame (engine.run); [-1]=failing frame (== exception_origin). Each entry: .tb_frame, .tb_lineno, .tb_frame.f_code.co_filename, .tb_frame.f_code.co_name, .tb_frame.f_locals — read .tb_frame.f_locals to inspect that frame's variables. Same navigation pattern as abort_stack."),
        ])
        return

    @staticmethod
    def _exception_origin(exc: BaseException) -> dict:
        """Deepest traceback frame as a {file, line, function} dict —
        i.e., the actual raise site. Returns None values if `exc` has
        no traceback (shouldn't happen for a normally-raised exception
        caught at runtime, but handled defensively)."""
        tb = exc.__traceback__
        if tb is None:
            return {"file": None, "line": None, "function": None}
        while tb.tb_next is not None:
            tb = tb.tb_next
        return {
            "file": tb.tb_frame.f_code.co_filename,
            "line": tb.tb_lineno,
            "function": tb.tb_frame.f_code.co_name,
        }

    @staticmethod
    def _exception_stack(exc: BaseException) -> list:
        """Flatten `exc.__traceback__` into a list of traceback objects
        ordered outer→inner (parallel to `abort_stack` semantics but
        sourced from the preserved traceback rather than the live
        stack, since the stack has unwound by the time we catch)."""
        chain = []
        tb = exc.__traceback__
        while tb is not None:
            chain.append(tb)
            tb = tb.tb_next
        return chain

    def break_after_runtime_stage(self) -> None:
        self._enter_breakpoint("break_after_runtime_stage", [
            _Symbol("debug", "Debugger", "Debugger — debug.record(dict) to log, debug.help() to re-print, debug.args dict for notes, debug.break_lambda for custom-predicate runtime stop, debug.log_path for log file path."),
            _Symbol("engine", "Engine", "Engine — exposes engine.sched (scheduler), engine.sys, engine.signal_abort, engine.job_stats, and internal state."),
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor"),
        ])
        return
