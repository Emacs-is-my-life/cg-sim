"""MCP server exposing the Debugger to an agentic LLM client.

Tools forward to `session.debugger`, which the `main_agent.py` loop
re-binds for each fresh Simulator instance. When no run is bound
(between simulations), forwarding tools return an error pointing the
caller at `restart_simulation`.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from .agent_runner import Phase

if TYPE_CHECKING:
    from .agent_runner import AgentSession


_SERVER_INSTRUCTIONS = (
    "Interactive debugger for the cg-sim compute graph simulator.\n"
    "\n"
    "== Workflow ==\n"
    "  1. `list_breakpoints` — see available BREAK_* flags and their\n"
    "     current On/Off status.\n"
    "  2. `toggle_breakpoint(name)` — flip a flag.\n"
    "  3. `start_simulation` — release the simulator (blocking). Returns\n"
    "     once the first breakpoint fires or the run finishes, with the\n"
    "     resulting state in the same response.\n"
    "  4. `execute(code)` — run Python against the parked breakpoint's\n"
    "     namespace (see Namespace and Debugger methods below). A bare\n"
    "     last expression is echoed REPL-style in `output`.\n"
    "  5. `continue_simulation` — resume (blocking). Returns at the next\n"
    "     breakpoint or `simulation_finished=true`.\n"
    "  6. Repeat 4-5. When `simulation_finished=true`, call\n"
    "     `restart_simulation(input_path=None, overrides=None)` for a\n"
    "     fresh run (optionally with a new YAML and/or Hydra-style\n"
    "     config overrides like\n"
    "     `['scheduler.args.prefetch_window=8']`) or `shutdown` to end\n"
    "     the process.\n"
    "\n"
    "  If `start_simulation`/`continue_simulation` returns\n"
    "  `timed_out=true`, re-read with `current_state` (cheap; does not\n"
    "  release the worker again).\n"
    "\n"
    "== Stage breakpoints (BREAK_* flags on the Debugger) ==\n"
    "  BREAK_BEFORE_COMPILE_STAGE — before the scheduler's compile pass.\n"
    "  BREAK_AFTER_COMPILE_STAGE  — after compile, before layout.\n"
    "  BREAK_AFTER_LAYOUT_STAGE   — after initial tensor placement.\n"
    "  BREAK_IN_RUNTIME_STAGE     — master switch enabling per-Job\n"
    "                               breakpoints below.\n"
    "  BREAK_AFTER_RUNTIME_STAGE  — after the runtime loop exits.\n"
    "\n"
    "== Per-Node / per-Job breakpoints (fine-grained, runtime stage) ==\n"
    "  Requires BREAK_IN_RUNTIME_STAGE=True. Settable on any Node *or*\n"
    "  BaseJob object (Jobs inherit the flag from their Node at creation):\n"
    "    BREAK_AT_JOB_SUBMITTED   — job enqueued onto job_waiting.\n"
    "    BREAK_AT_JOB_HEAD        — job reaches the head of job_waiting.\n"
    "    BREAK_AT_JOB_DISPATCHED  — job moves into job_running.\n"
    "    BREAK_AT_JOB_RETIRED     — job completes and is retired.\n"
    "  Typical flow: enable BREAK_AFTER_COMPILE_STAGE, inspect\n"
    "  `trace.node_map` to find the target node, arm the flag, then\n"
    "  `continue_simulation`. Flags persist across continues — disarm\n"
    "  (`= False`) when done if you want the run to finish without\n"
    "  stopping again.\n"
    "\n"
    "  When one fires, the breakpoint name is\n"
    "  `break_in_runtime_stage[JOB_<PHASE>]` and `job` in the namespace\n"
    "  is the triggering job.\n"
    "\n"
    "== Namespace in `execute` ==\n"
    "  The exact set is in `current_state.variables` (name + type +\n"
    "  description) for the current breakpoint. Every breakpoint always\n"
    "  binds at least `debug` (the Debugger) and `engine` (the Engine);\n"
    "  most also bind `trace`, `hw`, and runtime breakpoints add\n"
    "  `timestamp_now`, `job`, `job_waiting`, `job_running`.\n"
    "\n"
    "  The namespace persists across `execute` calls within one\n"
    "  breakpoint — locals you assign survive until `continue_simulation`.\n"
    "\n"
    "== Debugger methods / accessors callable from `execute` ==\n"
    "  debug.record(dict_args) — write a JSON-serializable dict into the\n"
    "    simulation log file under track Engine → Debug. Use this to\n"
    "    leave breadcrumbs that survive after the agent session ends.\n"
    "    Raises if `dict_args` isn't JSON-serializable.\n"
    "  debug.args — free-form scratchpad dict. Stash anything you want\n"
    "    to remember across breakpoints within this run (findings,\n"
    "    working hypotheses, intermediate values). Per-run state;\n"
    "    cleared on `restart_simulation`. Use `debug.record(...)` for\n"
    "    persistent log entries instead.\n"
    "  debug.break_lambda — custom-predicate runtime breakpoint. Assign\n"
    "    a callable `(engine, sys) -> bool`; once per runtime-loop tick\n"
    "    (after retiring jobs, before progress update), it's evaluated\n"
    "    and a return value of True (strictly) fires\n"
    "    `break_in_runtime_stage[LAMBDA]`. No master flag needed — just\n"
    "    setting the field is the enable. Auto-clears if the predicate\n"
    "    raises, so a buggy lambda can't crash the run. Example:\n"
    "      execute(\"debug.break_lambda = lambda engine, sys: \"\n"
    "              \"engine.timestamp_now > 5_000_000\")\n"
    "    Set to None to disable. At the LAMBDA breakpoint, `job` is not\n"
    "    bound (no triggering Job); other runtime locals are.\n"
    "  debug.help() — re-print the current breakpoint context (banner +\n"
    "    variables table + tip) to stdout. Captured in `output`.\n"
    "    Redundant with `current_state` for agents but available.\n"
    "  debug.engine — the same Engine object reachable as the `engine`\n"
    "    variable.\n"
    "  debug.log_path — absolute Path of the simulation log file (where\n"
    "    debug.record() writes). Useful for post-mortem inspection after\n"
    "    `simulation_finished=True`; read it with normal Python file IO\n"
    "    from inside `execute`, e.g. `pathlib.Path(debug.log_path).read_text()`.\n"
    "\n"
    "== Abort breakpoint (BREAK_ON_ABORT, default On) ==\n"
    "  Generic soft-failure safety net. Every abort path in the\n"
    "  simulator funnels through `Engine._log_abort(args)` — engine-\n"
    "  internal deadlocks, scheduler-issued `sys.abort(...)`, job\n"
    "  assertion failures (transfer/compute), mutation invariant\n"
    "  breaks, and any user-added abort callers in new schedulers or\n"
    "  hardware. With BREAK_ON_ABORT enabled (default), `_log_abort`\n"
    "  fires the `break_on_abort` breakpoint *after* logging the\n"
    "  reason but *before* `signal_abort=True` tears the run down.\n"
    "\n"
    "  Namespace at the breakpoint: `abort_args` (the dict that was\n"
    "  logged — read `abort_args['msg']` and `abort_args['from']` to\n"
    "  see WHY; specific aborts add details, e.g. runtime deadlock\n"
    "  puts the stuck job's identity in `abort_args['job']`), the\n"
    "  standard runtime locals (`engine`, `timestamp_now`,\n"
    "  `job_waiting`, `job_running`, `hw`, `trace`), and\n"
    "  `abort_stack` — full call chain (`inspect.stack()`) so you can\n"
    "  inspect the frame that actually decided to abort, not just\n"
    "  `_log_abort`. Pattern:\n"
    "    execute(\"[(f.filename.rsplit('/')[-1], f.lineno, f.function) \"\n"
    "            \"for f in abort_stack[:8]]\")\n"
    "    # find the interesting frame (e.g. your scheduler), then:\n"
    "    execute(\"dict(abort_stack[3].frame.f_locals)\")\n"
    "  `abort_stack[0]` is `_log_abort` itself; `[1]` is its direct\n"
    "  caller; walk further up for nested context.\n"
    "\n"
    "  Continuing proceeds with the normal abort flow\n"
    "  (`signal_abort=True`; the run ends with\n"
    "  `simulation_success=false`). To fail-fast in batch/CI without\n"
    "  pausing, `toggle_breakpoint(\"BREAK_ON_ABORT\")` to disable.\n"
    "\n"
    "== Exception breakpoint (BREAK_ON_EXCEPTION, default On) ==\n"
    "  Hard-failure counterpart of BREAK_ON_ABORT. When an uncaught\n"
    "  exception propagates out of `engine.run()` — bug in your\n"
    "  scheduler, HW model, trace loader, anywhere — `Simulator.run`'s\n"
    "  outermost handler:\n"
    "    1. Writes a `SIMULATION_EXCEPTION` event to the log (type,\n"
    "       message, full traceback), so post-mortem reading is\n"
    "       symmetric with the soft-abort case.\n"
    "    2. Fires `break_on_exception[<ExceptionType>]` if enabled.\n"
    "    3. Ends the run gracefully (no re-raise) so\n"
    "       `restart_simulation` works.\n"
    "\n"
    "  The breakpoint name carries the exception type for at-a-glance\n"
    "  triage from `current_state.breakpoint` (no `execute` needed):\n"
    "  `break_on_exception[KeyError]`, `[AttributeError]`, etc.\n"
    "\n"
    "  Namespace at the breakpoint:\n"
    "    exception        — the caught BaseException.\n"
    "    exception_origin — {file, line, function} pointing to the\n"
    "                       raise site. The deepest traceback frame —\n"
    "                       surfaced top-level for one-glance lookup.\n"
    "    exception_stack  — list of traceback objects walked from\n"
    "                       `exception.__traceback__`. [0]=outermost\n"
    "                       (engine.run); [-1]=failing frame. Same\n"
    "                       navigation as abort_stack but sourced from\n"
    "                       __traceback__ (the live stack has unwound\n"
    "                       by the time we catch):\n"
    "                         dict(exception_stack[-1].tb_frame.f_locals)\n"
    "    debug, engine    — standard.\n"
    "\n"
    "  Toggle off for fail-fast: `toggle_breakpoint(\"BREAK_ON_EXCEPTION\")`.\n"
    "\n"
    "== Config overrides at restart ==\n"
    "  `restart_simulation(overrides=[...])` accepts a list of\n"
    "  Hydra-style override strings — same syntax as appending extra\n"
    "  args to `python main.py -i <yaml>` on the shell. Use it to sweep\n"
    "  scheduler/hardware/trace knobs without editing the YAML or\n"
    "  hand-mutating constructed objects (which only catches values\n"
    "  read live; values consumed inside `__init__` won't re-derive).\n"
    "    overrides=['scheduler.args.prefetch_window=8']\n"
    "    overrides=['hardware.memory.0.args.memory_size_KB=10485760']\n"
    "    overrides=['+debug=on', 'logger.args.log_level=3']  # +adds key\n"
    "  Semantics: `None` (default) keeps the previous overrides (sticky\n"
    "  across restarts; initialized from the CLI args given to\n"
    "  `main_agent.py`). `[]` clears them. A list replaces. The applied\n"
    "  list is echoed back as `overrides` in the response so the agent\n"
    "  can verify. Invalid override strings raise during construction\n"
    "  and the session lands in CONSTRUCT_FAILED — recover with another\n"
    "  `restart_simulation` call.\n"
    "\n"
    "== Hot-reloading user code ==\n"
    "  `restart_simulation(reload=True)` (the default) drops user-editable\n"
    "  modules from `sys.modules` before rebuilding the Simulator, so\n"
    "  source edits to schedulers (`sim/sched/<impl>/...`), hardware\n"
    "  (`sim/hw/<type>/<impl>/...`), and trace loaders (`sim/load/<impl>`)\n"
    "  are picked up without restarting the agent process.\n"
    "\n"
    "  Base classes in `sim/.../common/` are spared so isinstance checks\n"
    "  in framework code keep working. Don't edit base classes or core\n"
    "  framework code (`sim/core/...`) between runs — those changes will\n"
    "  not take effect.\n"
    "\n"
    "  Pass `reload=False` if you want to preserve the current class\n"
    "  identities (e.g. to compare two runs with the exact same code).\n"
    "\n"
    "== Environment knobs ==\n"
    "  CG_SIM_BREAKPOINTS=BREAK_X,BREAK_Y — comma-separated BREAK_* flag\n"
    "    names to pre-enable. Re-applied to each fresh Debugger built by\n"
    "    `restart_simulation`, so it's a persistent default across runs\n"
    "    in a single agent session."
)


_NO_BOUND_DEBUGGER = {
    "ok": False,
    "error": "No simulator is currently bound. Call `restart_simulation` first.",
}

# Time we'll wait for the very-first Simulator to finish constructing
# before giving up. Construction touches I/O (config + trace loading),
# so the FastMCP server can start serving before the main loop's first
# `_construct_and_bind` returns. Tools that need a bound Debugger wait
# briefly so the README's "typical session" works even when an agent
# fires `list_breakpoints` immediately after MCP `initialize`.
_BIND_WAIT_SECS = 60.0


_BOUND_PHASES = (Phase.READY, Phase.RUNNING, Phase.FINISHED)


def _bound_or_error(session: "AgentSession") -> dict[str, Any] | None:
    """Return None if a Debugger is bound (or becomes bound within
    `_BIND_WAIT_SECS`), else an error dict.

    Waits under `session.cv` for the phase to settle into one of the
    bound phases (READY/RUNNING/FINISHED). If construction fails
    (`CONSTRUCT_FAILED`) or the session shuts down before then, return
    the error dict — agents can interpret "no simulator bound" as
    "call restart_simulation with a valid input_path".
    """
    if session.debugger is not None:
        return None
    session.wait_until(
        lambda: session.phase in _BOUND_PHASES
                or session.phase in (Phase.CONSTRUCT_FAILED, Phase.SHUTDOWN),
        timeout=_BIND_WAIT_SECS,
    )
    if session.debugger is not None:
        return None
    return dict(_NO_BOUND_DEBUGGER)


# Subtrees of `sim.*` whose modules are dropped from sys.modules on a
# hot-reload. The LOAD_*_CLASS functions in `sim/core/init/` do live
# `importlib.import_module` lookups against these, so after invalidation
# the next call re-executes the package's `__init__.py` (which walks
# subpackages via pkgutil) and the user-edited leaf files. Base classes
# under `*.common.*` are *spared* — they must keep their identity
# because framework code (e.g. `sim/core/engine/engine.py`) holds them
# in `isinstance` checks and as superclasses.
_HOT_RELOAD_ROOTS = ("sim.sched", "sim.hw", "sim.load")


def _is_reloadable(module_name: str) -> bool:
    """True if `module_name` is in a user-editable subtree and not a base class."""
    in_root = any(
        module_name == r or module_name.startswith(r + ".")
        for r in _HOT_RELOAD_ROOTS
    )
    if not in_root:
        return False
    # Spare base classes / ABCs so isinstance checks stay valid across reload.
    return "common" not in module_name.split(".")


def _hot_reload_user_modules() -> list[str]:
    """Drop user-editable modules from `sys.modules`. Returns evicted names.

    Run from the FastMCP tool thread before signaling `request("restart")`
    so the main loop's subsequent `Simulator(...)` construction triggers
    fresh imports of any user-edited scheduler / hardware / loader code.
    """
    evicted: list[str] = []
    for name in list(sys.modules):
        if _is_reloadable(name):
            del sys.modules[name]
            evicted.append(name)
    importlib.invalidate_caches()
    return evicted


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
        err = _bound_or_error(session)
        if err is not None:
            return err
        # Atomically check we're in READY and transition to RUNNING.
        # The main loop is waiting on `phase != READY`; this transition
        # wakes it and it proceeds into `sim.run()`. The worker then
        # parks at the first enabled breakpoint and sets
        # `_state_changed_event`, which `_wait_for_state_change()` waits
        # on below.
        with session.cv:
            if session.phase != Phase.READY:
                return {"ok": False, "error": "Simulation already started."}
            dbg = session.debugger
            dbg._state_changed_event.clear()
            dbg._start_event.set()  # legacy marker; harmless to keep
            session.phase = Phase.RUNNING
            session.cv.notify_all()
        return dbg._wait_for_state_change()

    @server.tool(
        description=(
            "Return the current debugger state: whether the simulator is "
            "parked at a breakpoint, the breakpoint name, available "
            "variables and types, tip text, and whether the simulation "
            "has finished."
        )
    )
    def current_state() -> dict[str, Any]:
        err = _bound_or_error(session)
        if err is not None:
            return err
        return session.debugger.agent_current_state()

    @server.tool(
        description="Return a mapping of all BREAK_* flags to their On/Off status."
    )
    def list_breakpoints() -> dict[str, bool] | dict[str, Any]:
        err = _bound_or_error(session)
        if err is not None:
            return err
        return session.debugger.agent_list_breakpoints()

    @server.tool(
        description=(
            "Toggle a BREAK_* flag on the Debugger. Argument is the flag "
            "name (e.g. 'BREAK_AFTER_COMPILE_STAGE'). Returns the new value."
        )
    )
    def toggle_breakpoint(name: str) -> dict[str, Any]:
        err = _bound_or_error(session)
        if err is not None:
            return err
        return session.debugger.agent_toggle_breakpoint(name)

    @server.tool(
        description=(
            "Execute Python `code` against the parked breakpoint's "
            "namespace. The namespace persists across calls within a "
            "single breakpoint and exposes the variables advertised by "
            "`current_state` — always including `debug` (the Debugger) "
            "and `engine` (the Engine). A bare last expression is echoed "
            "in `output`, REPL-style. Useful Debugger members callable "
            "from here: `debug.record(dict_args)` writes a "
            "JSON-serializable dict to the simulation log under track "
            "Engine → Debug; `debug.args` is a free-form scratchpad dict "
            "for per-run notes; `debug.help()` re-prints the breakpoint "
            "context. Errors are returned as tracebacks rather than "
            "raised. Only callable when `at_breakpoint=True`. See the "
            "server's connection-time instructions for the full "
            "debugging surface."
        )
    )
    def execute(code: str) -> dict[str, Any]:
        err = _bound_or_error(session)
        if err is not None:
            return err
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
        err = _bound_or_error(session)
        if err is not None:
            return err
        return session.debugger.agent_continue()

    @server.tool(
        description=(
            "Tear down the just-finished simulator and build a fresh one. "
            "Only callable after `simulation_finished=True` (or before "
            "the first run).\n\n"
            "Parameters:\n"
            "  input_path — optional new YAML config for the next run.\n"
            "  overrides — optional list of Hydra-style override strings "
            "(same syntax as CLI overrides for `main.py`, e.g. "
            "`['scheduler.args.prefetch_window=8', "
            "'hardware.memory.0.args.memory_size_KB=10485760']`). "
            "Semantics: `None` (default) keeps the previous overrides "
            "(sticky across restarts; initialized from the CLI args "
            "given to `main_agent.py`); `[]` clears them; a list "
            "replaces. Invalid override strings cause construction to "
            "fail — the response reports the failure and the session "
            "lands in `CONSTRUCT_FAILED`; recover by calling "
            "`restart_simulation` again with corrected `overrides`.\n"
            "  reload — when True (default), drop user-editable modules "
            "(`sim.sched.*`, `sim.hw.*`, `sim.load.*`, except `*.common.*` "
            "base classes) from `sys.modules` so source edits to "
            "schedulers / hardware / loaders are picked up. Set False to "
            "preserve current class identities (e.g. for back-to-back "
            "runs of identical code).\n\n"
            "Blocks until the new Simulator is constructed and ready to "
            "accept breakpoint toggles. Response carries the same state "
            "shape as `current_state` (at_breakpoint=False, "
            "simulation_finished=False) plus `input_path`, `overrides` "
            "(the list applied to this construction), and, when "
            "reload was requested, `reloaded_modules` (count of evicted "
            "sys.modules entries)."
        )
    )
    def restart_simulation(
        input_path: str | None = None,
        overrides: list[str] | None = None,
        reload: bool = True,
    ) -> dict[str, Any]:
        # Restart is legal from READY ("before the first run"), FINISHED
        # (between runs), or CONSTRUCT_FAILED (recover from a bad path).
        # If we're stuck in RUNNING, refuse — the agent must drive the
        # current run to completion first.
        ok_phases = (Phase.READY, Phase.FINISHED, Phase.CONSTRUCT_FAILED)
        if not session.wait_until(
                lambda: session.phase in ok_phases
                        or session.phase == Phase.SHUTDOWN,
                timeout=_BIND_WAIT_SECS):
            return {
                "ok": False,
                "error": (
                    "Current simulation is not finished. Drive it to "
                    "completion via `continue_simulation` first, or call "
                    "`shutdown` to abort."
                ),
            }

        # Atomically: snapshot args, evict modules, transition.
        # Doing all three under the cv prevents a concurrent restart
        # from clobbering our input_path between the read and the
        # transition, and prevents the lost-edge race a bare-Event
        # protocol had.
        reloaded_count = 0
        with session.cv:
            if session.phase == Phase.SHUTDOWN:
                return {
                    "ok": False,
                    "error": "Session has been shut down.",
                }
            if session.phase not in ok_phases:
                # Raced with another transition; reject.
                return {
                    "ok": False,
                    "error": (
                        "Current simulation is not finished. Drive it to "
                        "completion via `continue_simulation` first, or "
                        "call `shutdown` to abort."
                    ),
                }
            if input_path:
                session.next_input_path = input_path
            # None = sticky (keep previous). Pass `[]` to clear. Anything
            # truthy (including `[]` after this branch — see below) replaces.
            if overrides is not None:
                session.next_overrides = list(overrides)
            if reload:
                reloaded_count = len(_hot_reload_user_modules())
            session.debugger = None
            session.phase = Phase.CONSTRUCTING
            session.cv.notify_all()

        # Wait for the next construction to settle.
        if not session.wait_until(
                lambda: session.phase in (Phase.READY,
                                          Phase.CONSTRUCT_FAILED,
                                          Phase.SHUTDOWN),
                timeout=120.0):
            return {
                "ok": False,
                "error": "Simulator construction timed out.",
                "timed_out": True,
            }
        if session.phase == Phase.CONSTRUCT_FAILED:
            return {
                "ok": False,
                "error": "Simulator construction failed; see process stderr.",
                "input_path": session.next_input_path,
                "overrides": list(session.next_overrides),
            }
        if session.phase == Phase.SHUTDOWN:
            return {
                "ok": False,
                "error": "Session has been shut down.",
            }
        state = session.debugger.agent_current_state()
        return {
            "ok": True,
            "input_path": session.next_input_path,
            "overrides": list(session.next_overrides),
            "reloaded_modules": reloaded_count,
            **state,
        }

    @server.tool(
        description=(
            "End the agent session. The main_agent.py process exits after "
            "this returns. Call after `simulation_finished=True` for a "
            "clean shutdown."
        )
    )
    def shutdown() -> dict[str, Any]:
        # Atomically: release any parked worker, signal shutdown.
        # SHUTDOWN takes priority — even if a concurrent restart raced
        # in, the next loop iteration will observe SHUTDOWN and exit.
        # Main loop's RUNNING-phase exit also checks for SHUTDOWN
        # before transitioning to FINISHED, so a shutdown during a
        # live run still terminates cleanly after sim.run() returns.
        with session.cv:
            dbg = session.debugger
            if dbg is not None and dbg._at_breakpoint:
                dbg._continue_event.set()
            session.phase = Phase.SHUTDOWN
            session.cv.notify_all()
        return {"ok": True}

    return server
