"""Adversarial test of the cg-sim MCP debugging surface.

Each experiment exercises a fragile combination — coupling between
flags, edge cases in `execute`, error paths in the lifecycle tools.
Experiments are state-isolated: every one starts already parked at
`break_before_compile_stage` (pre-enabled via CG_SIM_BREAKPOINTS), and
every one is responsible for driving the run to `simulation_finished`
before returning. The harness then calls `restart_simulation` to set
up the next experiment.

Run from repo root:  python scripts/test_mcp_fragile.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_YAML = REPO_ROOT / "examples" / "llama3-flexinfer" / "input.yaml"


# ---------- protocol helpers ----------

def _unwrap(result) -> dict:
    for c in result.content:
        if getattr(c, "type", None) == "text":
            return json.loads(c.text)
    raise RuntimeError(f"No text content: {result}")


# Accumulate pass/fail per experiment so one bad combo doesn't tank the rest.
RESULTS: list[tuple[str, bool, str]] = []


def _record(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""),
          file=sys.stderr)
    RESULTS.append((label, ok, detail))


async def _call(session, tool: str, args: dict | None = None) -> dict:
    return _unwrap(await session.call_tool(tool, args or {}))


async def _drive_to_finish(session, max_iters: int = 80) -> dict:
    """Continue until simulation_finished. Disarms any custom breakpoints
    so a buggy test can't pin the worker at an unexpected stop."""
    for _ in range(max_iters):
        r = await _call(session, "continue_simulation")
        if r.get("simulation_finished"):
            return r
        if r.get("timed_out"):
            r = await _call(session, "current_state")
            if r.get("simulation_finished"):
                return r
        # Defensive: if we got parked at an unexpected breakpoint, disarm
        # anything we can to break out.
        if r.get("at_breakpoint"):
            # Best-effort cleanup; ignore errors.
            try:
                await _call(session, "execute", {"code":
                    "debug.break_lambda = None"})
                await _call(session, "execute", {"code":
                    "for n in trace.node_map.values():\n"
                    "    n.BREAK_AT_JOB_SUBMITTED = False\n"
                    "    n.BREAK_AT_JOB_HEAD = False\n"
                    "    n.BREAK_AT_JOB_DISPATCHED = False\n"
                    "    n.BREAK_AT_JOB_RETIRED = False"})
            except Exception:
                pass
    raise SystemExit("drive_to_finish exceeded max_iters")


# ---------- experiments ----------
# Pre-condition: each experiment is invoked while parked at
# break_before_compile_stage. Post-condition: each MUST drive the sim
# to simulation_finished before returning.


async def e1_per_node_without_master(session) -> None:
    """Arm BREAK_AT_JOB_RETIRED on every node WITHOUT enabling
    BREAK_IN_RUNTIME_STAGE. Per engine.py the master switch gates every
    per-Node check, so the run must finish without a runtime stop."""
    await _call(session, "toggle_breakpoint", {"name": "BREAK_AFTER_COMPILE_STAGE"})
    r = await _call(session, "continue_simulation")
    _record("E1 reached after_compile (sanity)",
            r.get("breakpoint") == "break_after_compile_stage",
            r.get("breakpoint"))
    r = await _call(session, "execute", {"code":
        "for n in trace.node_map.values(): n.BREAK_AT_JOB_RETIRED = True"})
    _record("E1 arm BREAK_AT_JOB_RETIRED on all nodes", r.get("ok"))
    flags = await _call(session, "list_breakpoints")
    _record("E1 master switch BREAK_IN_RUNTIME_STAGE is OFF",
            flags.get("BREAK_IN_RUNTIME_STAGE") is False)
    await _call(session, "toggle_breakpoint",
                {"name": "BREAK_AFTER_COMPILE_STAGE"})
    r = await _drive_to_finish(session)
    _record("E1 finished WITHOUT firing runtime breakpoint",
            r.get("simulation_finished") and r.get("breakpoint") is None,
            f"bp={r.get('breakpoint')}")


async def e2_all_four_flags_one_node(session) -> None:
    """Arm SUBMITTED+HEAD+DISPATCHED+RETIRED on one node; expect 4 stops
    in lifecycle order and verify `job.node.id` matches each time."""
    await _call(session, "toggle_breakpoint", {"name": "BREAK_AFTER_COMPILE_STAGE"})
    r = await _call(session, "continue_simulation")
    if r.get("breakpoint") != "break_after_compile_stage":
        _record("E2 setup", False, str(r.get("breakpoint")))
        return
    r = await _call(session, "execute", {"code":
        "from sim.core.trace import TerminalNode\n"
        "candidates = [n for n in trace.node_map.values() "
        "              if not isinstance(n, TerminalNode)]\n"
        "target = candidates[5]\n"
        "target.BREAK_AT_JOB_SUBMITTED = True\n"
        "target.BREAK_AT_JOB_HEAD = True\n"
        "target.BREAK_AT_JOB_DISPATCHED = True\n"
        "target.BREAK_AT_JOB_RETIRED = True\n"
        "print((target.id, target.name))"})
    target_id_name = r.get("output", "").strip()
    _record("E2 armed all 4 flags on a node", r.get("ok"),
            f"target={target_id_name}")
    await _call(session, "toggle_breakpoint", {"name": "BREAK_AFTER_COMPILE_STAGE"})
    await _call(session, "toggle_breakpoint", {"name": "BREAK_IN_RUNTIME_STAGE"})

    expected_phases = ["SUBMITTED", "HEAD", "DISPATCHED", "RETIRED"]
    seen_phases: list[str] = []
    for _ in range(80):
        r = await _call(session, "continue_simulation")
        if r.get("timed_out"):
            r = await _call(session, "current_state")
        if r.get("simulation_finished"):
            break
        bp = r.get("breakpoint", "") or ""
        if "break_in_runtime_stage[JOB_" in bp:
            phase = bp.split("[JOB_", 1)[1].rstrip("]")
            jr = await _call(session, "execute",
                             {"code": "print((job.node.id, job.node.name))"})
            if jr.get("output", "").strip() == target_id_name:
                if phase not in seen_phases:
                    seen_phases.append(phase)
        if len(seen_phases) >= 4:
            break

    _record(
        "E2 saw all 4 phases for the target node in order",
        seen_phases == expected_phases,
        f"saw={seen_phases}",
    )
    # Cleanup.
    await _call(session, "execute", {"code":
        "for n in trace.node_map.values():\n"
        "    n.BREAK_AT_JOB_SUBMITTED = False\n"
        "    n.BREAK_AT_JOB_HEAD = False\n"
        "    n.BREAK_AT_JOB_DISPATCHED = False\n"
        "    n.BREAK_AT_JOB_RETIRED = False"})
    await _call(session, "toggle_breakpoint", {"name": "BREAK_IN_RUNTIME_STAGE"})
    await _drive_to_finish(session)


async def e3_lambda_vs_per_node_coexist(session) -> None:
    """Both break_lambda (always-True) and per-Node BREAK_AT_JOB_RETIRED
    armed. Per engine.py: on the FIRST runtime tick no jobs are retired
    yet, so LAMBDA (evaluated unconditionally each tick) fires before
    any JOB_RETIRED. Verify both kinds fire and the run can finish
    after clearing them."""
    await _call(session, "toggle_breakpoint",
                {"name": "BREAK_AFTER_COMPILE_STAGE"})
    r = await _call(session, "continue_simulation")
    if r.get("breakpoint") != "break_after_compile_stage":
        _record("E3 setup", False, str(r))
        return
    await _call(session, "execute", {"code":
        "for n in trace.node_map.values(): n.BREAK_AT_JOB_RETIRED = True"})
    await _call(session, "execute", {"code":
        "debug.break_lambda = lambda engine, sys: True"})
    await _call(session, "toggle_breakpoint", {"name": "BREAK_IN_RUNTIME_STAGE"})
    await _call(session, "toggle_breakpoint",
                {"name": "BREAK_AFTER_COMPILE_STAGE"})

    seen: list[str] = []
    for _ in range(12):
        r = await _call(session, "continue_simulation")
        if r.get("timed_out"):
            r = await _call(session, "current_state")
        if r.get("simulation_finished"):
            break
        bp = r.get("breakpoint", "") or ""
        if "break_in_runtime_stage" in bp:
            seen.append(bp.split("[", 1)[1].rstrip("]"))
        if "LAMBDA" in seen and ("JOB_RETIRED" in seen
                                 or any("JOB_" in s for s in seen)):
            break

    _record("E3 LAMBDA fired",
            "LAMBDA" in seen, f"seen={seen}")
    _record("E3 some JOB_* fired alongside LAMBDA",
            any(s.startswith("JOB_") for s in seen),
            f"seen={seen}")
    _record("E3 LAMBDA observed BEFORE any JOB_ on first encounter",
            seen and seen[0] == "LAMBDA",
            f"first={seen[0] if seen else None}")
    # Cleanup.
    await _call(session, "execute", {"code":
        "debug.break_lambda = None\n"
        "for n in trace.node_map.values(): n.BREAK_AT_JOB_RETIRED = False"})
    await _call(session, "toggle_breakpoint", {"name": "BREAK_IN_RUNTIME_STAGE"})
    await _drive_to_finish(session)


async def e4_lambda_non_callable(session) -> None:
    """`debug.break_lambda = 42` — first invocation raises TypeError;
    `_break_lambda`'s except catches it and clears the lambda."""
    await _call(session, "execute", {"code": "debug.break_lambda = 42"})
    r = await _call(session, "execute",
                    {"code": "print(debug.break_lambda)"})
    _record("E4 non-callable assigned to break_lambda",
            "42" in r.get("output", ""),
            r.get("output", "").strip())
    r = await _drive_to_finish(session)
    _record("E4 run completed after non-callable lambda (auto-cleared)",
            r.get("simulation_finished"))


async def e5_lambda_raises_later_tick(session) -> None:
    """Lambda returns False for first few ticks then raises. The
    auto-clear logic must catch the late raise (not just first-call)."""
    # Define via exec mode (multi-line block, no semicolons).
    await _call(session, "execute", {"code":
        "_state = [0]\n"
        "def _f(engine, sys):\n"
        "    _state[0] += 1\n"
        "    if _state[0] > 3:\n"
        "        raise RuntimeError('boom on tick %d' % _state[0])\n"
        "    return False\n"
        "debug.break_lambda = _f"})
    r = await _call(session, "execute",
                    {"code": "print(callable(debug.break_lambda))"})
    _record("E5 lambda installed as callable",
            "True" in r.get("output", ""),
            r.get("output", "").strip())
    r = await _drive_to_finish(session)
    _record("E5 run completed despite lambda raising mid-run",
            r.get("simulation_finished"))


async def e7_toggle_invalid_flag(session) -> None:
    """toggle_breakpoint with garbage name returns a clean error dict."""
    r = await _call(session, "toggle_breakpoint", {"name": "BREAK_BANANA"})
    _record("E7 toggling invalid flag returns ok=False",
            r.get("ok") is False and "error" in r,
            f"r={r}")
    r = await _call(session, "toggle_breakpoint", {"name": ""})
    _record("E7 toggling empty-string flag returns ok=False",
            r.get("ok") is False,
            f"r={r}")
    r = await _call(session, "toggle_breakpoint", {"name": "args"})  # exists, not bool
    _record("E7 toggling non-bool attribute name returns ok=False",
            r.get("ok") is False,
            f"r={r}")
    flags = await _call(session, "list_breakpoints")
    _record("E7 list_breakpoints still works after invalid toggles",
            isinstance(flags, dict) and "BREAK_IN_RUNTIME_STAGE" in flags)
    await _drive_to_finish(session)


async def e8_start_twice_and_state(session) -> None:
    """Pre-condition gives us a parked sim. start_simulation again is
    an error. current_state still works while parked."""
    r = await _call(session, "current_state")
    _record("E8 current_state while parked reads the breakpoint",
            r.get("at_breakpoint") and
            r.get("breakpoint") == "break_before_compile_stage",
            f"bp={r.get('breakpoint')}")
    r = await _call(session, "start_simulation")
    _record("E8 start_simulation a second time returns ok=False",
            r.get("ok") is False,
            f"err={r.get('error')!r}")
    # Drive to finish.
    await _drive_to_finish(session)
    # After finish, start_simulation should still error.
    r = await _call(session, "start_simulation")
    _record("E8 start_simulation after finish also returns ok=False",
            r.get("ok") is False,
            f"err={r.get('error')!r}")


async def e9_continue_and_execute_after_finish(session) -> None:
    """continue_simulation and execute after simulation_finished both error."""
    await _drive_to_finish(session)
    r = await _call(session, "continue_simulation")
    _record("E9 continue after finish returns ok=False (no hang)",
            r.get("ok") is False,
            f"err={r.get('error')!r}")
    r = await _call(session, "execute", {"code": "1+1"})
    _record("E9 execute after finish returns ok=False",
            r.get("ok") is False,
            f"err={r.get('error')!r}")


async def e11_restart_before_finish(session) -> None:
    """We are parked at break_before_compile_stage — sim not finished.
    restart_simulation must reject this; the sim must remain alive."""
    r = await _call(session, "restart_simulation", {})
    _record("E11 restart before finish returns ok=False",
            r.get("ok") is False,
            f"err={(r.get('error') or '')[:80]!r}")
    r = await _call(session, "current_state")
    _record(
        "E11 simulator still alive after rejected restart",
        r.get("at_breakpoint")
        and r.get("breakpoint") == "break_before_compile_stage",
        f"bp={r.get('breakpoint')}",
    )
    await _drive_to_finish(session)


async def e13_record_non_serializable(session) -> None:
    """debug.record({non-serializable}) — graceful (prints, doesn't raise)."""
    r = await _call(session, "execute", {"code":
        "import threading\n"
        "debug.record({'lock': threading.Lock()})"})
    _record("E13 debug.record with non-serializable does not raise",
            r.get("ok") is True,
            f"out={r.get('output','').strip()[:80]!r}")
    _record(
        "E13 debug.record output mentions JSON failure",
        "JSON" in r.get("output", "")
        or "serializable" in r.get("output", ""),
        r.get("output", "").strip()[:80],
    )
    # Also verify a JSON-clean dict actually writes through.
    r = await _call(session, "execute", {"code":
        "debug.record({'ok-probe': 'present'})"})
    _record("E13 follow-up: valid debug.record after the rejected one",
            r.get("ok") is True and not (r.get("output") or "").strip(),
            f"out={r.get('output','').strip()[:80]!r}")
    await _drive_to_finish(session)


async def e14_execute_edge_cases(session) -> None:
    """execute(): empty, whitespace, comment, syntax error, runtime error
    in multi-stmt, namespace persistence through errors."""
    # Empty.
    r = await _call(session, "execute", {"code": ""})
    _record("E14 execute('') is ok=True", r.get("ok") is True, str(r))
    # Whitespace only.
    r = await _call(session, "execute", {"code": "   \n  \t"})
    _record("E14 execute(whitespace) is ok=True", r.get("ok") is True, str(r))
    # Pure comment.
    r = await _call(session, "execute", {"code": "# just a comment"})
    _record("E14 execute('# comment') is ok=True", r.get("ok") is True, str(r))

    # Syntax error.
    r = await _call(session, "execute", {"code": "def("})
    _record(
        "E14 execute(syntax error) returns ok=False with SyntaxError text",
        r.get("ok") is False and "SyntaxError" in (r.get("error") or ""),
        (r.get("error") or "").splitlines()[-1] if r.get("error") else "",
    )

    # Sim still alive at the breakpoint after a SyntaxError.
    r = await _call(session, "execute", {"code": "1+1"})
    _record("E14 namespace still works after syntax error",
            r.get("ok") and "2" in (r.get("output") or ""),
            r.get("output", "").strip())

    # Multi-stmt with mid-block raise — use exec mode (newlines, no `;`).
    r = await _call(session, "execute", {"code":
        "_marker_a = 'first'\n"
        "raise ValueError('mid-block')\n"
        "_marker_b = 'second'"})
    _record(
        "E14 multi-stmt with mid-block raise returns ok=False",
        r.get("ok") is False and "ValueError" in (r.get("error") or ""),
        (r.get("error") or "").splitlines()[-1] if r.get("error") else "",
    )
    r = await _call(session, "execute", {"code":
        "print((_marker_a, '_marker_b' in dir()))"})
    _record(
        "E14 partial side effect: _marker_a persists, _marker_b does not",
        r.get("ok") and "first" in (r.get("output") or "")
        and "False" in (r.get("output") or ""),
        r.get("output", "").strip(),
    )

    # Bare last expression (in `single` mode) is echoed REPL-style.
    r = await _call(session, "execute", {"code": "40 + 2"})
    _record("E14 bare expression echoed REPL-style",
            r.get("ok") and "42" in (r.get("output") or ""),
            r.get("output", "").strip())

    await _drive_to_finish(session)


async def e17_namespace_corruption_recovery(session) -> None:
    """Rebinding `debug=None` in the namespace breaks subsequent calls
    using `debug.*` — but the namespace is fresh at the next breakpoint."""
    await _call(session, "execute", {"code": "debug = None"})
    r = await _call(session, "execute", {"code": "debug.record({'x':1})"})
    _record(
        "E17 debug.record after `debug=None` raises AttributeError",
        r.get("ok") is False
        and "AttributeError" in (r.get("error") or ""),
        (r.get("error") or "").splitlines()[-1] if r.get("error") else "",
    )
    # Move to next stage breakpoint.
    await _call(session, "toggle_breakpoint",
                {"name": "BREAK_AFTER_COMPILE_STAGE"})
    r = await _call(session, "continue_simulation")
    _record("E17 reached next breakpoint after corruption",
            r.get("breakpoint") == "break_after_compile_stage",
            r.get("breakpoint"))
    r = await _call(session, "execute",
                    {"code": "print(type(debug).__name__)"})
    _record("E17 namespace is fresh at next breakpoint (debug rebound)",
            r.get("ok") and "Debugger" in (r.get("output") or ""),
            r.get("output", "").strip())
    await _call(session, "toggle_breakpoint",
                {"name": "BREAK_AFTER_COMPILE_STAGE"})
    await _drive_to_finish(session)


async def e18_abort_does_not_fire_exception(session) -> None:
    """An abort goes through Engine._log_abort; an uncaught exception goes
    through Simulator.run's handler. They are disjoint. With BREAK_ON_ABORT
    OFF and BREAK_ON_EXCEPTION ON, an aborting scenario must NOT fire
    break_on_exception (the abort never raised)."""
    flags = await _call(session, "list_breakpoints")
    _record("E18 BREAK_ON_ABORT default On",
            flags.get("BREAK_ON_ABORT") is True)
    _record("E18 BREAK_ON_EXCEPTION default On",
            flags.get("BREAK_ON_EXCEPTION") is True)
    r = await _call(session, "toggle_breakpoint", {"name": "BREAK_ON_ABORT"})
    _record("E18 toggled BREAK_ON_ABORT off",
            r.get("new_value") is False, str(r))
    # Force a deadlock in layout phase.
    await _call(session, "execute", {"code":
        "from sim.core.job.transfer_job import TransferJob\n"
        "TransferJob.is_runnable = lambda self, sys: False"})
    r = await _drive_to_finish(session)
    _record(
        "E18 abort with ABORT-off+EXCEPTION-on: no breakpoint fired",
        r.get("simulation_finished") and r.get("breakpoint") is None,
        f"bp={r.get('breakpoint')}",
    )


async def e19_debug_args_resets_across_restart(session) -> None:
    """debug.args is per-Debugger; each restart_simulation builds a new
    Debugger, so debug.args is empty at the start of every run."""
    # First run: stash a value.
    r = await _call(session, "execute", {"code":
        "debug.args['flag'] = 'set-in-run-A'\n"
        "print(debug.args)"})
    _record("E19 stash and read debug.args in run A",
            r.get("ok") and "set-in-run-A" in (r.get("output") or ""),
            r.get("output", "").strip())
    await _drive_to_finish(session)
    # Restart + start to begin run B.
    r = await _call(session, "restart_simulation", {})
    _record("E19 restart_simulation between runs",
            r.get("ok"),
            f"err={r.get('error')!r}")
    await _call(session, "start_simulation")
    r = await _call(session, "execute", {"code": "print(debug.args)"})
    _record(
        "E19 debug.args is reset on restart_simulation",
        r.get("ok") and r.get("output", "").strip() == "{}",
        r.get("output", "").strip(),
    )
    await _drive_to_finish(session)


# ---------- harness ----------

async def _safe_run(session, name: str, coro) -> None:
    """Run one experiment with proper setup/teardown isolation:

      1. Caller's pre-condition: sim was finished by the previous
         experiment (or this is the very first one — handled by main()).
      2. Restart + start so the experiment finds itself parked at
         `break_before_compile_stage`.
      3. Run the experiment.
      4. If the experiment didn't already drive to finish, do it now.

    Captures uncaught exceptions as a FAIL row.
    """
    print(f"\n=== {name} ===", file=sys.stderr)
    # Restart so the experiment gets a fresh Debugger / Engine / Trace.
    r = await _call(session, "restart_simulation", {})
    if not r.get("ok"):
        _record(f"{name} setup: restart_simulation",
                False, f"err={r.get('error')!r}")
        return
    r = await _call(session, "start_simulation")
    if r.get("breakpoint") != "break_before_compile_stage":
        _record(f"{name} setup: parked at break_before_compile_stage",
                False, f"bp={r.get('breakpoint')}")
        # Try to finish gracefully so the next experiment can start clean.
        await _drive_to_finish(session)
        return

    try:
        await coro(session)
    except BaseException as e:
        tb = traceback.format_exc()
        _record(f"{name} (uncaught exception)", False, str(e))
        print(tb, file=sys.stderr)
    # Ensure the sim is finished before the next experiment's restart.
    try:
        state = await _call(session, "current_state")
        if not state.get("simulation_finished"):
            await _drive_to_finish(session)
    except BaseException:
        pass


async def main() -> int:
    env = {**os.environ, "CG_SIM_BREAKPOINTS": "BREAK_BEFORE_COMPILE_STAGE"}
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "main_agent.py"), "-i", str(INPUT_YAML)],
        env=env, cwd=str(REPO_ROOT),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # First experiment needs special handling: the initial
            # Simulator was already constructed and is waiting at
            # _start_event.wait(). Calling restart_simulation right now
            # would race the initial-construction's `restart_complete.set()`.
            # Instead we just start the sim, drive it, then let _safe_run
            # take over with proper restart cycling.
            print("=== Bootstrap: drain initial run ===", file=sys.stderr)
            r = await _call(session, "start_simulation")
            _record(
                "bootstrap parks at break_before_compile_stage",
                r.get("breakpoint") == "break_before_compile_stage",
                r.get("breakpoint"),
            )
            await _drive_to_finish(session)

            await _safe_run(session, "E1: per-Node w/o master switch",
                            e1_per_node_without_master)
            await _safe_run(session, "E2: 4 per-Node flags on one node",
                            e2_all_four_flags_one_node)
            await _safe_run(session, "E3: break_lambda + per-Node coexist",
                            e3_lambda_vs_per_node_coexist)
            await _safe_run(session, "E4: non-callable break_lambda",
                            e4_lambda_non_callable)
            await _safe_run(session, "E5: lambda raising on later tick",
                            e5_lambda_raises_later_tick)
            await _safe_run(session, "E7: toggle invalid flag names",
                            e7_toggle_invalid_flag)
            await _safe_run(session, "E8: start_simulation twice + state",
                            e8_start_twice_and_state)
            await _safe_run(session, "E9: continue/execute after finish",
                            e9_continue_and_execute_after_finish)
            await _safe_run(session, "E11: restart before finish",
                            e11_restart_before_finish)
            await _safe_run(session, "E13: debug.record non-serializable",
                            e13_record_non_serializable)
            await _safe_run(session, "E14: execute edge cases",
                            e14_execute_edge_cases)
            await _safe_run(session, "E17: namespace corruption recovery",
                            e17_namespace_corruption_recovery)
            await _safe_run(session, "E18: abort doesn't fire exception bp",
                            e18_abort_does_not_fire_exception)
            await _safe_run(session, "E19: debug.args reset across restart",
                            e19_debug_args_resets_across_restart)

            await session.call_tool("shutdown", {})

    # E6: garbage in CG_SIM_BREAKPOINTS — fresh subprocess so env differs.
    print("\n=== E6: CG_SIM_BREAKPOINTS with garbage ===", file=sys.stderr)
    env6 = {
        **os.environ,
        "CG_SIM_BREAKPOINTS":
            "BREAK_NONSENSE, BREAK_AFTER_COMPILE_STAGE,, __dunder__,"
            "BREAK_AFTER_LAYOUT_STAGE"
    }
    server6 = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "main_agent.py"), "-i", str(INPUT_YAML)],
        env=env6, cwd=str(REPO_ROOT),
    )
    async with stdio_client(server6) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            flags = _unwrap(await session.call_tool("list_breakpoints", {}))
            _record(
                "E6 valid flags pre-enabled despite garbage entries",
                flags.get("BREAK_AFTER_COMPILE_STAGE") is True
                and flags.get("BREAK_AFTER_LAYOUT_STAGE") is True,
                f"compile={flags.get('BREAK_AFTER_COMPILE_STAGE')}, "
                f"layout={flags.get('BREAK_AFTER_LAYOUT_STAGE')}",
            )
            _record(
                "E6 garbage 'BREAK_NONSENSE' did not create a new flag",
                "BREAK_NONSENSE" not in flags,
            )
            # Make sure the server still works — start + drain.
            r = _unwrap(await session.call_tool("start_simulation", {}))
            _record(
                "E6 server still functional: start parks at after_compile",
                r.get("breakpoint") == "break_after_compile_stage",
                r.get("breakpoint"),
            )
            # Drain.
            for _ in range(40):
                r = _unwrap(await session.call_tool("continue_simulation", {}))
                if r.get("simulation_finished"):
                    break
            await session.call_tool("shutdown", {})

    # E12: bad input_path — fresh subprocess so a failed restart in the
    # middle of a session doesn't poison later experiments.
    print("\n=== E12: restart_simulation with bad input_path ===",
          file=sys.stderr)
    server12 = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "main_agent.py"), "-i", str(INPUT_YAML)],
        env={**os.environ, "CG_SIM_BREAKPOINTS": "BREAK_BEFORE_COMPILE_STAGE"},
        cwd=str(REPO_ROOT),
    )
    async with stdio_client(server12) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Finish the initial run first.
            _unwrap(await session.call_tool("start_simulation", {}))
            for _ in range(40):
                r = _unwrap(await session.call_tool("continue_simulation", {}))
                if r.get("simulation_finished"):
                    break
            # Now try the bad path.
            r = _unwrap(await session.call_tool("restart_simulation",
                {"input_path": "/tmp/no/such/path.yaml"}))
            _record(
                "E12 restart with bad input_path returns ok=False",
                r.get("ok") is False,
                f"ok={r.get('ok')}, err={(r.get('error') or '')[:120]!r}",
            )
            # After failure, no debugger should be bound — list_breakpoints
            # should error.
            r = _unwrap(await session.call_tool("list_breakpoints", {}))
            _record(
                "E12 after failed restart, no Debugger is bound",
                r.get("ok") is False
                and "No simulator" in (r.get("error") or ""),
                str(r)[:120],
            )
            # Recovery: restart with the good path.
            r = _unwrap(await session.call_tool("restart_simulation",
                {"input_path": str(INPUT_YAML)}))
            _record("E12 recovery: restart with good input_path",
                    r.get("ok") is True,
                    f"ok={r.get('ok')}, err={r.get('error')!r}")
            await session.call_tool("shutdown", {})

    # ---------- summary ----------
    print("\n" + "=" * 70, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [r for r in RESULTS if not r[1]]
    for label, ok, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {label}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line, file=sys.stderr)
    print(f"\n{passed}/{len(RESULTS)} checks passed; {len(failed)} failed",
          file=sys.stderr)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
