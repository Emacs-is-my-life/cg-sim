"""End-to-end test for `BREAK_ON_EXCEPTION` and `break_on_exception`.

Verifies:
  1. `_SERVER_INSTRUCTIONS` advertises BREAK_ON_EXCEPTION + exception_origin.
  2. BREAK_ON_EXCEPTION defaults On.
  3. A forced AttributeError (engine.sched=None → engine._compile calls
     self.sched.compile → AttributeError) fires
     `break_on_exception[AttributeError]`.
  4. `exception` is bound and `type(exception).__name__ == 'AttributeError'`.
  5. `exception_origin` points to engine.py / `_compile`.
  6. `exception_stack` walks the traceback and `[-1].tb_frame.f_locals`
     contains the failing frame's locals (e.g., `self` for Engine).
  7. Variables list advertises `exception`, `exception_origin`,
     `exception_stack`.
  8. Continuing → simulation_finished=True (graceful end, no re-raise).
  9. Log file (read out-of-band after shutdown) contains a
     `SIMULATION_EXCEPTION` event with type and traceback.
 10. With BREAK_ON_EXCEPTION toggled off, same scenario fails-fast (no
     breakpoint, run ends with simulation_finished=True).

Run from repo root:  python scripts/sim_test/test_mcp_breakonexception.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_YAML = REPO_ROOT / "examples" / "llama3-flexinfer" / "input.yaml"


def _unwrap(result) -> dict:
    for c in result.content:
        if getattr(c, "type", None) == "text":
            return json.loads(c.text)
    raise RuntimeError(f"No text content: {result}")


def _check(condition: bool, label: str) -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}", file=sys.stderr)
    if not condition:
        raise SystemExit(f"FAIL: {label}")


# Force an AttributeError during compile-stage: by nulling engine.sched,
# `engine._compile` will call `None.compile(self.sys.trace)` and raise.
# The exception propagates out of engine.run() and is caught by
# Simulator.run's hard-failure handler.
_FORCE_EXCEPTION = "engine.sched = None"


async def _drive_to_finish(session: ClientSession, max_iters: int = 30) -> dict:
    for _ in range(max_iters):
        r = _unwrap(await session.call_tool("continue_simulation", {}))
        if r.get("simulation_finished"):
            return r
        if r.get("timed_out"):
            r = _unwrap(await session.call_tool("current_state", {}))
            if r.get("simulation_finished"):
                return r
    raise SystemExit("FAIL: never reached simulation_finished")


async def main() -> int:
    env = {**os.environ, "CG_SIM_BREAKPOINTS": "BREAK_BEFORE_COMPILE_STAGE"}
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "main_agent.py"), "-i", str(INPUT_YAML)],
        env=env, cwd=str(REPO_ROOT),
    )

    log_path = None

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()

            # 1. Advertisement.
            print("=== 1. advertisement ===", file=sys.stderr)
            instr = init.instructions or ""
            _check("BREAK_ON_EXCEPTION" in instr,
                   "_SERVER_INSTRUCTIONS mentions BREAK_ON_EXCEPTION")
            _check("break_on_exception" in instr,
                   "_SERVER_INSTRUCTIONS mentions break_on_exception")
            _check("exception_origin" in instr,
                   "_SERVER_INSTRUCTIONS mentions exception_origin")
            _check("SIMULATION_EXCEPTION" in instr,
                   "_SERVER_INSTRUCTIONS mentions SIMULATION_EXCEPTION log entry")

            # 2. Defaults On.
            print("\n=== 2. BREAK_ON_EXCEPTION defaults On ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("list_breakpoints", {}))
            _check(r.get("BREAK_ON_EXCEPTION") is True,
                   f"BREAK_ON_EXCEPTION default On (got {r.get('BREAK_ON_EXCEPTION')})")

            # 3. Force an exception; expect break_on_exception[AttributeError].
            print("\n=== 3. forced AttributeError fires break_on_exception ===",
                  file=sys.stderr)
            await session.call_tool("start_simulation", {})  # at before_compile
            # Grab the log path for out-of-band check later.
            r = _unwrap(await session.call_tool(
                "execute", {"code": "print(debug.log_path)"}))
            log_path = r["output"].strip()
            print(f"  log_path: {log_path}", file=sys.stderr)
            # Null the scheduler.
            r = _unwrap(await session.call_tool(
                "execute", {"code": _FORCE_EXCEPTION}))
            _check(r["ok"], f"set engine.sched=None (got {r})")
            r = _unwrap(await session.call_tool("continue_simulation", {}))
            _check(r.get("breakpoint") == "break_on_exception[AttributeError]",
                   f"fired break_on_exception[AttributeError] (got {r.get('breakpoint')})")

            # 4. exception bound, correct type.
            print("\n=== 4. exception object ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("execute", {"code":
                "type(exception).__name__"}))
            _check("AttributeError" in r["output"],
                   f"exception is AttributeError (got {r['output']!r})")

            # 5. exception_origin points to engine.py / _compile.
            print("\n=== 5. exception_origin ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("execute", {"code":
                "(exception_origin['function'], "
                "'engine.py' in exception_origin['file'], "
                "exception_origin['line'] > 0)"}))
            _check("_compile" in r["output"] and "True" in r["output"],
                   f"exception_origin points to engine.py:_compile (got {r['output']!r})")

            # 6. exception_stack navigation.
            print("\n=== 6. exception_stack navigation ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("execute", {"code":
                "[(f.tb_frame.f_code.co_name, f.tb_lineno) for f in exception_stack]"}))
            _check("_compile" in r["output"],
                   f"exception_stack includes _compile frame (got {r['output']!r})")
            # Inspect the failing frame's locals.
            r = _unwrap(await session.call_tool("execute", {"code":
                "'self' in dict(exception_stack[-1].tb_frame.f_locals)"}))
            _check(r["output"].strip() == "True",
                   f"failing frame exposes its locals (got {r['output']!r})")
            # And `self` is the Engine.
            r = _unwrap(await session.call_tool("execute", {"code":
                "type(exception_stack[-1].tb_frame.f_locals['self']).__name__"}))
            _check("Engine" in r["output"],
                   f"failing frame's `self` is the Engine (got {r['output']!r})")

            # 7. Variables list.
            print("\n=== 7. variables list ===", file=sys.stderr)
            state = _unwrap(await session.call_tool("current_state", {}))
            var_names = {v["name"] for v in state["variables"]}
            for needed in ("debug", "engine", "exception",
                           "exception_origin", "exception_stack"):
                _check(needed in var_names,
                       f"variables include {needed!r}")

            # 8. Continue → graceful finish.
            print("\n=== 8. graceful finish ===", file=sys.stderr)
            r = await _drive_to_finish(session)
            _check(r["simulation_finished"],
                   f"reached simulation_finished after exception (got {r})")

            # 10. Toggle off → fail-fast.
            print("\n=== 10. BREAK_ON_EXCEPTION=False: fail-fast ===",
                  file=sys.stderr)
            await session.call_tool("restart_simulation", {})
            r = _unwrap(await session.call_tool(
                "toggle_breakpoint", {"name": "BREAK_ON_EXCEPTION"}))
            _check(r["new_value"] is False,
                   f"toggled BREAK_ON_EXCEPTION off (got {r})")
            await session.call_tool("start_simulation", {})
            await session.call_tool("execute", {"code": _FORCE_EXCEPTION})
            r = await _drive_to_finish(session)
            _check(r["simulation_finished"] and r["breakpoint"] is None,
                   f"fail-fast: never stopped on exception (got {r})")

            await session.call_tool("shutdown", {})

    # 9. Out-of-band: verify SIMULATION_EXCEPTION log entry was written.
    # (Done after shutdown so the writer thread has finalized the JSON.)
    print("\n=== 9. log entry written ===", file=sys.stderr)
    trace = json.loads(open(log_path).read())
    exc_events = [
        e for e in trace["traceEvents"]
        if e.get("name") == "SIMULATION_EXCEPTION"
    ]
    _check(len(exc_events) >= 1,
           f"at least one SIMULATION_EXCEPTION event (got {len(exc_events)})")
    if exc_events:
        a = exc_events[0]["args"]
        _check(a.get("exception_type") == "AttributeError",
               f"log entry has exception_type=AttributeError (got {a.get('exception_type')!r})")
        _check(isinstance(a.get("traceback"), list) and a["traceback"],
               f"log entry has traceback list (got {type(a.get('traceback')).__name__})")

    print("\nALL BREAK_ON_EXCEPTION CHECKS PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
