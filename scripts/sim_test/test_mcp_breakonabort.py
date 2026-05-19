"""End-to-end test for `BREAK_ON_ABORT` and the centralized
`break_on_abort` breakpoint.

Verifies:
  1. `_SERVER_INSTRUCTIONS` advertises BREAK_ON_ABORT and `break_on_abort`.
  2. BREAK_ON_ABORT defaults On.
  3. A forced deadlock (TransferJob.is_runnable patched to return False)
     fires `break_on_abort` from `Engine._log_abort`.
  4. At the breakpoint, `abort_args` is bound, contains `msg` and `from`.
  5. The variables list advertises `abort_args`.
  6. Continuing from `break_on_abort` lets the abort proceed and the run
     reaches `simulation_finished=true`.
  7. Toggling BREAK_ON_ABORT Off makes the same scenario fail-fast (no
     stop, run finishes directly).

Run from repo root:  python scripts/sim_test/test_mcp_breakonabort.py
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
INPUT_YAML = REPO_ROOT / "examples" / "run" / "llamacpp_llama-3-8B_flexinfer.yaml"


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


# Monkey-patches TransferJob.is_runnable to always return False — this
# triggers the LAYOUT-phase deadlock check (layout's only abort site for
# the FlexInfer example), which routes through Engine._log_abort and
# fires break_on_abort.
_FORCE_DEADLOCK = (
    "from sim.core.job.transfer_job import TransferJob;"
    " TransferJob.is_runnable = lambda self, sys: False"
)


async def _drive_to_finish(session: ClientSession, max_iters: int = 60) -> dict:
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

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()

            # 1. Advertisement.
            print("=== 1. advertisement ===", file=sys.stderr)
            instr = init.instructions or ""
            _check("BREAK_ON_ABORT" in instr,
                   "_SERVER_INSTRUCTIONS mentions BREAK_ON_ABORT")
            _check("break_on_abort" in instr,
                   "_SERVER_INSTRUCTIONS mentions break_on_abort")
            _check("_log_abort" in instr,
                   "_SERVER_INSTRUCTIONS mentions centralized _log_abort")

            # 2. Defaults On.
            print("\n=== 2. BREAK_ON_ABORT defaults On ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("list_breakpoints", {}))
            _check(r.get("BREAK_ON_ABORT") is True,
                   f"BREAK_ON_ABORT default On (got {r.get('BREAK_ON_ABORT')})")

            # 3. Force a deadlock; expect break_on_abort to fire.
            print("\n=== 3. forced deadlock fires break_on_abort ===",
                  file=sys.stderr)
            await session.call_tool("start_simulation", {})  # at before_compile
            r = _unwrap(await session.call_tool(
                "execute", {"code": _FORCE_DEADLOCK}))
            _check(r["ok"], f"monkey-patched is_runnable (got {r})")
            r = _unwrap(await session.call_tool("continue_simulation", {}))
            _check(r.get("breakpoint") == "break_on_abort",
                   f"fired break_on_abort (got {r.get('breakpoint')})")

            # 4. abort_args is bound with msg + from.
            print("\n=== 4. abort_args is bound ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("execute", {"code":
                "(abort_args.get('msg'), abort_args.get('from'))"}))
            _check("Deadlock" in r["output"] and "Engine" in r["output"],
                   f"abort_args has msg and from (got {r['output']!r})")

            # 5. Variables list advertises abort_args + abort_stack.
            print("\n=== 5. variables list ===", file=sys.stderr)
            state = _unwrap(await session.call_tool("current_state", {}))
            var_names = {v["name"] for v in state["variables"]}
            _check("abort_args" in var_names,
                   f"abort_args in variables (got {sorted(var_names)})")
            _check("abort_stack" in var_names,
                   f"abort_stack in variables (got {sorted(var_names)})")
            for needed in ("debug", "engine", "timestamp_now",
                           "job_waiting", "job_running", "hw", "trace"):
                _check(needed in var_names,
                       f"variables include {needed!r}")

            # 5b. abort_stack navigation: the agent can reach the actual
            #     decision-making frame and read its locals.
            print("\n=== 5b. abort_stack navigation ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("execute", {"code":
                "abort_stack[0].function"}))
            _check(r["output"].strip() == "'_log_abort'",
                   f"abort_stack[0] is _log_abort (got {r['output']!r})")
            # Find the deadlock-detection frame (Engine._layout for our
            # forced-deadlock scenario).
            r = _unwrap(await session.call_tool("execute", {"code":
                "next((i, f.function) for i, f in enumerate(abort_stack) "
                "if f.function == '_layout')"}))
            _check("_layout" in r["output"],
                   f"can locate _layout frame (got {r['output']!r})")
            # Read locals from that frame — should include `job_w` (the
            # stuck TransferJob at the head of job_waiting) and `args`
            # (same dict that became abort_args).
            r = _unwrap(await session.call_tool("execute", {"code":
                "idx = next(i for i, f in enumerate(abort_stack) "
                "if f.function == '_layout');"
                "frame_locals = abort_stack[idx].frame.f_locals;"
                "sorted(k for k in frame_locals if not k.startswith('_'))"}))
            _check("'args'" in r["output"] and "'job_w'" in r["output"],
                   f"_layout frame exposes its locals (got {r['output']!r})")
            # Verify they're the actual values.
            r = _unwrap(await session.call_tool("execute", {"code":
                "idx = next(i for i, f in enumerate(abort_stack) "
                "if f.function == '_layout');"
                "abort_stack[idx].frame.f_locals['args'] is abort_args"}))
            _check(r["output"].strip() == "True",
                   f"_layout's `args` is the same object as abort_args (got {r['output']!r})")

            # 6. Continue lets the abort proceed and the run ends.
            print("\n=== 6. continue completes the abort flow ===",
                  file=sys.stderr)
            r = await _drive_to_finish(session)
            _check(r["simulation_finished"],
                   f"reached simulation_finished after abort (got {r})")

            # 7. With BREAK_ON_ABORT off, same scenario should fail-fast.
            print("\n=== 7. BREAK_ON_ABORT=False: fail-fast ===", file=sys.stderr)
            # Restart (reload=True clears the monkey-patch automatically).
            await session.call_tool("restart_simulation", {})
            r = _unwrap(await session.call_tool("list_breakpoints", {}))
            _check(r["BREAK_ON_ABORT"] is True,
                   "BREAK_ON_ABORT default-On on fresh Debugger")
            r = _unwrap(await session.call_tool(
                "toggle_breakpoint", {"name": "BREAK_ON_ABORT"}))
            _check(r["new_value"] is False,
                   f"toggled BREAK_ON_ABORT off (got {r})")
            await session.call_tool("start_simulation", {})
            await session.call_tool("execute", {"code": _FORCE_DEADLOCK})
            r = await _drive_to_finish(session)
            _check(r["simulation_finished"] and r["breakpoint"] is None,
                   f"fail-fast: never stopped on abort (got {r})")

            await session.call_tool("shutdown", {})

    print("\nALL BREAK_ON_ABORT CHECKS PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
