"""End-to-end test for the `debug.break_lambda` custom-predicate breakpoint.

Verifies:
  1. `_SERVER_INSTRUCTIONS` advertises break_lambda.
  2. A predicate that returns True fires `break_in_runtime_stage[LAMBDA]`.
  3. At the LAMBDA breakpoint, the variables list omits `job` (none
     triggered it) but includes the other runtime locals.
  4. `engine.timestamp_now` and other runtime state are accessible in
     the LAMBDA-breakpoint namespace.
  5. Setting `debug.break_lambda = None` lets the run finish.
  6. A predicate returning a truthy-but-not-`True` value (e.g. `1`) does
     NOT fire the breakpoint — strict-`True` check.
  7. A predicate that raises auto-clears `debug.break_lambda` and the
     run continues without further interruption.

Run from repo root:  python scripts/sim_test/test_mcp_breaklambda.py
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


async def _drive_to_finish(session: ClientSession, max_iters: int = 60) -> dict:
    """Continue until simulation_finished. Returns final state."""
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

            # 1. Instructions advertise break_lambda.
            print("=== 1. _SERVER_INSTRUCTIONS coverage ===", file=sys.stderr)
            _check("break_lambda" in (init.instructions or ""),
                   "instructions mention break_lambda")
            _check("LAMBDA" in (init.instructions or ""),
                   "instructions mention LAMBDA breakpoint name")

            # 2. Set a predicate that returns True on the first tick.
            print("\n=== 2. lambda returning True fires LAMBDA breakpoint ===",
                  file=sys.stderr)
            r = _unwrap(await session.call_tool("start_simulation", {}))
            _check(r["breakpoint"] == "break_before_compile_stage",
                   "stopped at before_compile")

            r = _unwrap(await session.call_tool("execute", {"code":
                "debug.break_lambda = lambda engine, sys: engine.timestamp_now > 0"
            }))
            _check(r["ok"], f"set break_lambda (got {r})")

            r = _unwrap(await session.call_tool("continue_simulation", {}))
            _check(r.get("at_breakpoint") and
                   r["breakpoint"] == "break_in_runtime_stage[LAMBDA]",
                   f"fired LAMBDA breakpoint (got {r['breakpoint']})")

            # 3. variables list at LAMBDA omits `job`, includes the rest.
            print("\n=== 3. LAMBDA variables list ===", file=sys.stderr)
            var_names = {v["name"] for v in r["variables"]}
            _check("job" not in var_names,
                   f"no `job` in LAMBDA variables (got {sorted(var_names)})")
            for needed in ("debug", "engine", "timestamp_now",
                           "job_waiting", "job_running", "hw", "trace"):
                _check(needed in var_names,
                       f"LAMBDA variables include {needed!r}")

            # 4. Namespace is accessible.
            print("\n=== 4. namespace accessible at LAMBDA ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("execute", {"code":
                "(engine.timestamp_now, len(job_waiting), len(job_running))"
            }))
            _check(r["ok"] and "," in r["output"],
                   f"can read engine.timestamp_now and queue lengths (got {r['output']!r})")

            # 5. Disable and drive to finish.
            print("\n=== 5. disable lambda and finish ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("execute", {"code":
                "debug.break_lambda = None"}))
            _check(r["ok"], "cleared break_lambda")
            r = await _drive_to_finish(session)
            _check(r["simulation_finished"], "ran to completion after disabling")

            # 6. Truthy-but-not-True does NOT fire.
            print("\n=== 6. strict-True check: returning 1 does not fire ===",
                  file=sys.stderr)
            await session.call_tool("restart_simulation", {})
            r = _unwrap(await session.call_tool("start_simulation", {}))
            _check(r["breakpoint"] == "break_before_compile_stage",
                   "stopped at before_compile after restart")
            await session.call_tool("execute", {"code":
                "debug.break_lambda = lambda engine, sys: 1"})  # truthy, not True
            r = await _drive_to_finish(session)
            _check(r["simulation_finished"],
                   "truthy-but-not-True predicate never fired LAMBDA breakpoint")

            # 7. Buggy lambda auto-clears; run continues.
            print("\n=== 7. buggy lambda auto-clears ===", file=sys.stderr)
            await session.call_tool("restart_simulation", {})
            r = _unwrap(await session.call_tool("start_simulation", {}))
            _check(r["breakpoint"] == "break_before_compile_stage",
                   "stopped at before_compile")
            # Enable AFTER_RUNTIME so we can inspect post-run state.
            await session.call_tool("toggle_breakpoint",
                                    {"name": "BREAK_AFTER_RUNTIME_STAGE"})
            await session.call_tool("execute", {"code":
                "debug.break_lambda = lambda engine, sys: 1/0"})
            # Continue: runtime starts, first tick raises, lambda cleared,
            # run finishes, AFTER_RUNTIME_STAGE breakpoint fires.
            r = _unwrap(await session.call_tool("continue_simulation", {}))
            if r.get("timed_out"):
                r = _unwrap(await session.call_tool("current_state", {}))
            _check(r["breakpoint"] == "break_after_runtime_stage",
                   f"reached after_runtime_stage (got {r.get('breakpoint')})")
            # `debug.break_lambda` would print '' as a bare expression
            # (REPL suppresses None) — phrase the check as a boolean.
            r = _unwrap(await session.call_tool("execute",
                {"code": "debug.break_lambda is None"}))
            _check(r["output"].strip() == "True",
                   f"buggy lambda was auto-cleared (got {r['output']!r})")
            await _drive_to_finish(session)

            await session.call_tool("shutdown", {})

    print("\nALL break_lambda CHECKS PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
