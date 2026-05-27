"""End-to-end smoke test for the cg-sim MCP debugging surface.

Spawns `main_agent.py` as a subprocess via the MCP stdio client, then
exercises every advertised debugging feature in sequence:

  1. Connection-time instructions are non-empty and mention the new
     features (engine var, debug.record, debug.help, BREAK_AT_JOB_*).
  2. `list_breakpoints` returns the 5 stage flags.
  3. `toggle_breakpoint` flips a flag.
  4. CG_SIM_BREAKPOINTS pre-enables a flag.
  5. `start_simulation` blocks and returns the new state with
     `variables` including `debug` and `engine`.
  6. `current_state` returns the same shape without releasing the worker.
  7. `execute` works for inspection, bare expressions echoed REPL-style.
  8. `execute("debug.record({...}}")` runs without error.
  9. `execute("debug.help()")` produces output captured in `output`.
 10. `execute("engine.sched")` confirms the new engine variable resolves.
 11. Per-Node BREAK_AT_JOB_RETIRED arms and fires.
 12. `continue_simulation` reaches simulation_finished=true.
 13. `restart_simulation` rebuilds the simulator.
 14. `shutdown` ends the process.

Run from repo root:  python scripts/sim_test/test_mcp_debugger.py
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
    """FastMCP returns tool results as CallToolResult with content list."""
    for c in result.content:
        if getattr(c, "type", None) == "text":
            return json.loads(c.text)
    raise RuntimeError(f"No text content: {result}")


def _check(condition: bool, label: str) -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}", file=sys.stderr)
    if not condition:
        raise SystemExit(f"FAIL: {label}")


async def main() -> int:
    env = {**os.environ, "CG_SIM_BREAKPOINTS": "BREAK_AFTER_COMPILE_STAGE"}
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "main_agent.py"), "-i", str(INPUT_YAML)],
        env=env,
        cwd=str(REPO_ROOT),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()

            # 1. Connection-time instructions advertise the new surface.
            instructions = init.instructions or ""
            print("=== 1. _SERVER_INSTRUCTIONS coverage ===", file=sys.stderr)
            for needle in [
                "engine",
                "debug.record",
                "debug.help",
                "BREAK_AT_JOB_SUBMITTED",
                "BREAK_AT_JOB_HEAD",
                "BREAK_AT_JOB_DISPATCHED",
                "BREAK_AT_JOB_RETIRED",
                "CG_SIM_BREAKPOINTS",
                "current_state",
                "restart_simulation",
            ]:
                _check(needle in instructions, f"instructions mention {needle!r}")

            # 2. list_breakpoints
            print("\n=== 2. list_breakpoints ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("list_breakpoints", {}))
            stage_flags = {
                "BREAK_BEFORE_COMPILE_STAGE", "BREAK_AFTER_COMPILE_STAGE",
                "BREAK_AFTER_LAYOUT_STAGE", "BREAK_IN_RUNTIME_STAGE",
                "BREAK_AFTER_RUNTIME_STAGE",
            }
            _check(stage_flags <= set(r), "all 5 stage flags present")
            _check(r["BREAK_AFTER_COMPILE_STAGE"] is True,
                   "CG_SIM_BREAKPOINTS pre-enabled BREAK_AFTER_COMPILE_STAGE")
            _check(r["BREAK_BEFORE_COMPILE_STAGE"] is False,
                   "other flags default Off")

            # 3. toggle_breakpoint
            print("\n=== 3. toggle_breakpoint ===", file=sys.stderr)
            r = _unwrap(await session.call_tool(
                "toggle_breakpoint", {"name": "BREAK_IN_RUNTIME_STAGE"}))
            _check(r["ok"] is True and r["new_value"] is True,
                   "toggling BREAK_IN_RUNTIME_STAGE -> True")

            # 4. start_simulation blocks until first breakpoint
            print("\n=== 4. start_simulation -> first breakpoint ===",
                  file=sys.stderr)
            r = _unwrap(await session.call_tool("start_simulation", {}))
            _check(r["at_breakpoint"] is True, "parked at a breakpoint")
            _check(r["breakpoint"] == "break_after_compile_stage",
                   f"first stop = break_after_compile_stage (got {r['breakpoint']})")
            var_names = {v["name"] for v in r["variables"]}
            _check("debug" in var_names, "variables include 'debug'")
            _check("engine" in var_names, "variables include 'engine'")
            _check("trace" in var_names, "variables include 'trace'")
            _check(r.get("tip") and "BREAK_AT_" in r["tip"],
                   "tip text present and mentions BREAK_AT_*")

            # 5. current_state cheap re-read
            print("\n=== 5. current_state ===", file=sys.stderr)
            r2 = _unwrap(await session.call_tool("current_state", {}))
            _check(r2["breakpoint"] == r["breakpoint"],
                   "current_state agrees with start_simulation state")
            _check(r2["at_breakpoint"] is True, "still parked")

            # 6. execute basic inspection (bare expression REPL echo)
            print("\n=== 6. execute (inspection) ===", file=sys.stderr)
            r = _unwrap(await session.call_tool(
                "execute", {"code": "len(trace.node_map)"}))
            _check(r["ok"] is True, "execute ok")
            _check(r["output"].strip().isdigit(),
                   f"bare expr echoed as number (got {r['output']!r})")

            # 7. execute(debug.record(...)) — advertised debugger method
            print("\n=== 7. execute -> debug.record(...) ===", file=sys.stderr)
            r = _unwrap(await session.call_tool(
                "execute",
                {"code": "debug.record({'probe': 'mcp-test', 'count': 42})"}))
            _check(r["ok"] is True and not r.get("error"),
                   f"debug.record accepted ok (output={r.get('output')!r}, err={r.get('error')!r})")

            # 8. execute(debug.help()) — advertised re-print
            print("\n=== 8. execute -> debug.help() ===", file=sys.stderr)
            r = _unwrap(await session.call_tool(
                "execute", {"code": "debug.help()"}))
            _check(r["ok"] is True, "debug.help() runs")
            _check("BREAKPOINT" in r["output"] and "engine" in r["output"],
                   f"debug.help output includes banner and 'engine' (got len={len(r['output'])})")

            # 9. execute(engine.sched) — advertised navigation through engine
            print("\n=== 9. execute -> engine + engine.sched ===", file=sys.stderr)
            r = _unwrap(await session.call_tool(
                "execute", {"code": "type(engine).__name__"}))
            _check("Engine" in r["output"],
                   f"engine variable resolves (output={r['output']!r})")
            r = _unwrap(await session.call_tool(
                "execute", {"code": "engine.sched.name"}))
            _check(r["ok"] is True and r["output"].strip(),
                   f"engine.sched.name resolves (output={r['output']!r})")

            # 10. Persistent namespace across execute calls
            print("\n=== 10. namespace persistence ===", file=sys.stderr)
            await session.call_tool("execute", {"code": "_probe_val = 12345"})
            r = _unwrap(await session.call_tool(
                "execute", {"code": "_probe_val"}))
            _check("12345" in r["output"],
                   f"namespace persists across calls (output={r['output']!r})")

            # 11. Arm a per-Node BREAK_AT_JOB_RETIRED, then continue
            print("\n=== 11. per-Node BREAK_AT_JOB_RETIRED ===",
                  file=sys.stderr)
            r = _unwrap(await session.call_tool("execute", {"code": (
                "target = next(iter(trace.node_map.values()));"
                " target.BREAK_AT_JOB_RETIRED = True;"
                " (target.id, target.name)"
            )}))
            _check(r["ok"] is True, f"arm BREAK_AT_JOB_RETIRED on first node (out={r.get('output')!r}, err={r.get('error')!r})")
            print(f"      armed on: {r['output'].strip()}", file=sys.stderr)

            # 12. continue -> should land on break_in_runtime_stage[JOB_RETIRED]
            print("\n=== 12. continue_simulation -> runtime breakpoint ===",
                  file=sys.stderr)
            r = _unwrap(await session.call_tool("continue_simulation", {}))
            if r.get("timed_out"):
                print("      (timed_out=true; re-reading state)",
                      file=sys.stderr)
                r = _unwrap(await session.call_tool("current_state", {}))
            _check(r["breakpoint"] == "break_in_runtime_stage[JOB_RETIRED]",
                   f"hit break_in_runtime_stage[JOB_RETIRED] (got {r['breakpoint']})")
            var_names = {v["name"] for v in r["variables"]}
            for needed in ["debug", "engine", "job", "timestamp_now",
                           "job_waiting", "job_running"]:
                _check(needed in var_names,
                       f"runtime variables include {needed!r}")

            # 13. Disarm and drive to completion
            print("\n=== 13. drive to simulation_finished ===", file=sys.stderr)
            await session.call_tool("execute", {"code": (
                "for n in trace.node_map.values():"
                " n.BREAK_AT_JOB_RETIRED = False"
            )})
            await session.call_tool("toggle_breakpoint", {
                "name": "BREAK_IN_RUNTIME_STAGE"})  # turn master off
            # Loop continue until done
            for i in range(20):
                r = _unwrap(await session.call_tool(
                    "continue_simulation", {}))
                if r.get("simulation_finished"):
                    break
                if r.get("timed_out"):
                    r = _unwrap(await session.call_tool(
                        "current_state", {}))
                    if r.get("simulation_finished"):
                        break
            _check(r.get("simulation_finished") is True,
                   f"simulation_finished after drive loop (state={r})")

            # 14. restart_simulation with a different input
            print("\n=== 14. restart_simulation ===", file=sys.stderr)
            other = REPO_ROOT / "examples" / "run" / "llamacpp_llama-3-8B_vanilla.yaml"
            r = _unwrap(await session.call_tool(
                "restart_simulation", {"input_path": str(other)}))
            _check(r["ok"] is True,
                   f"restart_simulation ok (got {r})")
            _check(r["input_path"] == str(other),
                   "input_path switched")
            r2 = _unwrap(await session.call_tool("list_breakpoints", {}))
            _check(r2["BREAK_AFTER_COMPILE_STAGE"] is True,
                   "CG_SIM_BREAKPOINTS re-applied to fresh Debugger")

            # 15. shutdown
            print("\n=== 15. shutdown ===", file=sys.stderr)
            r = _unwrap(await session.call_tool("shutdown", {}))
            _check(r["ok"] is True, "shutdown ok")

    print("\nALL CHECKS PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
