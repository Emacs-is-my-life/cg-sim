"""End-to-end test for full-simulator hot-reload via
`restart_simulation(reload=True)`.

The MCP daemon reloads the ENTIRE `sim.*` tree on reload=True — user
code (schedulers / hardware / loaders) AND framework core
(engine / system / job / trace / ...), sparing only the live daemon
harness (`agent_runner`, `agent_server`). This test verifies all of:

  * USER-CODE reload: an edit to the scheduler source
    (sim/sched/llamacpp_flexinfer/flexinfer.py) is picked up, and the
    scheduler class object identity changes.
  * CORE reload (the capability this change adds): an edit to a core
    file (sim/core/engine/engine.py) is picked up, and the Engine class
    object identity changes.
  * Identity consistency (regression guard): the `DataRegionAccess`
    enum imported by core mutation code is the SAME object the hardware
    layer defines. A partial reload (core spared, hw reloaded) split
    this enum across two module instances and deadlocked the scheduler
    on its own regions; reloading core together with hw closes the gap.
  * reload=False control: nothing is evicted and class identities are
    preserved.
  * The reloaded common base class still satisfies isinstance.

Both edited files are restored in a `finally`.

Run from repo root:  python scripts/sim_test/test_mcp_hotreload.py
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
INPUT_YAML = REPO_ROOT / "examples" / "run" / "llamacpp__llama-3-8B__flexinfer.yaml"

# User-editable scheduler source (renamed llamacpp_flexinfer / LlamaCppFlexInfer).
SCHED_FILE = REPO_ROOT / "sim" / "sched" / "llamacpp_flexinfer" / "flexinfer.py"
SCHED_MARKER_SUFFIX = "\n\nLlamaCppFlexInfer.HOT_RELOAD_MARKER = 'sched-reload-ok'\n"

# Framework core source — now in the reload scope.
CORE_FILE = REPO_ROOT / "sim" / "core" / "engine" / "engine.py"
CORE_MARKER_SUFFIX = "\n\nEngine.HOT_RELOAD_CORE_MARKER = 'core-reload-ok'\n"


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


async def _exec_str(session: ClientSession, code: str) -> str:
    r = _unwrap(await session.call_tool("execute", {"code": code}))
    _check(r.get("ok"), f"execute ok: {code.splitlines()[0]!r} (got {r})")
    return r["output"].strip()


async def _probe(session: ClientSession) -> dict:
    """start_simulation, then capture identity / marker / enum probes at
    the break_before_compile_stage breakpoint."""
    r = _unwrap(await session.call_tool("start_simulation", {}))
    _check(r["at_breakpoint"], f"parked at breakpoint (got {r})")
    sched_id = await _exec_str(session, "print(id(type(engine.sched)))")
    sched_marker = await _exec_str(
        session, "print(getattr(type(engine.sched), 'HOT_RELOAD_MARKER', None))")
    engine_id = await _exec_str(session, "print(id(type(engine)))")
    core_marker = await _exec_str(
        session, "print(getattr(type(engine), 'HOT_RELOAD_CORE_MARKER', None))")
    # The enum-split regression guard: the DataRegionAccess that core
    # mutation code holds must be the very object the hw layer defines.
    enum_consistent = await _exec_str(session, (
        "import sim.core.job.mutation.transfer_mutation as _m\n"
        "from sim.hw.common.data_region import DataRegionAccess as _dra\n"
        "print(getattr(_m, 'DataRegionAccess', None) is _dra)"
    ))
    return {
        "sched_id": sched_id, "sched_marker": sched_marker,
        "engine_id": engine_id, "core_marker": core_marker,
        "enum_consistent": enum_consistent,
    }


async def _drive_to_finish(session: ClientSession) -> None:
    for _ in range(30):
        r = _unwrap(await session.call_tool("continue_simulation", {}))
        if r.get("simulation_finished"):
            return
        if r.get("timed_out"):
            r = _unwrap(await session.call_tool("current_state", {}))
            if r.get("simulation_finished"):
                return
    raise SystemExit("FAIL: drive-to-finish never reached simulation_finished")


async def main() -> int:
    env = {**os.environ, "CG_SIM_BREAKPOINTS": "BREAK_BEFORE_COMPILE_STAGE"}
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "main_agent.py"), "-i", str(INPUT_YAML)],
        env=env, cwd=str(REPO_ROOT),
    )

    sched_original = SCHED_FILE.read_text()
    core_original = CORE_FILE.read_text()
    try:
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # --- Run 1: baseline ------------------------------------
                print("=== Run 1: baseline ===", file=sys.stderr)
                p1 = await _probe(session)
                print(f"  {p1}", file=sys.stderr)
                _check(p1["sched_marker"] == "None", "baseline: no scheduler marker")
                _check(p1["core_marker"] == "None", "baseline: no core marker")
                _check(p1["enum_consistent"] == "True",
                       "baseline: DataRegionAccess identity consistent (core<->hw)")
                await _drive_to_finish(session)

                # --- Control: reload=False preserves identity -----------
                print("\n=== Control: restart reload=False ===", file=sys.stderr)
                r = _unwrap(await session.call_tool(
                    "restart_simulation", {"reload": False}))
                _check(r["ok"], f"restart reload=False ok (got {r})")
                _check(r.get("reloaded_modules", -1) == 0,
                       f"reload=False evicts nothing (got {r.get('reloaded_modules')})")
                pc = await _probe(session)
                _check(pc["sched_id"] == p1["sched_id"],
                       "reload=False preserves scheduler class identity")
                _check(pc["engine_id"] == p1["engine_id"],
                       "reload=False preserves Engine (core) class identity")
                _check(pc["enum_consistent"] == "True",
                       "reload=False: enum identity still consistent")
                await _drive_to_finish(session)

                # --- Edit BOTH a user file and a core file --------------
                print("\n=== Editing scheduler + core source ===", file=sys.stderr)
                SCHED_FILE.write_text(sched_original + SCHED_MARKER_SUFFIX)
                CORE_FILE.write_text(core_original + CORE_MARKER_SUFFIX)

                # --- Run 2: reload=True picks up BOTH edits --------------
                print("\n=== Run 2: restart reload=True ===", file=sys.stderr)
                r = _unwrap(await session.call_tool(
                    "restart_simulation", {"reload": True}))
                _check(r["ok"], f"restart reload=True ok (got {r})")
                _check(r.get("reloaded_modules", 0) > 0,
                       f"reload=True evicted modules (got {r.get('reloaded_modules')})")
                print(f"  reloaded_modules: {r['reloaded_modules']}", file=sys.stderr)
                p2 = await _probe(session)
                print(f"  {p2}", file=sys.stderr)
                _check(p2["sched_id"] != p1["sched_id"],
                       "reload=True: NEW scheduler class object")
                _check(p2["sched_marker"] == "sched-reload-ok",
                       f"reload=True applied scheduler edit (got {p2['sched_marker']!r})")
                _check(p2["engine_id"] != p1["engine_id"],
                       "reload=True: NEW Engine class object — CORE was reloaded")
                _check(p2["core_marker"] == "core-reload-ok",
                       f"reload=True applied CORE edit (got {p2['core_marker']!r})")
                _check(p2["enum_consistent"] == "True",
                       "reload=True: DataRegionAccess identity consistent (no split)")
                await _drive_to_finish(session)

                # --- Run 3: revert, reload, verify clean ----------------
                print("\n=== Reverting source; Run 3 ===", file=sys.stderr)
                SCHED_FILE.write_text(sched_original)
                CORE_FILE.write_text(core_original)
                r = _unwrap(await session.call_tool(
                    "restart_simulation", {"reload": True}))
                _check(r["ok"], "restart after revert ok")
                p3 = await _probe(session)
                _check(p3["sched_marker"] == "None", "reverted: scheduler marker gone")
                _check(p3["core_marker"] == "None", "reverted: core marker gone")
                _check(p3["sched_id"] != p2["sched_id"],
                       "another reload yields another fresh scheduler class")
                _check(p3["enum_consistent"] == "True",
                       "reverted: enum identity consistent")
                # The common base class was reloaded too, so isinstance holds.
                out = await _exec_str(session, (
                    "from sim.sched.common import BaseScheduler\n"
                    "print(isinstance(engine.sched, BaseScheduler))"))
                _check(out == "True",
                       f"isinstance against (reloaded) BaseScheduler True (got {out!r})")
                await _drive_to_finish(session)
                await session.call_tool("shutdown", {})
    finally:
        # Always restore both sources, even if the test crashed.
        if SCHED_FILE.read_text() != sched_original:
            SCHED_FILE.write_text(sched_original)
            print(f"\n[teardown] restored {SCHED_FILE.name}", file=sys.stderr)
        if CORE_FILE.read_text() != core_original:
            CORE_FILE.write_text(core_original)
            print(f"[teardown] restored {CORE_FILE.name}", file=sys.stderr)

    print("\nALL HOT-RELOAD CHECKS PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
