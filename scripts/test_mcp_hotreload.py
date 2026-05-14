"""End-to-end test for hot-reload via `restart_simulation(reload=True)`.

Workflow:
  1. Start MCP server with BREAK_BEFORE_COMPILE_STAGE pre-enabled.
  2. start_simulation → record class identity + confirm marker absent.
  3. Drive to simulation_finished.
  4. Append a marker class attribute to `sim/sched/flexinfer/flexinfer.py`
     on disk.
  5. restart_simulation(reload=True) → expect `reloaded_modules > 0`.
  6. start_simulation → record class identity + check marker.
     - Class id MUST differ from step 2 (fresh import happened).
     - Marker MUST be present (the modified source was loaded).
  7. Drive to simulation_finished.
  8. Revert source file (always, in a `finally`).
  9. restart_simulation(reload=True) so the agent doesn't carry the
     marker-bearing class into any later use.
 10. shutdown.

Also runs a control: `restart_simulation(reload=False)` between runs 1
and 2 to confirm that the class identity is *preserved* when reload is
disabled.

Run from repo root:  python scripts/test_mcp_hotreload.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_YAML = REPO_ROOT / "examples" / "llama3-flexinfer" / "input.yaml"
SCHED_FILE = REPO_ROOT / "sim" / "sched" / "flexinfer" / "flexinfer.py"
MARKER_SUFFIX = "\n\nFlexInfer.HOT_RELOAD_MARKER = 'hot-reload-ok'\n"


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


async def _start_and_capture(session: ClientSession) -> tuple[str, str]:
    """start_simulation, then read class id + marker via execute. Returns
    (id_str, marker_str). marker_str is 'NoneType' if marker absent."""
    r = _unwrap(await session.call_tool("start_simulation", {}))
    _check(r["at_breakpoint"], f"parked at breakpoint (got {r})")
    r = _unwrap(await session.call_tool("execute",
        {"code": "print(id(type(engine.sched)))"}))
    id_str = r["output"].strip()
    r = _unwrap(await session.call_tool("execute", {"code": (
        "print(getattr(type(engine.sched), 'HOT_RELOAD_MARKER', None))"
    )}))
    marker_str = r["output"].strip()
    return id_str, marker_str


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

    original_source = SCHED_FILE.read_text()
    try:
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # --- Run 1: baseline ----------------------------------
                print("=== Run 1: baseline ===", file=sys.stderr)
                id1, marker1 = await _start_and_capture(session)
                print(f"  class id: {id1}, marker: {marker1}", file=sys.stderr)
                _check(marker1 == "None",
                       f"baseline: no HOT_RELOAD_MARKER yet (got {marker1!r})")
                await _drive_to_finish(session)

                # --- Control: reload=False preserves class identity ---
                print("\n=== Control: restart with reload=False ===",
                      file=sys.stderr)
                r = _unwrap(await session.call_tool(
                    "restart_simulation", {"reload": False}))
                _check(r["ok"], f"restart_simulation reload=False ok (got {r})")
                _check(r.get("reloaded_modules", 0) == 0,
                       f"reload=False evicts nothing (got {r.get('reloaded_modules')})")
                id_ctrl, marker_ctrl = await _start_and_capture(session)
                print(f"  class id: {id_ctrl}, marker: {marker_ctrl}",
                      file=sys.stderr)
                _check(id_ctrl == id1,
                       f"reload=False preserves class identity ({id1} vs {id_ctrl})")
                _check(marker_ctrl == "None",
                       "reload=False: marker still absent")
                await _drive_to_finish(session)

                # --- Edit source on disk -----------------------------
                print(f"\n=== Editing {SCHED_FILE.name} ===", file=sys.stderr)
                SCHED_FILE.write_text(original_source + MARKER_SUFFIX)
                print(f"  appended marker (file size now "
                      f"{SCHED_FILE.stat().st_size} bytes)", file=sys.stderr)

                # --- Run 2: with reload=True -------------------------
                print("\n=== Run 2: restart with reload=True ===",
                      file=sys.stderr)
                r = _unwrap(await session.call_tool(
                    "restart_simulation", {"reload": True}))
                _check(r["ok"], f"restart_simulation reload=True ok (got {r})")
                _check(r.get("reloaded_modules", 0) > 0,
                       f"reload=True evicted some modules (got {r.get('reloaded_modules')})")
                print(f"  reloaded_modules: {r['reloaded_modules']}",
                      file=sys.stderr)
                id2, marker2 = await _start_and_capture(session)
                print(f"  class id: {id2}, marker: {marker2}", file=sys.stderr)
                _check(id2 != id1,
                       f"reload=True yields a NEW class object ({id1} vs {id2})")
                _check(marker2 == "hot-reload-ok",
                       f"reload=True applied source edit (marker={marker2!r})")
                await _drive_to_finish(session)

                # --- Run 3: revert source, reload, verify ------------
                print(f"\n=== Reverting {SCHED_FILE.name} ===", file=sys.stderr)
                SCHED_FILE.write_text(original_source)

                r = _unwrap(await session.call_tool(
                    "restart_simulation", {"reload": True}))
                _check(r["ok"], "restart after revert ok")
                id3, marker3 = await _start_and_capture(session)
                print(f"  class id: {id3}, marker: {marker3}", file=sys.stderr)
                _check(marker3 == "None",
                       f"reverted source: marker absent again (got {marker3!r})")
                _check(id3 != id2,
                       f"another reload yields another fresh class ({id2} vs {id3})")

                # --- Sanity: common base class identity preserved ----
                # Done at the run-3 breakpoint (before drive_to_finish)
                # so `execute` is still callable.
                print("\n=== Sanity: common base class identity preserved ===",
                      file=sys.stderr)
                r = _unwrap(await session.call_tool("execute", {"code": (
                    "from sim.sched.common import BaseScheduler;"
                    " print(isinstance(engine.sched, BaseScheduler))"
                )}))
                _check(r.get("ok") and r.get("output", "").strip() == "True",
                       f"isinstance against BaseScheduler still True (got {r})")

                await _drive_to_finish(session)
                await session.call_tool("shutdown", {})
    finally:
        # Always restore the source, even if the test crashed.
        if SCHED_FILE.read_text() != original_source:
            SCHED_FILE.write_text(original_source)
            print(f"\n[teardown] restored {SCHED_FILE.name}", file=sys.stderr)

    print("\nALL HOT-RELOAD CHECKS PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
