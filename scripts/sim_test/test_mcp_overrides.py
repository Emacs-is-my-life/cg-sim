"""End-to-end test for Hydra-style overrides via
`restart_simulation(overrides=[...])`.

Coverage matrix:
  1. Initial CLI overrides flow through to construction.
  2. Restart with scalar override replaces the value.
  3. Restart with `overrides=None` is sticky (keeps prior overrides).
  4. Restart with `overrides=[]` clears overrides (falls back to YAML).
  5. Indexed list override (`hardware.memory.0.args.memory_size_KB`).
  6. Multiple overrides applied in a single restart.
  7. Invalid override → CONSTRUCT_FAILED, recoverable.
  8. Combined with `input_path` switch (vanilla YAML).
  9. Response echoes `overrides` field for verification.
 10. Combined with `reload=False` — overrides still apply.

Run from repo root:  python scripts/sim_test/test_mcp_overrides.py
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
FLEXINFER_YAML = REPO_ROOT / "examples" / "llama3-flexinfer" / "input.yaml"
VANILLA_YAML = REPO_ROOT / "examples" / "llama3-vanilla" / "input.yaml"


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


async def _exec_int(session: ClientSession, expr: str) -> int:
    """Run `print(<expr>)` via execute, parse stdout as int."""
    r = _unwrap(await session.call_tool("execute", {"code": f"print({expr})"}))
    _check(r.get("ok"), f"execute({expr!r}) returned ok=True (got {r})")
    return int(r["output"].strip())


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


async def _expect_at_breakpoint(session: ClientSession, label: str) -> None:
    """start_simulation and assert we land at a breakpoint."""
    r = _unwrap(await session.call_tool("start_simulation", {}))
    _check(r.get("at_breakpoint", False),
           f"{label}: parked at breakpoint after start (got {r})")


async def main() -> int:
    env = {**os.environ, "CG_SIM_BREAKPOINTS": "BREAK_BEFORE_COMPILE_STAGE"}
    # Launch with an initial Hydra override on the CLI — Test #1 verifies
    # this flows through to the first construction. Picked
    # prefetch_window=11 (distinct from YAML's 5 and FlexInfer's hardcoded
    # default 3) so a stale value can't accidentally match.
    server = StdioServerParameters(
        command=sys.executable,
        args=[
            str(REPO_ROOT / "main_agent.py"),
            "-i", str(FLEXINFER_YAML),
            "scheduler.args.prefetch_window=11",
        ],
        env=env, cwd=str(REPO_ROOT),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- Test 1: initial CLI override flows through --------------
            print("=== Test 1: initial CLI override flows through ===",
                  file=sys.stderr)
            await _expect_at_breakpoint(session, "Test 1")
            pw = await _exec_int(session, "engine.sched.prefetch_window")
            _check(pw == 11, f"prefetch_window from CLI override == 11 (got {pw})")
            await _drive_to_finish(session)

            # --- Test 2: scalar override on restart ---------------------
            print("\n=== Test 2: scalar override on restart ===",
                  file=sys.stderr)
            r = _unwrap(await session.call_tool("restart_simulation", {
                "overrides": ["scheduler.args.prefetch_window=7"],
            }))
            _check(r.get("ok"), f"restart with overrides ok (got {r})")
            _check(r.get("overrides") == ["scheduler.args.prefetch_window=7"],
                   f"response echoes overrides exactly (got {r.get('overrides')!r})")
            await _expect_at_breakpoint(session, "Test 2")
            pw = await _exec_int(session, "engine.sched.prefetch_window")
            _check(pw == 7, f"prefetch_window after restart == 7 (got {pw})")
            await _drive_to_finish(session)

            # --- Test 3: overrides=None is sticky -----------------------
            print("\n=== Test 3: overrides=None preserves previous ===",
                  file=sys.stderr)
            r = _unwrap(await session.call_tool("restart_simulation", {}))
            _check(r.get("ok"), f"restart with no overrides arg ok (got {r})")
            _check(r.get("overrides") == ["scheduler.args.prefetch_window=7"],
                   f"sticky overrides preserved (got {r.get('overrides')!r})")
            await _expect_at_breakpoint(session, "Test 3")
            pw = await _exec_int(session, "engine.sched.prefetch_window")
            _check(pw == 7, f"sticky prefetch_window still 7 (got {pw})")
            await _drive_to_finish(session)

            # --- Test 4: overrides=[] clears ----------------------------
            print("\n=== Test 4: overrides=[] clears, falls back to YAML ===",
                  file=sys.stderr)
            r = _unwrap(await session.call_tool("restart_simulation", {
                "overrides": [],
            }))
            _check(r.get("ok"), f"restart with empty overrides ok (got {r})")
            _check(r.get("overrides") == [],
                   f"overrides cleared in response (got {r.get('overrides')!r})")
            await _expect_at_breakpoint(session, "Test 4")
            pw = await _exec_int(session, "engine.sched.prefetch_window")
            # YAML default is 5 (examples/llama3-flexinfer/input.yaml).
            _check(pw == 5, f"prefetch_window falls back to YAML default 5 (got {pw})")
            await _drive_to_finish(session)

            # --- Test 5: indexed list override (memory size) ------------
            print("\n=== Test 5: indexed list override (memory.0) ===",
                  file=sys.stderr)
            target_kb = 4194304  # 4 GiB, divisible by 4
            r = _unwrap(await session.call_tool("restart_simulation", {
                "overrides": [f"hardware.memory.0.args.memory_size_KB={target_kb}"],
            }))
            _check(r.get("ok"), f"restart with memory override ok (got {r})")
            await _expect_at_breakpoint(session, "Test 5")
            num_total_pages = await _exec_int(session, "hw['ram'].space.num_total_pages")
            _check(num_total_pages * 4 == target_kb,
                   f"memory_size_KB applied: pages*4 == {target_kb} "
                   f"(got {num_total_pages * 4})")
            await _drive_to_finish(session)

            # --- Test 6: multiple overrides in one restart --------------
            print("\n=== Test 6: multiple overrides at once ===",
                  file=sys.stderr)
            target_kb_2 = 8388608  # 8 GiB
            r = _unwrap(await session.call_tool("restart_simulation", {
                "overrides": [
                    "scheduler.args.prefetch_window=9",
                    f"hardware.memory.0.args.memory_size_KB={target_kb_2}",
                ],
            }))
            _check(r.get("ok"), f"restart with multiple overrides ok (got {r})")
            await _expect_at_breakpoint(session, "Test 6")
            pw = await _exec_int(session, "engine.sched.prefetch_window")
            num_total_pages = await _exec_int(session, "hw['ram'].space.num_total_pages")
            _check(pw == 9, f"prefetch_window from multi-override == 9 (got {pw})")
            _check(num_total_pages * 4 == target_kb_2,
                   f"memory from multi-override applied (got {num_total_pages * 4})")
            await _drive_to_finish(session)

            # --- Test 7: invalid override → CONSTRUCT_FAILED, recoverable
            print("\n=== Test 7: invalid override fails gracefully ===",
                  file=sys.stderr)
            r = _unwrap(await session.call_tool("restart_simulation", {
                # No such key in YAML; without leading `+`, Hydra refuses
                # to add. This is the most agent-likely typo class to hit.
                "overrides": ["scheduler.args.totally_made_up_field=42"],
            }))
            _check(not r.get("ok"),
                   f"restart with bad override returns ok=False (got {r})")
            _check("construction failed" in str(r.get("error", "")).lower(),
                   f"error mentions construction failure (got {r.get('error')!r})")
            _check(r.get("overrides") == ["scheduler.args.totally_made_up_field=42"],
                   f"failed-restart response echoes the attempted overrides "
                   f"(got {r.get('overrides')!r})")
            # Recovery: pass a valid overrides list, session should rebuild.
            r = _unwrap(await session.call_tool("restart_simulation", {
                "overrides": ["scheduler.args.prefetch_window=4"],
            }))
            _check(r.get("ok"), f"recovery restart ok (got {r})")
            await _expect_at_breakpoint(session, "Test 7 recovery")
            pw = await _exec_int(session, "engine.sched.prefetch_window")
            _check(pw == 4, f"prefetch_window after recovery == 4 (got {pw})")
            await _drive_to_finish(session)

            # --- Test 8: combined with input_path switch ----------------
            print("\n=== Test 8: input_path swap + overrides ===",
                  file=sys.stderr)
            target_kb_3 = 2097152  # 2 GiB
            r = _unwrap(await session.call_tool("restart_simulation", {
                "input_path": str(VANILLA_YAML),
                "overrides": [f"hardware.memory.0.args.memory_size_KB={target_kb_3}"],
            }))
            _check(r.get("ok"), f"restart with new input_path + overrides ok (got {r})")
            _check(r.get("input_path") == str(VANILLA_YAML),
                   f"input_path updated in response (got {r.get('input_path')!r})")
            await _expect_at_breakpoint(session, "Test 8")
            sched_type = (_unwrap(await session.call_tool("execute", {
                "code": "print(type(engine.sched).__name__)"
            })))["output"].strip()
            _check(sched_type == "Vanilla",
                   f"scheduler is Vanilla after input_path swap (got {sched_type!r})")
            num_total_pages = await _exec_int(session, "hw['ram'].space.num_total_pages")
            _check(num_total_pages * 4 == target_kb_3,
                   f"memory override applies under new YAML (got {num_total_pages * 4})")
            await _drive_to_finish(session)

            # --- Test 9: combined with reload=False ---------------------
            print("\n=== Test 9: overrides + reload=False ===",
                  file=sys.stderr)
            # Go back to flexinfer so we can twiddle prefetch_window.
            r = _unwrap(await session.call_tool("restart_simulation", {
                "input_path": str(FLEXINFER_YAML),
                "overrides": ["scheduler.args.prefetch_window=2"],
                "reload": True,
            }))
            _check(r.get("ok"), "interim restart back to flexinfer ok")
            await _expect_at_breakpoint(session, "Test 9 prep")
            id_before = await _exec_int(session, "id(type(engine.sched))")
            await _drive_to_finish(session)

            r = _unwrap(await session.call_tool("restart_simulation", {
                "overrides": ["scheduler.args.prefetch_window=6"],
                "reload": False,
            }))
            _check(r.get("ok"), f"restart with reload=False ok (got {r})")
            _check(r.get("reloaded_modules", -1) == 0,
                   f"reload=False evicts nothing (got {r.get('reloaded_modules')})")
            await _expect_at_breakpoint(session, "Test 9")
            id_after = await _exec_int(session, "id(type(engine.sched))")
            pw = await _exec_int(session, "engine.sched.prefetch_window")
            _check(id_after == id_before,
                   f"reload=False preserves class identity ({id_before} vs {id_after})")
            _check(pw == 6,
                   f"overrides still apply under reload=False (got pw={pw})")
            await _drive_to_finish(session)

            # --- Test 10: shutdown ---------------------------------------
            await session.call_tool("shutdown", {})

    print("\nALL OVERRIDE CHECKS PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
