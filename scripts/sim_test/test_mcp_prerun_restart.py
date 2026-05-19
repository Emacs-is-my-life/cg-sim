"""Canary test for the pre-first-run `restart_simulation` race.

Reproduces the bug surfaced during test_mcp_fragile.py development: calling
`restart_simulation` as the very first tool call — explicitly advertised
by the README as legal ("Only callable after `simulation_finished=True` (or
before the first run)") — returns `ok=True` with a misleading payload
because the queued restart action is not processed until after the FIRST
natural run finishes. Subsequent toggles land on the wrong (about-to-be-
torn-down) Debugger.

The test starts main_agent.py with the default flexinfer YAML, then as
its very first MCP tool call issues `restart_simulation` pointing at the
vanilla YAML. After fix:
  - response carries `input_path=<vanilla.yaml>`.
  - the active simulator is a fresh Vanilla one (probed via
    `type(engine.sched).__name__`).

Before fix: the active simulator is still the initial FlexInfer one.

Run from repo root:  python scripts/sim_test/test_mcp_prerun_restart.py
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
DEFAULT_YAML = REPO_ROOT / "examples" / "run" / "llamacpp_llama-3-8B_flexinfer.yaml"
ALT_YAML = REPO_ROOT / "examples" / "run" / "llamacpp_llama-3-8B_vanilla.yaml"


def _unwrap(result) -> dict:
    for c in result.content:
        if getattr(c, "type", None) == "text":
            return json.loads(c.text)
    raise RuntimeError(f"No text content: {result}")


FAILED = False


def _check(condition: bool, label: str, detail: str = "") -> None:
    global FAILED
    mark = "PASS" if condition else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line, file=sys.stderr)
    if not condition:
        FAILED = True


async def main() -> int:
    # Default startup input is FlexInfer. The agent immediately requests
    # a switch to Vanilla via restart_simulation, before any other call.
    env = {**os.environ}
    env.pop("CG_SIM_BREAKPOINTS", None)  # Don't pre-enable anything;
                                          # we'll toggle explicitly.
    env["CG_SIM_BREAKPOINTS"] = "BREAK_BEFORE_COMPILE_STAGE"
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "main_agent.py"), "-i", str(DEFAULT_YAML)],
        env=env, cwd=str(REPO_ROOT),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # The bug case: restart_simulation as the very first tool call.
            print("\n=== restart_simulation as first tool call ===",
                  file=sys.stderr)
            r = _unwrap(await session.call_tool(
                "restart_simulation", {"input_path": str(ALT_YAML)}))
            print(f"  response: {json.dumps(r, indent=2)}", file=sys.stderr)
            _check(r.get("ok") is True,
                   "response ok=True",
                   f"ok={r.get('ok')}, err={(r.get('error') or '')[:80]!r}")
            _check(r.get("input_path") == str(ALT_YAML),
                   "response.input_path equals ALT YAML",
                   f"got={r.get('input_path')!r}")
            _check(r.get("simulation_finished") is False,
                   "response.simulation_finished is False (fresh sim)",
                   f"got={r.get('simulation_finished')!r}")

            # Now start the (supposedly fresh) simulator.
            r = _unwrap(await session.call_tool("start_simulation", {}))
            _check(r.get("breakpoint") == "break_before_compile_stage",
                   "start_simulation parks at break_before_compile_stage",
                   f"bp={r.get('breakpoint')}")

            # The strongest probe: is the bound scheduler Vanilla
            # (matches ALT) or FlexInfer (the default — bug case)?
            r = _unwrap(await session.call_tool(
                "execute", {"code": "print(type(engine.sched).__name__)"}))
            sched_class = r.get("output", "").strip()
            _check(
                sched_class == "Vanilla",
                "bound scheduler matches the ALT YAML's `Vanilla`",
                f"got type(engine.sched).__name__={sched_class!r} "
                f"(expected 'Vanilla'; 'FlexInfer' means the pre-first-run "
                f"restart was silently ignored)",
            )

            # Drive to finish, then shutdown cleanly.
            for _ in range(40):
                r = _unwrap(await session.call_tool(
                    "continue_simulation", {}))
                if r.get("simulation_finished"):
                    break
            await session.call_tool("shutdown", {})

    print(file=sys.stderr)
    if FAILED:
        print("FAIL: pre-first-run restart_simulation race reproduced.",
              file=sys.stderr)
        return 1
    print("PASS: pre-first-run restart_simulation handled correctly.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
