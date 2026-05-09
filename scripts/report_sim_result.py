#!/usr/bin/env python3
"""Report summary metrics from a cg-sim ``sim_result.json`` file.

Usage:
    python scripts/report_sim_result.py runs/sdxl-turbo-profile-0423/sim_result.json
    python scripts/report_sim_result.py --memory-name vram0 runs/foo/sim_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_events(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    raise ValueError(f"{path} is not a valid trace JSON file")


def _find_result_event(events: list[dict]) -> dict:
    for event in reversed(events):
        if event.get("name") == "SIMULATION_RESULT":
            return event
    raise ValueError("SIMULATION_RESULT event not found")


def _format_bool(value: str) -> str:
    return "success" if value == "True" else "failed"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report peak VRAM and end-to-end time from a sim_result.json file."
    )
    parser.add_argument("sim_result", type=Path, help="Path to sim_result.json")
    parser.add_argument(
        "--memory-name",
        default="vram0",
        help="Memory device to report as VRAM (default: vram0)",
    )
    args = parser.parse_args()

    events = _load_events(args.sim_result)
    result = _find_result_event(events)
    payload = result.get("args", {})

    simulation = payload.get("simulation", {})
    memories = payload.get("memory", [])
    jobs = payload.get("job", {})

    peak_kb = None
    for memory in memories:
        if memory.get("name") == args.memory_name:
            peak_kb = memory.get("peak_memory_usage_KB")
            break

    if peak_kb is None:
        available = ", ".join(sorted(str(m.get("name")) for m in memories)) or "<none>"
        raise ValueError(
            f"Memory device {args.memory_name!r} not found. Available devices: {available}"
        )

    e2e_us = float(simulation.get("time", 0.0))
    status = _format_bool(str(simulation.get("success", "False")))

    print(f"file: {args.sim_result}")
    print(f"status: {status}")
    print(f"peak_vram_kb: {peak_kb}")
    print(f"peak_vram_mb: {peak_kb / 1024:.2f}")
    print(f"peak_vram_gb: {peak_kb / (1024 ** 2):.4f}")
    print(f"e2e_time_us: {e2e_us:.3f}")
    print(f"e2e_time_ms: {e2e_us / 1000:.3f}")
    print(f"e2e_time_s: {e2e_us / 1_000_000:.6f}")

    if jobs:
        print(f"compute_jobs: {jobs.get('compute_job_counts', 0)}")
        print(f"transfer_jobs: {jobs.get('transfer_job_counts', 0)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
