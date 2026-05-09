#!/usr/bin/env python3
"""Compute time-weighted average memory usage from a cg-sim trace.

The simulator records every ``CLAIM_JOB`` / ``RELEASE_JOB`` instant event
on the resource threads (e.g. ``ram``, ``vram0``) with ``num_pages`` in the
args. Each event flips the live occupancy by ``num_pages``; between events
the occupancy is constant. The time-weighted mean is therefore

    avg = (1/T) * sum_i  occupancy_i * (ts_{i+1} - ts_i)

where ``T`` is the total simulation time. Pages are 4 KB
(``sim/hw/memory/common/utils.py``), matching ``peak_memory_usage_KB``.

Usage:
    python scripts/avg_memory_usage.py runs/sdxl-turbo-ws/sim_result.json
    python scripts/avg_memory_usage.py --device vram0 runs/foo/sim_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PAGE_SIZE_KB = 4


def _load_events(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    raise ValueError(f"{path} is not a valid trace JSON file")


def _resolve_device_threads(events: list[dict]) -> dict[int, str]:
    """Return {tid: name} for memory device threads (skip non-memory threads)."""
    skip = {"Trace", "Scheduler", "Engine", "cpu", "gpu0", "ssd"}
    seen: dict[int, str] = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            tid = e.get("tid")
            name = e.get("args", {}).get("name", "")
            if tid is None or name in skip or name == "":
                continue
            seen[tid] = name
    return seen


def _find_simulation_time(events: list[dict]) -> float:
    for e in reversed(events):
        if e.get("name") == "SIMULATION_RESULT":
            return float(e.get("args", {}).get("simulation", {}).get("time", 0.0))
    raise ValueError("SIMULATION_RESULT event not found")


def _peak_kb(events: list[dict], device: str) -> float | None:
    for e in reversed(events):
        if e.get("name") == "SIMULATION_RESULT":
            for m in e.get("args", {}).get("memory", []):
                if m.get("name") == device:
                    return float(m.get("peak_memory_usage_KB", 0.0))
            return None
    return None


def _integrate(
    events: list[dict], tid: int, sim_time_us: float
) -> tuple[float, int, int, int, int, int]:
    """Time-integral of *delta* occupancy over the run for one device thread.

    The walk starts at 0 (the true baseline is unknown — pre-loaded tensors
    have no ``CLAIM_JOB`` in the trace). Returns

        (delta_area_pages_us, end_occupancy, max_occupancy, min_occupancy,
         n_claim, n_release)

    where ``*_occupancy`` are running totals of the delta walk (relative to
    the unknown baseline). The caller anchors the baseline against the
    peak reported by ``SIMULATION_RESULT``.
    """
    moves: list[tuple[float, int]] = []
    n_claim = n_release = 0
    for e in events:
        if e.get("ph") != "i" or e.get("tid") != tid:
            continue
        name = e.get("name")
        if name == "CLAIM_JOB":
            moves.append((float(e["ts"]), int(e["args"]["num_pages"])))
            n_claim += 1
        elif name == "RELEASE_JOB":
            moves.append((float(e["ts"]), -int(e["args"]["num_pages"])))
            n_release += 1
    moves.sort(key=lambda m: m[0])

    area = 0.0
    occ = 0
    max_occ = 0
    min_occ = 0
    prev_ts = 0.0
    for ts, delta in moves:
        if ts > prev_ts:
            area += occ * (ts - prev_ts)
        occ += delta
        if occ > max_occ:
            max_occ = occ
        if occ < min_occ:
            min_occ = occ
        prev_ts = ts
    if sim_time_us > prev_ts:
        area += occ * (sim_time_us - prev_ts)

    return area, occ, max_occ, min_occ, n_claim, n_release


def _format_kb(kb: float) -> str:
    return f"{kb:.0f} KB ({kb / 1024:.2f} MB, {kb / (1024 ** 2):.4f} GB)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute time-weighted average memory usage from a sim_result.json trace."
    )
    parser.add_argument("sim_result", type=Path, help="Path to sim_result.json")
    parser.add_argument(
        "--device",
        action="append",
        help="Memory device name to report (repeatable). Defaults to all detected memory threads.",
    )
    args = parser.parse_args()

    events = _load_events(args.sim_result)
    sim_time = _find_simulation_time(events)
    devices = _resolve_device_threads(events)

    if args.device:
        wanted = set(args.device)
        devices = {tid: name for tid, name in devices.items() if name in wanted}
        missing = wanted - set(devices.values())
        if missing:
            raise ValueError(f"Devices not found in trace: {sorted(missing)}")

    if sim_time <= 0:
        raise ValueError(f"Non-positive simulation time: {sim_time}")

    print(f"file: {args.sim_result}")
    print(f"sim_time_us: {sim_time:.3f}")
    print()

    for tid, name in sorted(devices.items(), key=lambda kv: kv[1]):
        area, end_occ, max_occ, min_occ, n_claim, n_release = _integrate(
            events, tid, sim_time
        )
        peak_kb = _peak_kb(events, name)

        # Delta walk starts at 0; true baseline is unknown. Anchor it so that
        # max(delta_occupancy) + baseline == peak (in pages). This is exact
        # when peak_memory_usage_KB equals the global maximum live occupancy.
        if peak_kb is None:
            print(f"[{name}] (tid={tid}): no peak_memory_usage_KB in trace; skipping")
            print()
            continue

        peak_pages = peak_kb / PAGE_SIZE_KB
        baseline_pages = peak_pages - max_occ
        avg_pages = (area / sim_time) + baseline_pages
        avg_kb = avg_pages * PAGE_SIZE_KB

        print(f"[{name}] (tid={tid})")
        print(f"  claim_events:   {n_claim}")
        print(f"  release_events: {n_release}")
        print(f"  baseline_kb:    {baseline_pages * PAGE_SIZE_KB:.0f}  "
              f"(inferred initial occupancy: peak - max_delta)")
        print(f"  avg_usage:      {_format_kb(avg_kb)}")
        print(f"  peak_usage:     {_format_kb(peak_kb)}")
        ratio = avg_kb / peak_kb if peak_kb > 0 else 0.0
        print(f"  avg/peak:       {ratio:.3f}")

        # Sanity: minimum live occupancy should be >= 0.
        live_min_pages = baseline_pages + min_occ
        if live_min_pages < 0:
            print(
                f"  WARN: implied minimum live occupancy = {live_min_pages:.0f} pages "
                f"(< 0); peak in trace may not be the true global maximum."
            )
        if end_occ != 0:
            live_end_pages = baseline_pages + end_occ
            print(
                f"  note: live occupancy at sim end = {live_end_pages:.0f} pages "
                f"({live_end_pages * PAGE_SIZE_KB:.0f} KB) "
                f"(delta walk ended at {end_occ:+d})"
            )
        print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
