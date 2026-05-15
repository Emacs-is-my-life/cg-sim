#!/usr/bin/env python3
"""Compute exposed transfer time ("stall time") from a cg-sim event log.

    stall = | union(transfer intervals) \\ union(compute intervals) |

where the transfer set is every TRANSFER_JOB whose `args.Hardware.dest.name`
equals <memory_hw_name>, and the compute set is every COMPUTE_JOB whose
`args.Hardware.name` equals <compute_hw_name>. Only events from the Runtime
stage (ts >= RUNTIME_STAGE_START) with dur > 0 are considered. Hardware
names are matched exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


TRACK_ENGINE = 0
TRACK_EVENT = 1


def load_events(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Tolerate a log truncated mid-flush: close the array/object and retry.
        patched = text.rstrip().rstrip(",")
        if not patched.endswith("]}"):
            patched += "\n]}"
        data = json.loads(patched)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return list(data)


def find_runtime_start(events: list[dict]) -> float:
    for ev in events:
        if (
            ev.get("pid") == TRACK_ENGINE
            and ev.get("ph") == "i"
            and ev.get("name") == "RUNTIME_STAGE_START"
        ):
            return float(ev.get("ts", 0.0))
    raise RuntimeError("RUNTIME_STAGE_START event not found in event log")


def collect_compute_intervals(
    events: list[dict], hw_name: str, t_start: float
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for ev in events:
        if ev.get("pid") != TRACK_EVENT or ev.get("ph") != "X":
            continue
        if not ev.get("name", "").startswith("COMPUTE_JOB"):
            continue
        hw = ((ev.get("args") or {}).get("Hardware") or {}).get("name")
        if hw != hw_name:
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        if dur <= 0.0 or ts < t_start:
            continue
        out.append((ts, ts + dur))
    return out


def collect_transfer_intervals(
    events: list[dict], mem_name: str, t_start: float
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for ev in events:
        if ev.get("pid") != TRACK_EVENT or ev.get("ph") != "X":
            continue
        if not ev.get("name", "").startswith("TRANSFER_JOB"):
            continue
        dest = (((ev.get("args") or {}).get("Hardware") or {}).get("dest") or {}).get("name")
        if dest != mem_name:
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        if dur <= 0.0 or ts < t_start:
            continue
        out.append((ts, ts + dur))
    return out


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def difference_length(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Total length of (a \\ b), where a and b are sorted, non-overlapping."""
    total = sum(end - start for start, end in a)
    overlap = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        lo = max(a_start, b_start)
        hi = min(a_end, b_end)
        if hi > lo:
            overlap += hi - lo
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return total - overlap


def main(argv: list[str]) -> int:
    if len(argv) - 1 < 3:
        print("Usage: python parse_stall_time.py <event_log.json> <compute_hw_name> <memory_hw_name>")
        return 2

    path = Path(argv[1])
    compute_hw = argv[2]
    memory_hw = argv[3]

    events = load_events(path)
    t_start = find_runtime_start(events)

    compute_iv = merge_intervals(collect_compute_intervals(events, compute_hw, t_start))
    transfer_iv = merge_intervals(collect_transfer_intervals(events, memory_hw, t_start))

    transfer_total = sum(end - start for start, end in transfer_iv)
    compute_total = sum(end - start for start, end in compute_iv)
    stall = difference_length(transfer_iv, compute_iv)

    print(f"event log          : {path}")
    print(f"compute hardware   : {compute_hw}")
    print(f"memory hardware    : {memory_hw}")
    print(f"runtime start (us) : {t_start:.3f}")
    print(f"compute intervals  : {len(compute_iv)}  (total {compute_total:.3f} us)")
    print(f"transfer intervals : {len(transfer_iv)}  (total {transfer_total:.3f} us)")
    print(f"stall time (us)    : {stall:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
