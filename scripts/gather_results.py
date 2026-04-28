#!/usr/bin/env python3
"""Gather sweep results into a single plain-text .dat table.

Reads every .json file in tmp/results/, parses the last entry (the
SIMULATION_RESULT event written by engine._cleanup), and writes one row per
file to tmp/results/result_total.dat:

    <memory_MB>  <success 1|0>  <simulation_time_us>  <peak_memory_KB>

Rows are sorted by memory_MB ascending.
"""
import json
import re
import sys
from pathlib import Path


RESULTS_DIR = Path("tmp/results")
OUT_FILE = RESULTS_DIR / "result_total.dat"
NAME_RE = re.compile(r"flexinfer_(\d+)\.json$")


def parse_one(path: Path):
    """Return (mb, success, sim_time, peak_KB) or None if the file is unusable."""
    m = NAME_RE.search(path.name)
    if m is None:
        return None
    mb = int(m.group(1))

    with path.open() as f:
        data = json.load(f)

    # File is {"traceEvents": [...]}; the last event is SIMULATION_RESULT.
    events = data["traceEvents"]
    result = events[-1]
    if result.get("name") != "SIMULATION_RESULT":
        print(f"[WARN] {path.name}: last event is not SIMULATION_RESULT", file=sys.stderr)
        return None

    args = result["args"]
    success = 1 if args["simulation"]["success"] == "True" else 0
    sim_time = args["simulation"]["time"]
    peak_kb = args["memory"][0]["peak_memory_usage_KB"]

    return mb, success, sim_time, peak_kb


def main() -> int:
    if not RESULTS_DIR.is_dir():
        print(f"[ERROR] {RESULTS_DIR} does not exist", file=sys.stderr)
        return 1

    rows = []
    for p in sorted(RESULTS_DIR.glob("flexinfer_*.json")):
        parsed = parse_one(p)
        if parsed is not None:
            rows.append(parsed)

    rows.sort(key=lambda r: r[0])

    with OUT_FILE.open("w") as f:
        f.write("# memory_MB\tsuccess\tsimulation_time_us\tpeak_memory_KB\n")
        for mb, success, sim_time, peak_kb in rows:
            f.write(f"{mb}\t{success}\t{sim_time}\t{peak_kb}\n")

    print(f"Wrote {len(rows)} row(s) to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
