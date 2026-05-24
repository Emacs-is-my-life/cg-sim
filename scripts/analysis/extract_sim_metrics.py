#!/usr/bin/env python3
"""Extract simulation_time (s) and peak vram0 (GB) from cg-sim result.json files.

Reads SIMULATION_RESULT events emitted by sim/core/engine. Useful for
condensing the result.json files produced by `python3 main.py -i ...` into
a one-line-per-run summary table.

Usage:
    python3 scripts/analysis/extract_sim_metrics.py \\
        output/pytorch-eager__llama-3-3B__vanilla/sim_results/result.json \\
        output/pytorch-lazy__sd3__vanilla/sim_results/result.json
"""
import json
import sys
from pathlib import Path


def extract(path: Path) -> tuple[float, float]:
    """Returns (simulation_time_s, peak_vram0_GB)."""
    with open(path) as f:
        data = json.load(f)
    for e in data["traceEvents"]:
        if e.get("name") == "SIMULATION_RESULT":
            args = e["args"]
            mem = {m["name"]: m["peak_memory_usage_KB"] for m in args["memory"]}
            # engine.timestamp_now is in microseconds; vram0 peak is KB.
            sim_time_s = args["simulation"]["time"] / 1e6
            peak_vram_gb = mem.get("vram0", 0) / (1024 * 1024)
            return sim_time_s, peak_vram_gb
    raise RuntimeError(f"No SIMULATION_RESULT in {path}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        t, v = extract(Path(p))
        # Conventional layout: output/<setup>/sim_results/result.json
        tag = Path(p).parts[-3] if len(Path(p).parts) >= 3 else str(p)
        print(f"{tag:50s}  {t:.3f}s  {v:.2f} GB")
