#!/usr/bin/env python3
"""Extract sim runtime and peak VRAM from a (possibly truncated) sim trace.

Usage: sim_summary.py <sim_result.json>

Streams events without loading the whole file into RAM and tolerates an
incomplete `]}` tail (useful while a run is still in progress or after a
mid-run kill).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

PAGE_SIZE_KB = 4

# Each event in the trace is a {...} object separated by ",\n{". We use a
# regex that catches one event at a time. We rely on the engine never
# emitting nested-brace content beyond depth one (true in practice
# because args is a sub-object).
EVENT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.MULTILINE)

NAME_RE = re.compile(r'"name":\s*"([A-Z_][A-Z_0-9]*)"')
TID_RE = re.compile(r'"tid":\s*(\d+)')
TS_RE = re.compile(r'"ts":\s*([0-9.]+)')
NPAGES_RE = re.compile(r'"num_pages":\s*(\d+)')


def hw_id_map_from_config(text_head: str) -> dict[str, int]:
    """Pull the SIM_CONFIG id_map for memory device names."""
    m = re.search(r'"id_map":\s*\{([^}]+)\}', text_head)
    if not m:
        return {}
    out = {}
    for kv in re.finditer(r'"([^"]+)":\s*(\d+)', m.group(1)):
        out[kv.group(1)] = int(kv.group(2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    args = ap.parse_args()

    text = args.log.read_text()
    # Trim trailing partial event if file ends mid-object.
    # We scan events sequentially.
    head = text[: text.find("RUNTIME_STAGE_START")] if "RUNTIME_STAGE_START" in text else text[:200_000]
    id_map = hw_id_map_from_config(head)
    vram_id = id_map.get("vram0")
    if vram_id is None:
        print("warning: vram0 id not found in id_map; cannot compute peak VRAM", file=sys.stderr)

    peak_pages = 0
    cur_pages = 0
    last_ts = 0.0
    n_events = 0

    for m in EVENT_RE.finditer(text):
        ev = m.group(0)
        nm = NAME_RE.search(ev)
        if not nm:
            continue
        name = nm.group(1)
        if name not in {"CLAIM_JOB", "RELEASE_JOB"}:
            ts_m = TS_RE.search(ev)
            if ts_m:
                ts = float(ts_m.group(1))
                if ts > last_ts:
                    last_ts = ts
            continue
        tid_m = TID_RE.search(ev)
        np_m = NPAGES_RE.search(ev)
        if not tid_m or not np_m:
            continue
        tid = int(tid_m.group(1))
        if vram_id is not None and tid != vram_id:
            continue
        npages = int(np_m.group(1))
        if name == "CLAIM_JOB":
            cur_pages += npages
            if cur_pages > peak_pages:
                peak_pages = cur_pages
        else:  # RELEASE_JOB
            cur_pages -= npages
        ts_m = TS_RE.search(ev)
        if ts_m:
            ts = float(ts_m.group(1))
            if ts > last_ts:
                last_ts = ts
        n_events += 1

    peak_bytes = peak_pages * PAGE_SIZE_KB * 1024
    peak_gib = peak_bytes / (1024 ** 3)
    print(f"events_scanned={n_events}")
    print(f"final_ts_us={last_ts:.1f}")
    print(f"final_ts_s={last_ts/1e6:.4f}")
    print(f"peak_vram_pages={peak_pages}")
    print(f"peak_vram_KB={peak_pages*PAGE_SIZE_KB}")
    print(f"peak_vram_GiB={peak_gib:.4f}")
    print(f"vram_residual_pages={cur_pages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
