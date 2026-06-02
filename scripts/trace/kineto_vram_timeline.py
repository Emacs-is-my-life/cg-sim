#!/usr/bin/env python3
"""§3 step-1 prototype: extract the kineto CUDA allocator timeline (offline).

The raw kineto trace (examples/trace/diffusers-group-offload__*/<...>_gpu_trace.json,
0.5-1.4 GB, pretty-printed) carries `[memory]` instant events. Each is the
ground-truth alloc/free of one device buffer:

  {"ph":"i","cat":"cpu_instant_event","s":"t","name":"[memory]",
   "pid":..,"tid":..,
   "ts": <microseconds, float>,
   "args": {"Total Reserved":..,"Total Allocated":..,"Bytes":<signed>,
            "Device Id":<int>,"Device Type":<0=CPU|1=CUDA>,"Addr":<int>,..}}

`Bytes` > 0 = alloc, < 0 = free. We stream (never json.load the whole file),
keep only CUDA (Device Type 1), and:
  - track the running sum of Bytes -> its peak is the inference-scope VRAM peak
    (target: SDXL 129 MiB, SD3 464 MiB);
  - pair each free with the open alloc at the same Addr -> (size, alloc_ts,
    free_ts) allocations, the real lifetimes we want the sim to reproduce.

Usage: python3 tmp/kineto_vram_timeline.py {sdxl|sd3}
"""
import re
import sys
from pathlib import Path
from collections import Counter

TRACES = {
    "sdxl": "examples/trace/diffusers-group-offload__sdxl-turbo__RTX4090/"
            "sdxl_offload_group_leaf_level_nonblocking_gpu_trace.json",
    "sd3":  "examples/trace/diffusers-group-offload__sd3__RTX4090/"
            "sd3_offload_group_leaf_level_nonblocking_gpu_trace.json",
}

_TS = re.compile(r'"ts":\s*([0-9.eE+-]+)')
_BYTES = re.compile(r'"Bytes":\s*(-?\d+)')
_ADDR = re.compile(r'"Addr":\s*(\d+)')
_DTYPE = re.compile(r'"Device Type":\s*(-?\d+)')
_TALLOC = re.compile(r'"Total Allocated":\s*(-?\d+)')

MIB = 1024 * 1024


def stream_cuda_mem_events(path):
    """Yield (ts, bytes_signed, addr, total_allocated) for CUDA [memory] events."""
    state = None      # None | 'mem'
    cur_ts = None
    with open(path, "r") as f:
        for line in f:
            if '"name": "[memory]"' in line:
                state, cur_ts = "mem", None
                continue
            if state != "mem":
                continue
            if cur_ts is None and '"ts":' in line:
                m = _TS.search(line)
                if m:
                    cur_ts = float(m.group(1))
                continue
            if '"Addr":' in line:  # the args content line
                dtype = _DTYPE.search(line)
                if dtype and int(dtype.group(1)) == 1:   # CUDA only
                    b = int(_BYTES.search(line).group(1))
                    addr = int(_ADDR.search(line).group(1))
                    ta = int(_TALLOC.search(line).group(1))
                    yield (cur_ts, b, addr, ta)
                state = None
    return


def main(model: str) -> None:
    path = Path(TRACES[model])
    print(f"streaming {path} ({path.stat().st_size/1e9:.2f} GB) ...")

    open_allocs = {}            # addr -> (alloc_ts, size)
    allocs = []                 # (size, alloc_ts, free_ts)
    running = 0
    peak = 0
    peak_ts = None
    max_total_alloc = 0
    n = 0
    n_alloc = n_free = n_free_unmatched = 0

    for ts, b, addr, ta in stream_cuda_mem_events(path):
        n += 1
        running += b
        if running > peak:
            peak, peak_ts = running, ts
        if ta > max_total_alloc:
            max_total_alloc = ta
        if b > 0:
            n_alloc += 1
            open_allocs[addr] = (ts, b)       # caching allocator reuses addrs serially
        else:
            n_free += 1
            size = -b
            if addr in open_allocs:
                a_ts, a_size = open_allocs.pop(addr)
                allocs.append((a_size, a_ts, ts))
            else:
                n_free_unmatched += 1          # alloc happened before profiling scope

    still_open = len(open_allocs)
    print(f"\nCUDA [memory] events: {n}")
    print(f"  allocs={n_alloc} frees={n_free} (unmatched frees={n_free_unmatched}, still-open@end={still_open})")
    print(f"  running-sum PEAK = {peak/MIB:.2f} MiB  (target: sdxl 129.2 / sd3 464.4)")
    print(f"  max 'Total Allocated' = {max_total_alloc/MIB:.2f} MiB  (cross-check, should match peak)")
    print(f"  paired allocations: {len(allocs)}")

    # size distribution of paired allocations (the lifetimes we'd schedule)
    by_size = Counter(sz for sz, _, _ in allocs)
    tot_bytes_paired = sum(sz for sz, _, _ in allocs)
    print(f"\n  distinct allocation sizes: {len(by_size)}; total paired bytes (sum over lifetimes): {tot_bytes_paired/MIB:.0f} MiB")
    print("  top sizes by count*size (the dominant buffers to match):")
    ranked = sorted(by_size.items(), key=lambda kv: -kv[0]*kv[1])[:15]
    for sz, cnt in ranked:
        # mean lifetime for this size
        lifes = [ft - at for s, at, ft in allocs if s == sz]
        mean_life_us = sum(lifes)/len(lifes) if lifes else 0
        print(f"    {sz/MIB:8.3f} MiB x{cnt:5d}  = {sz*cnt/MIB:8.1f} MiB-instances  mean_life={mean_life_us:.0f} us")

    # how concentrated is the peak among big buffers?
    big = [(sz, at, ft) for sz, at, ft in allocs if sz >= 1*MIB]
    print(f"\n  allocations >= 1 MiB: {len(big)} (these dominate VRAM; small ones left to refcount)")
    if big:
        print(f"    their distinct sizes: {sorted(set(round(s/MIB,3) for s,_,_ in big))[:20]}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TRACES:
        print("usage: python3 tmp/kineto_vram_timeline.py {sdxl|sd3}")
        sys.exit(2)
    main(sys.argv[1])
