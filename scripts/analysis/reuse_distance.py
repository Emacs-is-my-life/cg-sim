#!/usr/bin/env python3
"""LRU reuse-distance / Mattson miss-rate curve over the offload tensor stream.

Each TRANSFER_JOB with dest=<memory_hw> is treated as an LRU access for
each tensor in its `batch`. For every access, the reuse distance is the
number of *distinct* tensors touched since the tensor's previous access
(cold = first touch). The miss-rate curve

    M(C) = P[reuse_distance >= C]

is the fraction of accesses an LRU cache of capacity C tensors would
still miss. Byte-weighted variant uses average per-tensor size.

This view is *scheduler-agnostic*: it characterizes the workload's
intrinsic offload bandwidth requirement at each memory budget, and is
the natural baseline against which to measure a real scheduler.

Usage:
    python reuse_distance.py <log.json> <memory_hw_name>
        [capacities_csv] [--out [DIR]]

`capacities_csv` is an optional positional, comma-separated integer
capacities (in tensor count). Default: powers of two through 2048.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    find_runtime_start,
    load_events,
    parse_out_flag,
    parse_transfer_jobs,
    write_meta,
    write_table,
)


DEFAULT_CAPACITIES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


class LruRecency:
    """Tracks LRU recency as a Python list (MRU at tail).

    `access(key)` returns the number of distinct keys more recently
    used than `key` (i.e. the LRU reuse distance, 0-based), or -1 if
    the key is cold (first touch).

    Naive O(N) per access; fine for ~1e4 accesses. Swap in a Fenwick
    tree if traces grow.
    """

    def __init__(self) -> None:
        self.recency: list = []

    def access(self, key) -> int:
        try:
            i = self.recency.index(key)
        except ValueError:
            self.recency.append(key)
            return -1
        dist = len(self.recency) - 1 - i
        del self.recency[i]
        self.recency.append(key)
        return dist


def main(
    log_path: Path,
    memory_hw: str,
    capacities: list[int] | None = None,
    *,
    out_dir: Path | None = None,
) -> None:
    if capacities is None:
        capacities = list(DEFAULT_CAPACITIES)
    capacities = sorted(set(capacities))

    events = load_events(log_path)
    t_start = find_runtime_start(events)
    transfers = parse_transfer_jobs(events, memory_hw, t_start)
    if not transfers:
        raise SystemExit(f"No transfers to {memory_hw} after t_start")

    # Average per-tensor size across the transfers carrying it.
    size_sum: dict[int, float] = {}
    size_n: dict[int, int] = {}
    for x in transfers:
        if not x.tensor_ids:
            continue
        per = x.size_KB / len(x.tensor_ids)
        for t in x.tensor_ids:
            size_sum[t] = size_sum.get(t, 0.0) + per
            size_n[t] = size_n.get(t, 0) + 1
    size_per_tensor = {t: size_sum[t] / size_n[t] for t in size_sum}

    stat = LruRecency()
    distances: list[int] = []
    access_sizes: list[float] = []
    cold_count = 0
    cold_kb = 0.0
    for x in transfers:
        for t in x.tensor_ids:
            d = stat.access(t)
            sz = size_per_tensor.get(t, 0.0)
            if d < 0:
                cold_count += 1
                cold_kb += sz
            else:
                distances.append(d)
                access_sizes.append(sz)

    total_acc = len(distances) + cold_count
    total_kb = sum(access_sizes) + cold_kb

    print(f"event log          : {log_path}")
    print(f"memory hardware    : {memory_hw}")
    print(f"runtime start (us) : {t_start:.3f}")
    print(f"\n== Reuse-distance / Mattson curve (dest={memory_hw}) ==")
    print(f"  accesses           : {total_acc}  (cold: {cold_count})")
    print(f"  unique tensors     : {len(size_per_tensor)}")
    print(f"  bytes accessed     : {total_kb:.1f} KB")

    print(
        f"\n  {'capacity':>10}  {'miss_rate':>10}  "
        f"{'miss_KB':>14}  {'miss_KB_frac':>14}"
    )
    curve: list[tuple[int, int, float, float, float]] = []
    for C in capacities:
        miss_ct = cold_count + sum(1 for d in distances if d >= C)
        miss_kb = cold_kb + sum(s for d, s in zip(distances, access_sizes) if d >= C)
        mr = miss_ct / max(total_acc, 1)
        mk = miss_kb / max(total_kb, 1e-9)
        curve.append((C, miss_ct, mr, miss_kb, mk))
        print(f"  {C:>10}  {mr:>10.4f}  {miss_kb:>14.1f}  {mk:>14.4f}")

    if out_dir is None:
        return

    write_meta(
        out_dir,
        script="reuse_distance",
        log_path=log_path,
        memory_hw=memory_hw,
        runtime_start_us=t_start,
        capacities=capacities,
        num_accesses=total_acc,
        cold_accesses=cold_count,
        unique_tensors=len(size_per_tensor),
        total_KB=total_kb,
    )

    write_table(
        out_dir,
        "miss_curve",
        ["capacity_tensors", "miss_count", "miss_rate", "miss_KB", "miss_KB_frac"],
        curve,
    )

    write_table(
        out_dir,
        "distances",
        ["access_index", "reuse_distance", "size_KB"],
        ((i, d, s) for i, (d, s) in enumerate(zip(distances, access_sizes))),
    )

    print(f"\nWrote tables + meta.json to: {out_dir}")


def _parse_capacities(token: str) -> list[int]:
    return [int(c) for c in token.split(",") if c.strip()]


if __name__ == "__main__":
    argv, out_dir = parse_out_flag(
        sys.argv[1:], __file__, sys.argv[1] if len(sys.argv) > 1 else ""
    )
    if len(argv) not in (2, 3):
        print(
            "Usage: python reuse_distance.py <log.json> <memory_hw_name> "
            "[capacities_csv] [--out [DIR]]"
        )
        sys.exit(2)
    caps = _parse_capacities(argv[2]) if len(argv) == 3 else None
    main(Path(argv[0]), argv[1], caps, out_dir=out_dir)
