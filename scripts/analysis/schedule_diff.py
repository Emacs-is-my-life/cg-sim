#!/usr/bin/env python3
"""Compare two cg-sim runs node-by-node.

Both logs must come from the same trace (so node_ids align). Reports:

  * Total runtime span, compute busy, transfer busy, stall for each
  * Per-node delta (Δbegin, Δend) on the common node-id set
  * Top winners (most-shifted-earlier end) and losers (most-shifted-later)
  * Per-module rollup of compute time and end-time deltas

Diff convention is `B - A`, so positive Δend on log B means B is slower
on that node.

Usage:
    python schedule_diff.py <log_a> <log_b> <compute_hw> <memory_hw>
        [--out [DIR]]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ComputeJob,
    find_runtime_start,
    load_events,
    module_key,
    parse_compute_jobs,
    parse_out_flag,
    parse_transfer_jobs,
    union_length,
    write_meta,
    write_table,
)


def _summarize(log_path: Path, compute_hw: str, memory_hw: str) -> dict:
    events = load_events(log_path)
    t0 = find_runtime_start(events)
    computes = parse_compute_jobs(events, compute_hw, t0)
    transfers = parse_transfer_jobs(events, memory_hw, t0)
    if not computes:
        raise SystemExit(f"{log_path}: no COMPUTE_JOB on {compute_hw} after t_start")
    span = max(c.end_us for c in computes) - t0
    compute_busy = union_length([(c.ts_us, c.end_us) for c in computes])
    transfer_busy = union_length([(x.ts_us, x.end_us) for x in transfers])
    return {
        "t0": t0,
        "span": span,
        "compute_busy": compute_busy,
        "transfer_busy": transfer_busy,
        "stall": span - compute_busy,
        "computes": computes,
        "transfers": transfers,
    }


def main(
    log_a: Path,
    log_b: Path,
    compute_hw: str,
    memory_hw: str,
    *,
    out_dir: Path | None = None,
    top: int = 10,
    module_depth: int = 3,
) -> None:
    a = _summarize(log_a, compute_hw, memory_hw)
    b = _summarize(log_b, compute_hw, memory_hw)

    print(f"log A              : {log_a}")
    print(f"log B              : {log_b}")
    print(f"compute hardware   : {compute_hw}")
    print(f"memory hardware    : {memory_hw}")

    print("\n== Aggregate diff ==")
    print(f"  {'metric':<28} {'A':>14} {'B':>14} {'B-A':>14}")
    for key, label in [
        ("span", "runtime span (us)"),
        ("compute_busy", "compute busy (us)"),
        ("transfer_busy", "transfer busy (us)"),
        ("stall", "stall (us)"),
    ]:
        va, vb = a[key], b[key]
        print(f"  {label:<28} {va:>14.1f} {vb:>14.1f} {vb-va:>+14.1f}")

    # Node alignment by node_id (runtime-relative timestamps).
    by_id_a = {c.node_id: c for c in a["computes"] if c.node_id is not None}
    by_id_b = {c.node_id: c for c in b["computes"] if c.node_id is not None}
    common = sorted(set(by_id_a) & set(by_id_b))
    a_only = sorted(set(by_id_a) - set(by_id_b))
    b_only = sorted(set(by_id_b) - set(by_id_a))

    print(f"\n  aligned nodes      : {len(common)}")
    if a_only:
        print(f"  A-only node_ids    : {len(a_only)} (first: {a_only[:5]})")
    if b_only:
        print(f"  B-only node_ids    : {len(b_only)} (first: {b_only[:5]})")

    deltas: list[tuple] = []
    for nid in common:
        ca: ComputeJob = by_id_a[nid]
        cb: ComputeJob = by_id_b[nid]
        deltas.append(
            (
                nid,
                ca.name,
                (cb.begin_us - b["t0"]) - (ca.begin_us - a["t0"]),
                (cb.end_us - b["t0"]) - (ca.end_us - a["t0"]),
                ca.dur_us,
                cb.dur_us,
            )
        )

    winners = sorted(deltas, key=lambda d: d[3])[:top]
    losers = sorted(deltas, key=lambda d: -d[3])[:top]

    print(f"\n== Top {top} winners (Δend most negative) ==")
    print(f"  {'node_id':>8}  {'Δend_us':>10}  {'Δbegin_us':>10}  name")
    for nid, name, db, de, _, _ in winners:
        print(f"  {nid:>8}  {de:>+10.1f}  {db:>+10.1f}  {name[:50]}")

    print(f"\n== Top {top} losers (Δend most positive) ==")
    print(f"  {'node_id':>8}  {'Δend_us':>10}  {'Δbegin_us':>10}  name")
    for nid, name, db, de, _, _ in losers:
        print(f"  {nid:>8}  {de:>+10.1f}  {db:>+10.1f}  {name[:50]}")

    # Per-module rollup
    mod: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for _, name, _, de, da, dbu in deltas:
        m = mod[module_key(name, module_depth)]
        m[0] += da
        m[1] += dbu
        m[2] += de
        m[3] += 1

    print(f"\n== Per-module rollup (depth={module_depth}) ==")
    print(
        f"  {'module':<40} {'count':>8} {'durA_us':>12} "
        f"{'durB_us':>12} {'ΣΔend_us':>14}"
    )
    rows = sorted(mod.items(), key=lambda kv: -abs(kv[1][2]))[:top]
    for k, (da, db, ds, n) in rows:
        print(f"  {k[:40]:<40} {n:>8} {da:>12.1f} {db:>12.1f} {ds:>+14.1f}")

    if out_dir is None:
        return

    write_meta(
        out_dir,
        script="schedule_diff",
        log_a=log_a,
        log_b=log_b,
        compute_hw=compute_hw,
        memory_hw=memory_hw,
        module_depth=module_depth,
        aligned_nodes=len(common),
        a_only_nodes=len(a_only),
        b_only_nodes=len(b_only),
        span_us={"A": a["span"], "B": b["span"]},
        compute_busy_us={"A": a["compute_busy"], "B": b["compute_busy"]},
        transfer_busy_us={"A": a["transfer_busy"], "B": b["transfer_busy"]},
        stall_us={"A": a["stall"], "B": b["stall"]},
    )

    write_table(
        out_dir,
        "node_diff",
        [
            "node_id",
            "node_name",
            "delta_begin_us",
            "delta_end_us",
            "dur_A_us",
            "dur_B_us",
        ],
        deltas,
    )

    write_table(
        out_dir,
        "module_rollup",
        ["module", "count", "dur_A_us", "dur_B_us", "sum_delta_end_us"],
        ((k, n, da, db, ds) for k, (da, db, ds, n) in mod.items()),
    )

    print(f"\nWrote tables + meta.json to: {out_dir}")


if __name__ == "__main__":
    # Use log_a's stem for the default out_dir.
    argv, out_dir = parse_out_flag(
        sys.argv[1:], __file__, sys.argv[1] if len(sys.argv) > 1 else ""
    )
    if len(argv) != 4:
        print(
            "Usage: python schedule_diff.py <log_a.json> <log_b.json> "
            "<compute_hw_name> <memory_hw_name> [--out [DIR]]"
        )
        sys.exit(2)
    main(Path(argv[0]), Path(argv[1]), argv[2], argv[3], out_dir=out_dir)
