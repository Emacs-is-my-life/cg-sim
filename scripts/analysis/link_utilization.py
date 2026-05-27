#!/usr/bin/env python3
"""Bandwidth-side view of an offload link.

For all TRANSFER_JOBs with dest=<memory_hw> in the runtime stage:
  * union-busy time and span                            → busy fraction
  * sum of `size_KB` divided by busy time              → achieved aggregate rate
  * max sampled `transfer_KBps`                        → effective peak observed
  * per-source breakdown (e.g. ssd→ram vs ram→vram0)   → which leg saturates

Optional `--window-us W` prints busy fraction per W-microsecond window
(a coarse timeline). Use the CSV side-output for full-resolution plots.

Usage:
    python link_utilization.py <log.json> <memory_hw_name> [window_us] [--out [DIR]]

Where `window_us` is an optional positional float (0 disables the
timeline). `--out` follows the convention in README.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    find_runtime_start,
    load_events,
    merge_intervals,
    parse_out_flag,
    parse_transfer_jobs,
    union_length,
    write_meta,
    write_table,
)


def main(
    log_path: Path,
    memory_hw: str,
    window_us: float = 0.0,
    *,
    out_dir: Path | None = None,
) -> None:
    events = load_events(log_path)
    t_start = find_runtime_start(events)
    transfers = parse_transfer_jobs(events, memory_hw, t_start)

    if not transfers:
        raise SystemExit(f"No TRANSFER_JOB with dest={memory_hw} after t_start")

    intervals = [(x.ts_us, x.end_us) for x in transfers]
    merged = merge_intervals(intervals)
    busy = sum(e - s for s, e in merged)
    t_min = min(s for s, _ in intervals)
    t_max = max(e for _, e in intervals)
    span = t_max - t_min

    total_kb = sum(x.size_KB for x in transfers)
    rates = [
        x.rate_KBps
        for x in transfers
        if x.rate_KBps > 0 and x.rate_KBps != float("inf")
    ]
    sample_peak = max(rates) if rates else 0.0
    achieved = (1_000_000.0 * total_kb / busy) if busy > 0 else 0.0

    print(f"event log          : {log_path}")
    print(f"memory hardware    : {memory_hw}")
    print(f"runtime start (us) : {t_start:.3f}")
    print(f"\n== Link utilization (dest={memory_hw}) ==")
    print(f"  transfers          : {len(transfers)}")
    print(f"  time span          : {span:.1f} us  ({t_min:.1f} → {t_max:.1f})")
    print(
        f"  busy union         : {busy:.1f} us  "
        f"({100*busy/max(span,1e-9):.1f}% of span)"
    )
    print(f"  bytes              : {total_kb:.1f} KB")
    print(f"  max sampled rate   : {sample_peak:.1f} KB/s")
    if sample_peak:
        print(
            f"  achieved aggregate : {achieved:.1f} KB/s "
            f"({100*achieved/sample_peak:.1f}% of max sample)"
        )
    else:
        print(f"  achieved aggregate : {achieved:.1f} KB/s")

    # Per-source breakdown
    by_src: dict[str, dict] = defaultdict(lambda: {"count": 0, "kb": 0.0, "dur": 0.0, "iv": []})
    for x in transfers:
        d = by_src[x.src_name]
        d["count"] += 1
        d["kb"] += x.size_KB
        d["dur"] += x.dur_us
        d["iv"].append((x.ts_us, x.end_us))

    print(f"\n  per source:")
    print(
        f"  {'src_name':<24} {'count':>8} {'size_KB':>14} "
        f"{'busy_us':>12} {'busy%':>8}"
    )
    for src, d in sorted(by_src.items(), key=lambda kv: -kv[1]["kb"]):
        b = union_length(d["iv"])
        print(
            f"  {src[:24]:<24} {d['count']:>8} {d['kb']:>14.1f} "
            f"{b:>12.1f} {100*b/max(span,1e-9):>7.2f}%"
        )

    timeline: list[tuple[float, float, float]] = []
    if window_us > 0:
        print(f"\n== Busy fraction per {window_us:.0f} us window ==")
        n = int((t_max - t_min) / window_us) + 1
        for i in range(n):
            lo = t_min + i * window_us
            hi = lo + window_us
            b = 0.0
            for s, e in merged:
                ovl = max(0.0, min(hi, e) - max(lo, s))
                b += ovl
            frac = b / window_us
            timeline.append((lo, hi, frac))
            print(f"  [{lo:>10.1f}, {hi:>10.1f})  busy={100*frac:>5.1f}%")

    if out_dir is None:
        return

    write_meta(
        out_dir,
        script="link_utilization",
        log_path=log_path,
        memory_hw=memory_hw,
        runtime_start_us=t_start,
        runtime_span_us=span,
        num_transfers=len(transfers),
        busy_us=busy,
        total_size_KB=total_kb,
        max_sampled_rate_KBps=sample_peak,
        achieved_aggregate_KBps=achieved,
        window_us=window_us,
    )

    write_table(
        out_dir,
        "transfers",
        [
            "ts_us",
            "end_us",
            "dur_us",
            "queue_wait_us",
            "head_wait_us",
            "xfer_us",
            "src_name",
            "dest_name",
            "size_KB",
            "rate_KBps",
            "tensor_ids",
        ],
        (
            (
                x.ts_us,
                x.end_us,
                x.dur_us,
                x.at_head_us - x.queued_us,
                x.begin_us - x.at_head_us,
                x.end_us - x.begin_us,
                x.src_name,
                x.dest_name,
                x.size_KB,
                x.rate_KBps,
                "|".join(str(t) for t in x.tensor_ids),
            )
            for x in transfers
        ),
    )

    write_table(
        out_dir,
        "per_source",
        ["src_name", "count", "size_KB", "busy_us", "busy_fraction"],
        (
            (
                src,
                d["count"],
                d["kb"],
                union_length(d["iv"]),
                union_length(d["iv"]) / max(span, 1e-9),
            )
            for src, d in sorted(by_src.items(), key=lambda kv: -kv[1]["kb"])
        ),
    )

    if timeline:
        write_table(
            out_dir,
            "busy_timeline",
            ["window_start_us", "window_end_us", "busy_fraction"],
            timeline,
        )

    print(f"\nWrote tables + meta.json to: {out_dir}")


if __name__ == "__main__":
    argv, out_dir = parse_out_flag(
        sys.argv[1:], __file__, sys.argv[1] if len(sys.argv) > 1 else ""
    )
    if len(argv) not in (2, 3):
        print(
            "Usage: python link_utilization.py <log.json> <memory_hw_name> "
            "[window_us] [--out [DIR]]"
        )
        sys.exit(2)
    log = Path(argv[0])
    mem = argv[1]
    win = float(argv[2]) if len(argv) == 3 else 0.0
    main(log, mem, win, out_dir=out_dir)
