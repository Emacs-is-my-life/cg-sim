#!/usr/bin/env python3
"""Per-transfer/per-stall analysis for offload+prefetch research.

Single log pass produces four views, all attributable to the same set of
(transfer, consumer) pairings:

  1. Stall blame — for each gap between consecutive compute jobs on
     <compute_hw>, identify the transfer (dest=<memory_hw>) ending
     latest in the gap. Gaps with no transfer in the window are
     reported as "unattributed" (scheduler idle, not link-bound).

  2. Prefetch slack — for each transfer T, find the first compute on
     <compute_hw> with begin >= T.end_us. slack = consumer.begin - T.end.
     Positive slack quantifies "how early did this transfer arrive";
     a tight slack distribution means the prefetcher is well-tuned.

  3. Transfer phase decomposition — per transfer, split lifecycle into
     queue_wait (queued→at_head), head_wait (at_head→begin), and xfer
     (begin→end). Distinguishes "stall because queue is deep" from
     "stall because the transfer itself is slow."

  4. Per-module attribution — group blamed stall + compute time by the
     first <module_depth> dot-components of node.name. Tells you which
     layers/modules are the actual stall hotspots.

Stdout: human-readable summary. With `--out`, also writes stalls.csv,
slack.csv, transfer_phases.csv, module_rollup.csv, meta.json.
"""
from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ComputeJob,
    TransferJob,
    find_runtime_start,
    load_events,
    module_key,
    parse_compute_jobs,
    parse_transfer_jobs,
    parse_out_flag,
    percentile,
    write_meta,
    write_table,
)


def _blame_stalls(
    computes: list[ComputeJob], transfers: list[TransferJob]
) -> list[tuple[float, float, TransferJob | None, ComputeJob]]:
    """For each consecutive compute pair with a positive gap, find the
    transfer ending latest in the gap and blame the stall on it."""
    xf_by_end = sorted(transfers, key=lambda t: t.end_us)
    end_keys = [t.end_us for t in xf_by_end]
    stalls: list[tuple[float, float, TransferJob | None, ComputeJob]] = []
    for i in range(1, len(computes)):
        prev = computes[i - 1]
        cur = computes[i]
        if cur.begin_us <= prev.end_us:
            continue
        lo = bisect.bisect_right(end_keys, prev.end_us)
        hi = bisect.bisect_right(end_keys, cur.begin_us)
        in_gap = xf_by_end[lo:hi]
        blamed = max(in_gap, key=lambda t: t.end_us) if in_gap else None
        stalls.append((prev.end_us, cur.begin_us, blamed, cur))
    return stalls


def _pair_slack(
    computes: list[ComputeJob], transfers: list[TransferJob]
) -> list[tuple[TransferJob, ComputeJob, float]]:
    """Pair each transfer with the earliest compute beginning at or
    after the transfer's end. Skip transfers with no such consumer."""
    cb_keys = [c.begin_us for c in computes]
    out: list[tuple[TransferJob, ComputeJob, float]] = []
    for x in transfers:
        idx = bisect.bisect_left(cb_keys, x.end_us)
        if idx >= len(computes):
            continue
        cons = computes[idx]
        out.append((x, cons, cons.begin_us - x.end_us))
    return out


def main(
    log_path: Path,
    compute_hw: str,
    memory_hw: str,
    *,
    out_dir: Path | None = None,
    top: int = 10,
    module_depth: int = 3,
) -> None:
    events = load_events(log_path)
    t_start = find_runtime_start(events)
    computes = parse_compute_jobs(events, compute_hw, t_start)
    transfers = parse_transfer_jobs(events, memory_hw, t_start)

    if not computes:
        raise SystemExit(f"No COMPUTE_JOB on {compute_hw} after t_start")

    stalls = _blame_stalls(computes, transfers) if transfers else []
    slacks = _pair_slack(computes, transfers) if transfers else []

    total_stall = sum(e - s for s, e, _, _ in stalls)
    attrib = [t for t in stalls if t[2] is not None]
    unattr = [t for t in stalls if t[2] is None]
    attrib_dur = sum(e - s for s, e, _, _ in attrib)
    unattr_dur = sum(e - s for s, e, _, _ in unattr)

    print(f"event log          : {log_path}")
    print(f"compute hardware   : {compute_hw}")
    print(f"memory hardware    : {memory_hw}")
    print(f"runtime start (us) : {t_start:.3f}")
    print(f"compute jobs       : {len(computes)}")
    print(f"transfer jobs      : {len(transfers)}")

    print("\n== Stall blame ==")
    print(f"  stall events        : {len(stalls)}  (total {total_stall:.1f} us)")
    if stalls:
        durs = sorted(e - s for s, e, _, _ in stalls)
        denom = max(total_stall, 1e-9)
        print(
            f"  transfer-attributed : {len(attrib)}  "
            f"({attrib_dur:.1f} us, {100*attrib_dur/denom:.1f}%)"
        )
        print(
            f"  unattributed        : {len(unattr)}  "
            f"({unattr_dur:.1f} us, {100*unattr_dur/denom:.1f}%)"
        )
        print(
            f"  stall dur p50/p90/p99/max (us) : "
            f"{percentile(durs, 0.5):.1f} / {percentile(durs, 0.9):.1f} / "
            f"{percentile(durs, 0.99):.1f} / {durs[-1]:.1f}"
        )

        n_show = min(top, len(stalls))
        print(f"\n  Top {n_show} stalls (longest gaps):")
        print(
            f"  {'dur_us':>10}  {'gap_start':>12}  {'blocked_node':<32}  blamed_transfer"
        )
        for s, e, x, c in sorted(stalls, key=lambda r: r[1] - r[0], reverse=True)[:n_show]:
            if x is None:
                xstr = "(no transfer in gap — scheduler-idle stall)"
            else:
                tids = x.tensor_ids[:3]
                more = "…" if len(x.tensor_ids) > 3 else ""
                xstr = f"size={x.size_KB:.1f}KB tids={tids}{more}"
            print(f"  {e-s:>10.1f}  {s:>12.1f}  {c.name[:32]:<32}  {xstr}")

    print("\n== Prefetch slack ==")
    if slacks:
        positive = sorted(s for _, _, s in slacks if s > 0)
        tight = sum(1 for _, _, s in slacks if s <= 0)
        print(f"  pairings           : {len(slacks)}  (slack<=0 / late: {tight})")
        if positive:
            print(
                f"  slack p50/p90/p99/max (us) : "
                f"{percentile(positive, 0.5):.1f} / {percentile(positive, 0.9):.1f} / "
                f"{percentile(positive, 0.99):.1f} / {positive[-1]:.1f}"
            )
            print(f"  slack mean         : {sum(positive)/len(positive):.1f} us")
    else:
        print("  (no transfers — nothing to pair)")

    print("\n== Transfer phase decomposition ==")
    if transfers:
        qw = [x.at_head_us - x.queued_us for x in transfers]
        hw = [x.begin_us - x.at_head_us for x in transfers]
        xf = [x.end_us - x.begin_us for x in transfers]
        n = len(transfers)
        print(
            f"  queue_wait (queued→at_head)  total/mean : "
            f"{sum(qw):.1f} / {sum(qw)/n:.2f} us"
        )
        print(
            f"  head_wait  (at_head→begin)   total/mean : "
            f"{sum(hw):.1f} / {sum(hw)/n:.2f} us"
        )
        print(
            f"  xfer       (begin→end)       total/mean : "
            f"{sum(xf):.1f} / {sum(xf)/n:.2f} us"
        )
    else:
        print("  (no transfers)")

    print(f"\n== Per-module attribution (depth={module_depth}) ==")
    mod_stall: dict[str, float] = defaultdict(float)
    mod_compute: dict[str, float] = defaultdict(float)
    for c in computes:
        mod_compute[module_key(c.name, module_depth)] += c.dur_us
    for s, e, _, c in stalls:
        mod_stall[module_key(c.name, module_depth)] += e - s
    total_s = sum(mod_stall.values())
    rows = sorted(
        set(mod_compute) | set(mod_stall),
        key=lambda k: mod_stall.get(k, 0.0),
        reverse=True,
    )
    print(f"  {'module':<40} {'compute_us':>12} {'stall_us':>12} {'stall%':>8}")
    for k in rows[: max(top, 1)]:
        c_us = mod_compute.get(k, 0.0)
        s_us = mod_stall.get(k, 0.0)
        pct = 100 * s_us / total_s if total_s else 0.0
        print(f"  {k[:40]:<40} {c_us:>12.1f} {s_us:>12.1f} {pct:>7.1f}%")

    if out_dir is None:
        return

    # --- Structured side-output ---
    runtime_span_us = max(c.end_us for c in computes) - t_start
    write_meta(
        out_dir,
        script="prefetch_quality",
        log_path=log_path,
        compute_hw=compute_hw,
        memory_hw=memory_hw,
        runtime_start_us=t_start,
        runtime_span_us=runtime_span_us,
        module_depth=module_depth,
        num_compute_jobs=len(computes),
        num_transfer_jobs=len(transfers),
        total_stall_us=total_stall,
        attributed_stall_us=attrib_dur,
        unattributed_stall_us=unattr_dur,
    )

    write_table(
        out_dir,
        "stalls",
        [
            "gap_start_us",
            "gap_end_us",
            "dur_us",
            "blocked_node_id",
            "blocked_node_name",
            "blamed_transfer_end_us",
            "blamed_size_KB",
            "blamed_tensor_ids",
        ],
        (
            (
                s,
                e,
                e - s,
                c.node_id if c.node_id is not None else "",
                c.name,
                x.end_us if x else "",
                x.size_KB if x else "",
                "|".join(str(t) for t in (x.tensor_ids if x else [])),
            )
            for s, e, x, c in stalls
        ),
    )

    write_table(
        out_dir,
        "slack",
        [
            "transfer_end_us",
            "consumer_begin_us",
            "slack_us",
            "size_KB",
            "rate_KBps",
            "src_name",
            "dest_name",
            "consumer_node_id",
            "consumer_node_name",
            "tensor_ids",
        ],
        (
            (
                x.end_us,
                c.begin_us,
                s,
                x.size_KB,
                x.rate_KBps,
                x.src_name,
                x.dest_name,
                c.node_id if c.node_id is not None else "",
                c.name,
                "|".join(str(t) for t in x.tensor_ids),
            )
            for x, c, s in slacks
        ),
    )

    write_table(
        out_dir,
        "transfer_phases",
        [
            "ts_us",
            "end_us",
            "queue_wait_us",
            "head_wait_us",
            "xfer_us",
            "size_KB",
            "rate_KBps",
            "src_name",
            "dest_name",
            "tensor_ids",
        ],
        (
            (
                x.ts_us,
                x.end_us,
                x.at_head_us - x.queued_us,
                x.begin_us - x.at_head_us,
                x.end_us - x.begin_us,
                x.size_KB,
                x.rate_KBps,
                x.src_name,
                x.dest_name,
                "|".join(str(t) for t in x.tensor_ids),
            )
            for x in transfers
        ),
    )

    write_table(
        out_dir,
        "module_rollup",
        ["module", "compute_us", "stall_us"],
        (
            (k, mod_compute.get(k, 0.0), mod_stall.get(k, 0.0))
            for k in rows
        ),
    )

    print(f"\nWrote tables + meta.json to: {out_dir}")


if __name__ == "__main__":
    argv, out_dir = parse_out_flag(sys.argv[1:], __file__, sys.argv[1] if len(sys.argv) > 1 else "")
    if len(argv) != 3:
        print(
            "Usage: python prefetch_quality.py <log.json> <compute_hw_name> "
            "<memory_hw_name> [--out [DIR]]"
        )
        sys.exit(2)
    main(Path(argv[0]), argv[1], argv[2], out_dir=out_dir)
