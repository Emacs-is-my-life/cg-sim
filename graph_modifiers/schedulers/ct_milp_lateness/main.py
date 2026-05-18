#!/usr/bin/env python3
"""CLI: pool-first lateness MILP weight-streaming scheduler.

Same I/O shape as ``ct_milp_multistream.main`` (takes a bundle path,
optional --hw, optional --peak-target-mb), but the LP works against the
runtime trace directly — no compile-side sidecar identity layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CG_SIM_ROOT = THIS_DIR.parent.parent.parent
if str(CG_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_SIM_ROOT))

from graph_modifiers.common import (
    build_node_timeline,
    build_unified_timeline,
    load_hw_params,
    load_multi_graph_sidecars,
    load_trace_from_bundle,
    neutral_to_pytorch,
    write_neutral_schedule,
    write_schedule_json,
)
from graph_modifiers.schedulers.ct_milp_lateness.scheduler import (
    print_summary,
    solve_neutral,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("--hw", required=True)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--time-limit-s", type=float, default=120.0)
    p.add_argument("--peak-target-mb", type=float, default=None)
    p.add_argument("--safety-margin-frac", type=float, default=0.05)
    p.add_argument("--max-peak-samples", type=int, default=256)
    p.add_argument("--lp-relaxation", action="store_true")
    p.add_argument("--audit", action="store_true")
    args = p.parse_args()

    trace = load_trace_from_bundle(args.bundle)
    hw = load_hw_params(args.hw)

    peak_target_bytes = (
        int(round(args.peak_target_mb * 1e6))
        if args.peak_target_mb is not None else None
    )
    neutral = solve_neutral(
        trace,
        hw=hw,
        peak_target_bytes=peak_target_bytes,
        safety_margin_frac=float(args.safety_margin_frac),
        max_peak_samples=int(args.max_peak_samples),
        time_limit_s=float(args.time_limit_s),
        lp_relaxation=bool(args.lp_relaxation),
        audit=bool(args.audit),
    )

    out_dir = (
        Path(args.output) if args.output
        else Path(args.bundle) / "ct_milp_lateness_output"
    )
    neutral_path = out_dir / "schedule.json"
    write_neutral_schedule(neutral_path, neutral, trace=trace)
    print(f"→ neutral schedule saved: {neutral_path}")

    # Also emit the pytorch-format schedule for downstream tooling that
    # consumes ``jit_sim_prune_schedule.json``. Reuse the existing
    # builder; it accepts our cgsim_tid-resolved NeutralSchedule.
    sidecars = load_multi_graph_sidecars(args.bundle)
    if sidecars.launch_maps:
        tl = build_unified_timeline(
            trace, sidecars, cpu_per_launch_ns=hw.cpu_per_launch_ns,
        )
        node_starts, node_ends = build_node_timeline(tl, trace)
        pytorch_doc = neutral_to_pytorch(
            neutral, trace=trace,
            node_starts=node_starts, node_ends=node_ends,
        )
        pytorch_path = out_dir / "jit_sim_prune_schedule.json"
        write_schedule_json(
            pytorch_path,
            trace=trace,
            node_starts=node_starts,
            node_ends=node_ends,
            io_operations=pytorch_doc["io_operations"],
            cold_start_prefetches=pytorch_doc["cold_start_prefetches"],
            summary=pytorch_doc["summary"],
            compilation_hash=pytorch_doc.get("compilation_hash", ""),
        )
        print(f"→ pytorch schedule saved: {pytorch_path}")

    print()
    print_summary(neutral)


if __name__ == "__main__":
    main()
