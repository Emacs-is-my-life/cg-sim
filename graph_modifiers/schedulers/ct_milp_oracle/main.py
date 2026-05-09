#!/usr/bin/env python3
"""CLI: MILP zero-stall scheduler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CG_SIM_ROOT = THIS_DIR.parent.parent.parent
if str(CG_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_SIM_ROOT))

from graph_modifiers.common import (
    add_calibration_args,
    build_node_timeline, build_unified_timeline,
    load_hw_params, load_multi_graph_sidecars, load_trace_from_bundle,
    neutral_to_pytorch, resolve_calibration,
    write_neutral_schedule, write_schedule_json,
)
from graph_modifiers.schedulers.ct_milp_oracle.scheduler import print_summary, solve, solve_neutral


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("--hw", required=True)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--lock", default="")
    p.add_argument("--duplex", action="store_true",
                   help="Assume full-duplex PCIe (H2D + D2H on separate lanes)")
    p.add_argument("--time-limit-s", type=float, default=120.0)
    p.add_argument(
        "--disable-per-rank-deadlines", action="store_true",
        help=(
            "Drop the per-(graph, iter, rank) cumulative H2D bytes ≤ "
            "deadline × bw constraints. They're a valid throughput bound "
            "under water-fill (the simulator's actual bandwidth model) "
            "and bw_aware-style schedulers obey an analogous windowed "
            "constraint. Disabling is for diagnostic/research use only."
        ),
    )
    p.add_argument(
        "--bw-overcommit-frac", type=float, default=0.9,
        help=(
            "Scale the per-graph H2D byte budget the LP enforces. "
            "Default 0.9 leaves 10%% margin for kernel-level queueing "
            "the LP doesn't model — at 1.0 the LP saturates the lane "
            "exactly and the runtime sees ~104%% busy. Raise toward "
            "1.0 only if you want maximum streaming aggressiveness; "
            "raise above 1.0 to deliberately overcommit."
        ),
    )
    p.add_argument(
        "--lp-relaxation", action="store_true",
        help=(
            "Solve the LP relaxation instead of the integer MILP. "
            "Fast (seconds) but the emission rounds fractional z at "
            "0.5 — best-effort, not optimal. Use for very large "
            "bundles where branch-and-bound times out."
        ),
    )
    p.add_argument(
        "--peak-target-mb", type=float, default=None,
        help=(
            "Switch to the peak-target dual objective: minimise total "
            "streamed bytes (proxy for E2E) subject to peak VRAM ≤ "
            "TARGET_MB. Single MILP solve — equivalent to running "
            "sim_loop with --objective=min_e2e but without the per-"
            "iteration cg-sim overhead. Default (omitted) keeps the "
            "min-peak behaviour."
        ),
    )
    add_calibration_args(p)
    args = p.parse_args()

    trace = load_trace_from_bundle(args.bundle)
    sidecars = load_multi_graph_sidecars(args.bundle)
    if not sidecars.launch_maps:
        raise RuntimeError(f"no compile sidecars in {args.bundle}")
    hw = load_hw_params(args.hw)
    iter_wall_ns = resolve_calibration(args, hw)
    locked = {t.strip() for t in args.lock.split(",") if t.strip()}

    peak_target_bytes = (
        int(round(args.peak_target_mb * 1e6))
        if args.peak_target_mb is not None else None
    )
    neutral = solve_neutral(
        trace, sidecars=sidecars, hw=hw,
        locked_graph_input_names=locked,
        duplex=args.duplex,
        time_limit_s=args.time_limit_s,
        iter_wall_ns_override=iter_wall_ns,
        disable_per_rank_deadlines=bool(args.disable_per_rank_deadlines),
        bw_overcommit_frac=float(args.bw_overcommit_frac),
        lp_relaxation=bool(args.lp_relaxation),
        peak_target_bytes=peak_target_bytes,
    )

    out_dir = (
        Path(args.output) if args.output
        else Path(args.bundle) / "ct_milp_oracle_output"
    )
    neutral_path = out_dir / "schedule.json"
    write_neutral_schedule(neutral_path, neutral)
    print(f"→ neutral schedule saved: {neutral_path}")

    # Also emit the pytorch runtime format for convenience.
    tl = build_unified_timeline(
        trace, sidecars, cpu_per_launch_ns=hw.cpu_per_launch_ns,
    )
    node_starts, node_ends = build_node_timeline(tl, trace)
    pytorch_doc = neutral_to_pytorch(
        neutral, trace=trace, node_starts=node_starts, node_ends=node_ends,
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

    # Compat summary printout uses the fields written in neutral.meta.
    print(
        f"\nVariant: {neutral.meta.get('io_model')} "
        f"| graphs: {neutral.meta.get('graph_order')} "
        f"| peak: {neutral.meta.get('milp_peak_mb')} MB "
        f"| PCIe used: {neutral.meta.get('pcie_used_mb')} MB "
        f"| prefetches: {len(neutral.prefetches)} "
        f"| evicts: {len(neutral.evicts)} "
        f"| cold_start: {len(neutral.cold_starts)}"
    )


if __name__ == "__main__":
    main()
