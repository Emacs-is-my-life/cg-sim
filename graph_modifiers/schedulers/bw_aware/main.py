#!/usr/bin/env python3
"""CLI: bandwidth-aware admission scheduler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CG_SIM_ROOT = THIS_DIR.parent.parent.parent
if str(CG_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_SIM_ROOT))

from graph_modifiers.schedulers.bw_aware.scheduler import (
    BWAwareKnobs, print_summary, solve_neutral,
)
from graph_modifiers.common import (
    add_calibration_args,
    build_node_timeline, build_unified_timeline,
    load_hw_params, load_multi_graph_sidecars, load_trace_from_bundle,
    neutral_to_pytorch, resolve_calibration,
    write_neutral_schedule, write_schedule_json,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Windowed bandwidth-utilization admission scheduler",
    )
    p.add_argument("bundle")
    p.add_argument("--hw", required=True)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--lock", default="",
                   help="Comma-separated graph_input_names to keep resident")
    p.add_argument("--bw-target", type=float, default=0.7,
                   help="Max H2D utilization fraction in any window (0-1).")
    p.add_argument("--value-model",
                   choices=("dram_density", "dram_size", "tensor_size",
                            "uniform", "start_ns"),
                   default="start_ns",
                   help="Order in which candidates are admitted.")
    p.add_argument("--no-drop-infeasible", dest="drop_infeasible",
                   action="store_false", default=True)
    p.add_argument("--peak-target-mb", type=float, default=None,
                   help=(
                       "Stop admitting prefetches once per-launch peak "
                       "weight VRAM ≤ TARGET_MB. Minimises streaming "
                       "(an E2E proxy) under a peak cap — same dual as "
                       "ct_milp_oracle's --peak-target-mb but with "
                       "bw_aware's heuristic admission instead of an "
                       "MILP. Note: the peak the scheduler tracks is "
                       "static-residency only; sim peak typically sits "
                       "above this by the activation/intermediate "
                       "footprint, so calibrate the target accordingly."
                   ))
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
    knobs = BWAwareKnobs(
        bw_target=float(args.bw_target),
        drop_infeasible=bool(args.drop_infeasible),
        value_model=str(args.value_model),
        peak_target_bytes=peak_target_bytes,
    )

    neutral = solve_neutral(
        trace, sidecars=sidecars, hw=hw,
        locked_graph_input_names=locked,
        knobs=knobs,
        iter_wall_ns_override=iter_wall_ns,
    )

    out_dir = (
        Path(args.output) if args.output
        else Path(args.bundle) / "bw_aware_output"
    )
    neutral_path = out_dir / "schedule.json"
    write_neutral_schedule(neutral_path, neutral)
    print(f"→ neutral schedule saved: {neutral_path}")

    tl = build_unified_timeline(
        trace, sidecars, cpu_per_launch_ns=hw.cpu_per_launch_ns,
        replicate_uses=True,
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
    print_summary(neutral)


if __name__ == "__main__":
    main()
