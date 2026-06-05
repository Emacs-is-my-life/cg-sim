#!/usr/bin/env python3
"""CLI: pool-first lateness MILP weight-streaming scheduler.

Same I/O shape as ``ct_milp_multistream.main`` (takes a bundle path,
optional --hw, optional --peak-target-mb), but the LP works against the
runtime trace directly — no compile-side sidecar identity layer.
"""

from __future__ import annotations

import argparse
import os
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
from graph_modifiers.schedulers.ct_milp_overlap.scheduler import (
    print_summary,
    solve_neutral,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("--hw", required=True)
    p.add_argument("--output", "-o", default=None)
    default_cores = max(1, os.cpu_count() or 1)
    p.add_argument(
        "--cores",
        "--core",
        dest="cores",
        type=int,
        default=default_cores,
        help=(
            "Number of CPU cores/solver threads to use. Defaults to "
            "all detected cores (%(default)s)."
        ),
    )
    p.add_argument(
        "--baseline-sim-result",
        default=None,
        help=(
            "Path to a baseline sim_result.json. The LP harvests "
            "per-trace-node sim wall-clock times from its COMPUTE_JOB "
            "events and uses them as deadlines (instead of "
            "trace_start_ns). Run sim once with cold-all / minimal "
            "streaming to get this baseline. Omit ⇒ falls back to "
            "trace_start_ns (identical to ct_milp_lateness)."
        ),
    )
    p.add_argument("--time-limit-s", type=float, default=120.0)
    p.add_argument(
        "--phase1-time-limit-s",
        type=float,
        default=None,
        help=(
            "Separate time limit (s) for the phase-1 LP relaxation. "
            "Default None ⇒ uses --time-limit-s. Set to a small value "
            "(e.g. 30-60) when the LP is large enough that phase 1 "
            "can't finish in seconds — its only purpose is a warm-start "
            "for phase 2 MILP, so it's better to bail fast and give the "
            "full --time-limit-s to phase 2."
        ),
    )
    p.add_argument("--peak-target-mb", type=float, default=None)
    p.add_argument("--safety-margin-frac", type=float, default=0.05)
    p.add_argument("--max-peak-samples", type=int, default=256)
    p.add_argument("--lp-relaxation", action="store_true")
    p.add_argument(
        "--backpressure-edges",
        action="store_true",
        help=(
            "Derive synthetic GPU→CPU control edges from the LP's "
            "per-window lateness. The edges cap sim's CPU race-ahead, "
            "modeling the real CachingAllocator's wait-on-event "
            "behavior under memory pressure. Edges are serialized to "
            "the schedule meta and applied by the injector."
        ),
    )
    p.add_argument(
        "--backpressure-lateness-threshold-us",
        type=float,
        default=100.0,
        help="Skip backpressure edges for windows whose LP-reported "
             "lateness is below this threshold (us). Default 100us.",
    )
    p.add_argument(
        "--arc-queue-factor",
        type=float,
        default=1.0,
        help=(
            "Multiply each streaming tid's residency-arc width by this "
            "factor in the LP's peak constraint. Models the PCIe queue "
            "serialization that holds dst VRAM claimed across multiple "
            "queued transfers in sim. 1.0 = no widening (original "
            "model). Try 3-10 if sim peak exceeds LP modeled peak."
        ),
    )
    p.add_argument(
        "--lateness-peak-coupling",
        action="store_true",
        help=(
            "Re-enable the lateness→peak coupling term (OFF by default). "
            "It adds bw×window_lateness bytes to each peak row, conflating "
            "stall time with resident bytes — measured to over-predict peak "
            "by ~2 GiB and lose to the no-coupling LP on every model at tight "
            "budgets. Only enable for A/B comparison."
        ),
    )
    p.add_argument(
        "--relax-cinfeasible",
        action="store_true",
        help=(
            "Let the LP stream c-infeasible tids (default off) instead of "
            "pinning them resident, trading a startup stall for VRAM. Makes "
            "tight budgets feasible where they'd otherwise be infeasible "
            "(e.g. llama8b@6gib). Opt-in: changes the feasible set."
        ),
    )
    p.add_argument(
        "--no-intermediate-axis-fix",
        dest="intermediate_axis_fix",
        action="store_false",
        help=(
            "ABLATION: disable the intermediate-residency axis-mix fix "
            "(default on). The fix uses sim-time-only endpoints; disabling "
            "reverts to the buggy sim/trace axis mix that inflated the "
            "modeled activation floor ~10x on diffusion."
        ),
    )
    p.set_defaults(intermediate_axis_fix=True)
    p.add_argument(
        "--lookahead-ms",
        type=float,
        default=5.0,
        help=(
            "Overlap lookahead horizon W (ms): a prefetch issues at most W "
            "ahead of its consumer. Sets both the channel release time "
            "(r = max(prior_use_end, deadline − W)) and the peak residency "
            "arc width, so larger W buys more compute/transfer overlap at "
            "the cost of more VRAM held per streamed weight (and more peak). The key "
            "overlap↔peak tradeoff knob; sweep it. Default 5ms."
        ),
    )
    p.add_argument("--audit", action="store_true")
    args = p.parse_args()
    if args.cores < 1:
        p.error("--cores must be >= 1")

    trace = load_trace_from_bundle(args.bundle)
    sidecars = load_multi_graph_sidecars(args.bundle)
    hw = load_hw_params(args.hw)

    peak_target_bytes = (
        int(round(args.peak_target_mb * 1e6))
        if args.peak_target_mb is not None else None
    )
    neutral = solve_neutral(
        trace,
        hw=hw,
        baseline_sim_result_path=args.baseline_sim_result,
        peak_target_bytes=peak_target_bytes,
        safety_margin_frac=float(args.safety_margin_frac),
        max_peak_samples=int(args.max_peak_samples),
        time_limit_s=float(args.time_limit_s),
        phase1_time_limit_s=(
            float(args.phase1_time_limit_s)
            if args.phase1_time_limit_s is not None else None
        ),
        solver_threads=int(args.cores),
        lp_relaxation=bool(args.lp_relaxation),
        backpressure_edges=bool(args.backpressure_edges),
        backpressure_lateness_threshold_ns=int(
            args.backpressure_lateness_threshold_us * 1000
        ),
        arc_queue_factor=float(args.arc_queue_factor),
        lateness_peak_coupling=bool(args.lateness_peak_coupling),
        relax_cinfeasible=bool(args.relax_cinfeasible),
        intermediate_axis_fix=bool(args.intermediate_axis_fix),
        lookahead_ns=int(round(args.lookahead_ms * 1e6)),
        audit=bool(args.audit),
        sidecars=sidecars,
    )

    out_dir = (
        Path(args.output) if args.output
        else Path(args.bundle) / "ct_milp_lateness_simtime_output"
    )
    neutral_path = out_dir / "schedule.json"
    write_neutral_schedule(neutral_path, neutral, trace=trace)
    print(f"→ neutral schedule saved: {neutral_path}")

    # Also emit the pytorch-format schedule for downstream tooling that
    # consumes ``jit_sim_prune_schedule.json``. Reuse the existing
    # builder; it accepts our cgsim_tid-resolved NeutralSchedule.
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
