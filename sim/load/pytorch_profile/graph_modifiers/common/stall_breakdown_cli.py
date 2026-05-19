"""CLI: print Level-2 stall breakdown for a bundle.

Usage:
    python -m graph_modifiers.common.stall_breakdown_cli BUNDLE_DIR --hw hw.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CG_SIM_ROOT = THIS_DIR.parent.parent
if str(CG_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_SIM_ROOT))

from sim.load.pytorch_profile.graph_modifiers.common import (
    load_trace_from_bundle,
    load_sidecars,
    load_hw_params,
)
from sim.load.pytorch_profile.graph_modifiers.common.streamed_profile import build_streamed_profile
from sim.load.pytorch_profile.graph_modifiers.common.stall_breakdown import compute_stall_breakdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--hw", default=None)
    ap.add_argument("--iter-wall-ns", type=int, default=None)
    ap.add_argument("--iter-wall-ms", type=float, default=None)
    ap.add_argument("--profiler-overhead-ns", type=int, default=None)
    args = ap.parse_args()

    trace = load_trace_from_bundle(args.bundle)
    sc = load_sidecars(args.bundle)
    if sc.launch_map is None or sc.tensor_map is None:
        raise SystemExit(f"bundle {args.bundle} missing sidecars")

    h2d_Bps = 25e9
    overhead_ns = 0
    if args.hw:
        hw = load_hw_params(args.hw)
        if hw.h2d_bw > 0:
            h2d_Bps = hw.h2d_bw * 1e9  # B/ns -> B/s
        overhead_ns = hw.profiler_overhead_per_event_ns
    if args.profiler_overhead_ns is not None:
        overhead_ns = int(args.profiler_overhead_ns)

    iter_wall_ns = args.iter_wall_ns
    if iter_wall_ns is None and args.iter_wall_ms is not None:
        iter_wall_ns = int(args.iter_wall_ms * 1e6)

    sp = build_streamed_profile(
        trace, sc.launch_map, sc.tensor_map,
        profiler_overhead_per_event_ns=overhead_ns,
        iter_wall_ns_override=iter_wall_ns,
    )
    summary = compute_stall_breakdown(sp, h2d_bw_Bps=h2d_Bps)
    print(summary.pretty(h2d_bw_Bps=h2d_Bps))
    print()
    print(f"  (StreamedProfile: {len(sp.launches)} launches, {len(sp.tid_to_size)} tids)")
    if overhead_ns or iter_wall_ns:
        print(f"  calibration: profiler_overhead_per_event_ns={overhead_ns}, "
              f"iter_wall_override={iter_wall_ns}")


if __name__ == "__main__":
    main()
