#!/usr/bin/env python3
"""CLI: convert a backend-neutral schedule (``schedule.json``) into the
PyTorch weight-streaming runtime schema (``jit_sim_prune_schedule.json``).

Usage::

    python -m graph_modifiers.convert_schedule.main \\
        --bundle /path/to/llama_bundle \\
        --neutral /path/to/schedule.json \\
        --output /path/to/jit_sim_prune_schedule.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CG_SIM_ROOT = THIS_DIR.parent.parent
if str(CG_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_SIM_ROOT))

from graph_modifiers.common import (
    build_unified_timeline,
    build_node_timeline,
    load_multi_graph_sidecars,
    load_neutral_schedule,
    load_trace_from_bundle,
    neutral_to_pytorch,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True,
                   help="Path to the llama_bundle directory (for trace+sidecars).")
    p.add_argument("--neutral", required=True,
                   help="Path to the neutral schedule.json")
    p.add_argument("--output", "-o", default=None,
                   help="Output path for jit_sim_prune_schedule.json "
                        "(default: alongside --neutral)")
    p.add_argument("--cpu-per-launch-ns", type=int, default=0,
                   help="CPU dispatch overhead per launch, used to "
                        "reconstruct the per-node timeline for the output "
                        "JSON's ``nodes`` section.")
    args = p.parse_args()

    trace = load_trace_from_bundle(args.bundle)
    sidecars = load_multi_graph_sidecars(args.bundle)
    neutral = load_neutral_schedule(args.neutral)

    tl = build_unified_timeline(
        trace, sidecars, cpu_per_launch_ns=args.cpu_per_launch_ns,
    )
    node_starts, node_ends = build_node_timeline(tl, trace)

    doc = neutral_to_pytorch(
        neutral, trace=trace,
        node_starts=node_starts, node_ends=node_ends,
    )

    out_path = (
        Path(args.output) if args.output
        else Path(args.neutral).parent / "jit_sim_prune_schedule.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"→ wrote PyTorch-format schedule: {out_path}")


if __name__ == "__main__":
    main()
