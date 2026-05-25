"""Shared CLI plumbing for the mode-specific hf_accelerate entry points.

Each mode (``sequential`` / ``module`` / ``model`` / ``module_hook`` /
``group``) constructs its own ``HFAccelerateKnobs`` with the values
that mirror the corresponding HF API and then hands them here for the
common trace-load / solve / write / summary path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CG_SIM_ROOT = THIS_DIR.parent.parent.parent
if str(CG_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_SIM_ROOT))

from graph_modifiers.common import load_trace_from_bundle
from graph_modifiers.schedulers.hf_accelerate.scheduler import (
    HFAccelerateKnobs, print_summary, solve, write_eager_schedule,
)


def base_parser(description: str) -> argparse.ArgumentParser:
    """Return an argparse parser preloaded with the shared options
    (``bundle`` positional, ``--output``, ``--keep``).
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "bundle",
        help="Path to the pytorch profile bundle (directory containing "
             "<bundle>/manifest.json).",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output schedule path. Default: "
             "<bundle>/hf_accelerate_output/schedule.json",
    )
    p.add_argument(
        "--keep", default="",
        help="Comma-separated module-path substrings to keep "
             "cuda-resident (cold-start, never offloaded). Not part of "
             "real HF — user-side experimental knob.",
    )
    return p


def keep_substrings(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(s.strip() for s in args.keep.split(",") if s.strip())


def run(args: argparse.Namespace, knobs: HFAccelerateKnobs) -> None:
    """Load the bundle, solve, write the schedule, print summary."""
    trace = load_trace_from_bundle(args.bundle)
    schedule = solve(trace, knobs)
    out_path = (
        Path(args.output) if args.output
        else Path(args.bundle) / "hf_accelerate_output" / "schedule.json"
    )
    write_eager_schedule(out_path, schedule)
    print(f"→ schedule saved: {out_path}")
    print_summary(schedule)
