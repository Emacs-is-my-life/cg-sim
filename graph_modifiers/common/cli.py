"""Shared CLI helpers for scheduler entry points.

Every scheduler main.py accepts the same calibration triplet:

* ``--iter-wall-ns`` / ``--iter-wall-ms`` — override the iter wall to a
  measured non-profiled value. The builder back-solves the per-CPU-event
  profiler overhead that produces that wall.
* ``--profiler-overhead-ns`` — set a fixed per-event overhead directly,
  overriding whatever is in the ``--hw`` JSON.

This module centralizes the flag registration + resolution so individual
schedulers don't hand-roll (and drift on) the same boilerplate.

Usage in a scheduler main::

    parser = argparse.ArgumentParser(...)
    ...
    add_calibration_args(parser)
    args = parser.parse_args()

    hw = load_hw_params(args.hw)
    iter_wall_ns = resolve_calibration(args, hw)

    schedule = solve(..., hw=hw, iter_wall_ns_override=iter_wall_ns)
"""

from __future__ import annotations

import argparse

from .hw import HwParams


def add_calibration_args(parser: argparse.ArgumentParser) -> None:
    """Register the three calibration flags on a parser."""
    group = parser.add_argument_group("profiler calibration")
    group.add_argument(
        "--iter-wall-ns", type=int, default=None,
        help=(
            "Override iter wall time (ns) — the builder back-solves the "
            "per-CPU-event profiler overhead that makes the simulator "
            "match. Preferred over --profiler-overhead-ns when a measured "
            "non-profiled iter wall is available."
        ),
    )
    group.add_argument(
        "--iter-wall-ms", type=float, default=None,
        help="Shortcut for --iter-wall-ns in milliseconds.",
    )
    group.add_argument(
        "--profiler-overhead-ns", type=int, default=None,
        help=(
            "Per-CPU-event profiler overhead (ns) to subtract from each "
            "CPU node duration. Overrides HwParams.profiler_overhead_per_event_ns."
        ),
    )


def resolve_calibration(args: argparse.Namespace, hw: HwParams) -> int | None:
    """Mutate ``hw.profiler_overhead_per_event_ns`` from --profiler-overhead-ns;
    return the resolved iter_wall_ns override (or None).

    When both --iter-wall-ns and --iter-wall-ms are supplied, --iter-wall-ns
    wins.
    """
    if getattr(args, "profiler_overhead_ns", None) is not None:
        hw.profiler_overhead_per_event_ns = int(args.profiler_overhead_ns)

    iter_wall_ns = getattr(args, "iter_wall_ns", None)
    if iter_wall_ns is None:
        ms = getattr(args, "iter_wall_ms", None)
        if ms is not None:
            iter_wall_ns = int(ms * 1e6)
    return iter_wall_ns
