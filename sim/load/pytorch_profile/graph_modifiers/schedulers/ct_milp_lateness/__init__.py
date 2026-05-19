"""ct_milp_lateness — pool-first MILP that minimizes compute lateness.

Optimization domain is a flat cgsim_tid pool sourced from the runtime
trace (no per-graph compile-time-tensor-identity layer). Objective is
the max consumer lateness (∝ end-to-end stall) subject to a hard peak
VRAM cap. See ``scheduler.py`` for the formulation and ``main.py`` for
the CLI.
"""

from sim.load.pytorch_profile.graph_modifiers.schedulers.ct_milp_lateness.scheduler import (
    print_summary,
    solve_neutral,
)

__all__ = ["print_summary", "solve_neutral"]
