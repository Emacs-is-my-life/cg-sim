"""ct_milp_peak — pool-first MILP that minimizes peak VRAM at zero stall.

Inverse of ct_milp_lateness: instead of minimizing stall under a peak
cap, this variant minimizes peak VRAM subject to a *hard* zero-stall
constraint (per-window PCIe load must fit inside the window's
wall-clock duration). Cold-all is always feasible, so the LP never
falls infeasible from the stall side.

See ``scheduler.py`` for the formulation and ``main.py`` for the CLI.
"""

from graph_modifiers.schedulers.ct_milp_peak.scheduler import (
    print_summary,
    solve_neutral,
)

__all__ = ["print_summary", "solve_neutral"]
