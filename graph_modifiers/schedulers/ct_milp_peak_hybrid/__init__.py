"""ct_milp_peak_hybrid — minimize peak VRAM under zero LP-stall, hybrid mode.

Drops the `c + e = 1` coupling of ct_milp_peak so a tid can be cold at
layout *and* evict mid-run + refetch (hybrid pattern). Unlocks
mid-timeline VRAM reclamation for early-and-late-used tensors that
the non-hybrid variants pin to full residency.

⚠ The injector's coverage_repair pass does NOT yet recognize cold-start
residency as a gate for consumers before a mid-run evict — so hybrid
plans may sim with peak > LP modeled. Verify before relying on the
plan; see scheduler.py docstring for details.
"""

from graph_modifiers.schedulers.ct_milp_peak_hybrid.scheduler import (
    print_summary,
    solve_neutral,
)

__all__ = ["print_summary", "solve_neutral"]
