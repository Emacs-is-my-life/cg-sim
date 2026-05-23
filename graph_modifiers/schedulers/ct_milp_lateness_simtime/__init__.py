"""ct_milp_lateness_simtime — lateness LP on sim-trace deadlines.

Same formulation as ct_milp_lateness (hybridized — c, e independent;
soft peak cap + per-window lateness slack), but the LP's time axis
is per-trace-node *sim wall-clock* harvested from a baseline
sim_result.json (option #1 of the wall-axis discussion). This is the
ground-truth deadline a prefetch must beat in sim, avoiding both the
trace's profiler idle gaps and the over-approximation of pure
gpu-cumulative.

The emit still writes transfer_start_ns / end_ns in trace wall-clock
(what the injector expects); only the LP's internal feasibility,
issuer search, and per-window lateness budget switch to sim time.

CLI: pass ``--baseline-sim-result PATH`` to enable. Without it the
scheduler degrades to identical behavior as ct_milp_lateness.
"""

from graph_modifiers.schedulers.ct_milp_lateness_simtime.scheduler import (
    print_summary,
    solve_neutral,
)

__all__ = ["print_summary", "solve_neutral"]
