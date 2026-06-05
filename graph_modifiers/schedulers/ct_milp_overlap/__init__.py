"""ct_milp_overlap — overlap-aware weight-streaming MILP.

Forked from ct_milp_lateness_simtime. Same c/e decision variables and
peak-VRAM model, but the lateness model is replaced by a SINGLE SERIAL
H2D CHANNEL SCHEDULE with release times: each prefetch/refetch is a job
with processing p=δ, release r=max(prior_use_end, deadline−W), deadline
= consumer start, sequenced EDF on one machine. The objective minimizes
L = max channel lateness (= makespan extension).

Why: the per-window throughput lateness of the parent had no release
time, so it assumed unlimited free overlap and priced a prefetched read
identically to a synchronous one — prefetch was invisible to the
optimizer, and the plan behaved like sync reads (losing to a no-prefetch
Belady baseline). The channel model VALUES overlap: a job with wide
[r, d] slack is free; only transfers the channel can't deliver in time
extend makespan. The lookahead horizon W ties the overlap benefit to its
peak cost (the peak rows charge W of residency before each consumer) and
the emit issues ~W ahead (not JIT) so sim realizes the overlap.

CLI: ``--baseline-sim-result PATH`` (sim-time deadlines) and
``--lookahead-ms`` (W, the overlap↔peak knob).
"""

from graph_modifiers.schedulers.ct_milp_overlap.scheduler import (
    print_summary,
    solve_neutral,
)

__all__ = ["print_summary", "solve_neutral"]
