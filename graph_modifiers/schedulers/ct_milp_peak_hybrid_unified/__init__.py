"""ct_milp_peak_hybrid_unified — hybrid + soft slack on the *unified* timeline.

Operates on UnifiedTimeline tensors (storage-coalesced) and uses
``tl.tasks[pos].start_ns`` (compacted gpu time, no trace idle gaps) as
the cumulative budget axis. Compiled-launch-level granularity means
aux/aten ops between compiled kernels become free PCIe-hiding budget,
not artificial deadlines the LP must schedule around.

See ``scheduler.py`` for the formulation; ``main.py`` for the CLI.
"""

from graph_modifiers.schedulers.ct_milp_peak_hybrid_unified.scheduler import (
    print_summary,
    solve_neutral,
)

__all__ = ["print_summary", "solve_neutral"]
