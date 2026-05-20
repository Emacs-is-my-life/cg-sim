"""Shared helpers for `scripts/experiments/*.py`.

Sweep runners share the same shape: produce a config matrix, run each
cell via subprocess, call the analysis helpers per cell, aggregate
into `output/<experiment-setup>/summary.csv`. See AGENTS.md "Output
directory conventions" for the full layout.
"""
from .runner import (
    Cell,
    SweepResult,
    analysis_dir,
    collect_metrics,
    default_sweep_dir,
    plots_dir,
    run_cell,
    sim_results_dir,
    update_latest_symlink,
    write_manifest,
    write_summary,
)

__all__ = [
    "Cell",
    "SweepResult",
    "analysis_dir",
    "collect_metrics",
    "default_sweep_dir",
    "plots_dir",
    "run_cell",
    "sim_results_dir",
    "update_latest_symlink",
    "write_manifest",
    "write_summary",
]
