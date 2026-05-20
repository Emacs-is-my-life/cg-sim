"""Shared helpers for `scripts/experiments/*.py`.

Sweep runners share the same shape: produce a config matrix, run each
cell via subprocess, call the analysis helpers per cell, aggregate into
`tmp/sweeps/<experiment>/summary.csv`.
"""
from .runner import (
    Cell,
    SweepResult,
    collect_metrics,
    default_sweep_dir,
    run_cell,
    write_summary,
)

__all__ = [
    "Cell",
    "SweepResult",
    "collect_metrics",
    "default_sweep_dir",
    "run_cell",
    "write_summary",
]
