"""Shared helpers for `scripts/visualization/*.py`.

Split by feature:
  * style — palette, figure defaults, paper-friendly matplotlib config
  * io    — read meta.json, glob run dirs, load summary.csv, default
            `out_path` derivation per the README viz convention
"""
from .io import (
    default_viz_out_path,
    glob_run_dirs,
    load_summary,
    parse_out_path_flag,
    read_meta,
    read_table,
)
from .style import (
    PALETTE,
    apply_matplotlib_defaults,
    color_for,
)

__all__ = [
    "default_viz_out_path",
    "glob_run_dirs",
    "load_summary",
    "parse_out_path_flag",
    "read_meta",
    "read_table",
    "PALETTE",
    "apply_matplotlib_defaults",
    "color_for",
]
