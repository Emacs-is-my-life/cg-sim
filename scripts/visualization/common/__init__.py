"""Shared helpers for `scripts/visualization/*.py`.

Split by feature:
  * style  — palette, figure defaults, paper-friendly matplotlib config
  * io     — read meta.json, glob run dirs, load summary.csv, default
             `out_path` derivation per the README viz convention
  * labels — column → display-name/unit map and time auto-scaling, so
             plots show "Memory Size [GB]" instead of "memory_GB" and
             "10 s" instead of "1e7 us"
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
from .labels import (
    STACK_COMPONENT_LABELS,
    axis_label,
    best_time_scale,
    display_name,
    legend_label,
    minsec_formatter,
    shared_unit,
    unit_of,
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
    "STACK_COMPONENT_LABELS",
    "axis_label",
    "best_time_scale",
    "display_name",
    "legend_label",
    "minsec_formatter",
    "shared_unit",
    "unit_of",
]
