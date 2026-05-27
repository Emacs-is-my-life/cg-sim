"""Display labels and unit-aware axis formatting.

Visualization scripts deal with column names that are snake_case with
unit suffixes (`total_stall_us`, `transfer_total_KB`). Those make sense
in code; they make for bad plots. This module maps each known column
to a `(display_name, unit)` pair and provides helpers that turn that
into axis/legend labels.

When all metrics on a plot share a unit, callers can put the unit on
the axis (`"time [s]"`) and keep legends clean. When units are mixed,
the axis becomes `"Value"` and per-metric units are pushed into the
legend.

For time columns (suffix `_us`), `best_time_scale` picks the most
readable display unit so a 10-second value plots as `10 s`, not
`1e7 us`. Convert raw values by dividing by the returned `divisor`.
"""
from __future__ import annotations

# col_name → (display_name, unit_or_None)
COLUMN_LABELS: dict[str, tuple[str, str | None]] = {
    # Sweep parameters
    "memory_GB": ("Memory Size", "GB"),
    "prefetch_window": ("Prefetch Window", None),
    "scheduler": ("Scheduler", None),
    "cell_label": ("Cell", None),

    # Time metrics (μs internally)
    "total_stall_us":        ("total_stall_time", "us"),
    "attributed_stall_us":   ("data_stall_time",  "us"),
    "unattributed_stall_us": ("idle_time",        "us"),
    "transfer_busy_us":      ("transfer_time",    "us"),
    "runtime_span_us":       ("total_runtime",    "us"),
    "compute_busy_us":       ("compute_time",     "us"),

    # Data volume / bandwidth
    "transfer_total_KB":       ("total_transfer",         "KB"),
    "achieved_aggregate_KBps": ("avg_transfer_bandwidth", "KBps"),

    # Per-event CDF columns (analysis CSVs)
    "dur_us":        ("stall_duration",     "us"),
    "slack_us":      ("slack",              "us"),
    "xfer_us":       ("transfer_duration",  "us"),
    "queue_wait_us": ("queue_wait",         "us"),
    "head_wait_us":  ("head_wait",          "us"),
    "begin_us":      ("begin",              "us"),
    "end_us":        ("end",                "us"),

    # Counters
    "num_compute_jobs":  ("compute_jobs",  None),
    "num_transfer_jobs": ("transfer_jobs", None),
    "num_transfers":     ("transfers",     None),

}


# Stack-component labels for plot_time_breakdown (internal key → legend label).
# These are NOT raw column names — they're the abstract roles in the stack.
STACK_COMPONENT_LABELS: dict[str, str] = {
    "compute":     "compute",
    "data_stall":  "data stall",
    "idle":        "idle",
}


def display_name(col: str) -> str:
    """Friendly name for `col`. Falls back to the raw column."""
    info = COLUMN_LABELS.get(col)
    return info[0] if info else col


def unit_of(col: str) -> str | None:
    """Unit string for `col`, or `None` if unitless / unknown."""
    info = COLUMN_LABELS.get(col)
    return info[1] if info else None


def axis_label(col: str) -> str:
    """'Display Name [unit]' if a unit is known, else just 'Display Name'."""
    name = display_name(col)
    unit = unit_of(col)
    return f"{name} [{unit}]" if unit else name


def legend_label(col: str, *, include_unit: bool = False) -> str:
    """Legend entry: display name, optionally with bracketed unit."""
    return axis_label(col) if include_unit else display_name(col)


def shared_unit(cols: list[str]) -> str | None:
    """Common unit across columns if they all share one; `None` if mixed
    or any column is unknown."""
    units = {unit_of(c) for c in cols}
    if len(units) == 1:
        unit = units.pop()
        return unit
    return None


def best_time_scale(max_us: float) -> tuple[float, str]:
    """Pick a divisor + unit for converting microseconds to a readable scale.

    Returns `(divisor, unit_str)`. Divide raw μs values by `divisor` for
    display. Above 60 s we still emit `s` so callers can apply a min:sec
    tick formatter if they want; the divisor alone makes "1e7" go away.
    """
    if max_us < 1_000.0:
        return 1.0, "us"
    if max_us < 1_000_000.0:
        return 1_000.0, "ms"
    return 1_000_000.0, "s"


def minsec_formatter():
    """Return a matplotlib FuncFormatter that renders seconds as 'm:ss'.

    Use only when scaled values are already in seconds and the max is
    over a minute — otherwise plain seconds read fine.
    """
    from matplotlib.ticker import FuncFormatter

    def _fmt(x, _pos):
        if x < 0:
            return f"-{_fmt(-x, _pos)}"
        m = int(x // 60)
        s = x - 60 * m
        if m == 0:
            return f"{s:.1f}s"
        return f"{m}:{int(round(s)):02d}"

    return FuncFormatter(_fmt)
