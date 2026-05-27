#!/usr/bin/env python3
"""Generic line plot: one or more metrics vs a swept parameter.

Reads `<sweep_dir>/summary.csv` (written by `scripts/experiments/*.py`)
and produces a line plot of selected metric column(s) against a chosen
parameter column. The classic memory-sweep "stall vs memory" figure is
one invocation; bandwidth/lead-time sweeps share the same code path.

Usage:
    python plot_metric_vs_param.py <sweep_dir> <param_col>
        <metric_col[,metric_col...]> [--out PATH]

Example:
    python plot_metric_vs_param.py tmp/sweeps/memory_sweep \\
        memory_GB total_stall_us,transfer_busy_us,runtime_span_us
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    apply_matplotlib_defaults,
    axis_label,
    best_time_scale,
    color_for,
    display_name,
    legend_label,
    load_summary,
    minsec_formatter,
    parse_out_path_flag,
    shared_unit,
)


def main(
    in_dir: Path,
    param_col: str,
    metric_cols: list[str],
    *,
    out_path: Path | None = None,
    log_y: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    apply_matplotlib_defaults()

    df = load_summary(in_dir)
    if param_col not in df.columns:
        raise SystemExit(f"summary.csv has no column {param_col!r}; "
                         f"columns: {list(df.columns)}")
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"summary.csv is missing metric column(s) {missing}")

    if "status" in df.columns:
        ok = df["status"] == "ok"
        if (~ok).any():
            print(f"  dropping {(~ok).sum()} non-ok cell(s)")
        df = df[ok]

    df = df.sort_values(param_col)

    # Choose Y-axis treatment. If all metrics share a unit we put it on
    # the axis and keep the legend clean; if they're mixed (e.g. KB +
    # KBps) the axis becomes "Value" and units move into the legend.
    # For time (us) we also auto-scale so values plot as ms or s, not 1e7.
    common_unit = shared_unit(metric_cols)
    divisor = 1.0
    if common_unit == "us":
        max_us = float(df[metric_cols].to_numpy().max())
        divisor, display_unit = best_time_scale(max_us)
        y_label = f"time [{display_unit}]"
        use_minsec = display_unit == "s" and (max_us / 1_000_000.0) >= 60.0
    elif common_unit is not None:
        y_label = f"[{common_unit}]"
        use_minsec = False
    else:
        y_label = "Value"
        use_minsec = False

    fig, ax = plt.subplots()
    for i, col in enumerate(metric_cols):
        y_vals = df[col] / divisor if common_unit == "us" else df[col]
        label = legend_label(col, include_unit=(common_unit is None))
        ax.plot(
            df[param_col],
            y_vals,
            marker="o",
            color=color_for(i),
            label=label,
        )
    ax.set_xlabel(axis_label(param_col))
    ax.set_ylabel(y_label)
    if use_minsec:
        ax.yaxis.set_major_formatter(minsec_formatter())
    if log_y:
        ax.set_yscale("log")
    if len(metric_cols) > 1:
        ax.legend(frameon=False)
    metric_names = ", ".join(display_name(c) for c in metric_cols)
    ax.set_title(
        f"{Path(in_dir).name}: {metric_names} vs {display_name(param_col)}"
    )

    if out_path is None:
        out_path = in_dir / "plot_metric_vs_param.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 3:
        print(
            "Usage: python plot_metric_vs_param.py <sweep_dir> <param_col> "
            "<metric_col[,metric_col...]> [--out PATH]"
        )
        sys.exit(2)
    in_dir = Path(argv[0])
    argv_rest, out = parse_out_path_flag(argv[1:], __file__, in_dir)
    if len(argv_rest) != 2:
        print(
            "Usage: python plot_metric_vs_param.py <sweep_dir> <param_col> "
            "<metric_col[,metric_col...]> [--out PATH]"
        )
        sys.exit(2)
    param = argv_rest[0]
    metrics = [m.strip() for m in argv_rest[1].split(",") if m.strip()]
    main(in_dir, param, metrics, out_path=out)
