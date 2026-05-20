#!/usr/bin/env python3
"""Stacked bar of where time goes, one bar per cell.

Reads `<sweep_dir>/summary.csv`. Each bar is one cell (one scheduler,
one memory budget, …). Bar height = `runtime_span_us`. Stack
components = `compute_busy = runtime - total_stall`, then
`attributed_stall_us`, then `unattributed_stall_us`. This makes "did my
scheduler shrink the stall portion" visible at a glance.

Usage:
    python plot_time_breakdown.py <sweep_dir>
        [label_col]  [--out PATH]

`label_col` defaults to `cell_label`. Pass `scheduler` or `memory_GB`
or whatever swept axis you want on the x-axis.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    STACK_COMPONENT_LABELS,
    apply_matplotlib_defaults,
    axis_label,
    best_time_scale,
    color_for,
    load_summary,
    minsec_formatter,
    parse_out_path_flag,
)


REQUIRED = ("runtime_span_us", "total_stall_us", "attributed_stall_us",
            "unattributed_stall_us")


def main(
    in_dir: Path,
    label_col: str = "cell_label",
    *,
    out_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    apply_matplotlib_defaults()
    df = load_summary(in_dir)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"summary.csv is missing {missing}")
    if label_col not in df.columns:
        raise SystemExit(
            f"summary.csv has no column {label_col!r}; "
            f"columns: {list(df.columns)}"
        )
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    df = df.copy()
    df["compute_busy_us"] = df["runtime_span_us"] - df["total_stall_us"]
    compute = df["compute_busy_us"].clip(lower=0)
    # `attributed_stall_us` = gap blamed on an in-flight transfer (= data
    # stall). `unattributed_stall_us` = gap with no transfer to blame (=
    # idle). The CSV keeps the analysis-side jargon; the plot uses the
    # plain-language names from STACK_COMPONENT_LABELS.
    data_stall = df["attributed_stall_us"].clip(lower=0)
    idle = df["unattributed_stall_us"].clip(lower=0)
    labels = df[label_col].astype(str).tolist()

    # Auto-scale microseconds → ms/s so the axis reads "10" not "1e7".
    max_us = float((compute + data_stall + idle).max())
    divisor, unit = best_time_scale(max_us)
    use_minsec = unit == "s" and (max_us / 1_000_000.0) >= 60.0
    compute = compute / divisor
    data_stall = data_stall / divisor
    idle = idle / divisor

    fig, ax = plt.subplots()
    x = list(range(len(labels)))
    ax.bar(x, compute, color=color_for(0),
           label=STACK_COMPONENT_LABELS["compute"])
    ax.bar(x, data_stall, bottom=compute, color=color_for(1),
           label=STACK_COMPONENT_LABELS["data_stall"])
    ax.bar(x, idle, bottom=compute + data_stall, color=color_for(2),
           label=STACK_COMPONENT_LABELS["idle"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel(axis_label(label_col))
    ax.set_ylabel(f"time [{unit}]")
    if use_minsec:
        ax.yaxis.set_major_formatter(minsec_formatter())
    ax.set_title(f"{Path(in_dir).name}: runtime breakdown")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    if out_path is None:
        out_path = in_dir / "plot_time_breakdown.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        print(
            "Usage: python plot_time_breakdown.py <sweep_dir> "
            "[label_col] [--out PATH]"
        )
        sys.exit(2)
    in_dir = Path(argv[0])
    argv_rest, out = parse_out_path_flag(argv[1:], __file__, in_dir)
    label = argv_rest[0] if argv_rest else "cell_label"
    main(in_dir, label, out_path=out)
