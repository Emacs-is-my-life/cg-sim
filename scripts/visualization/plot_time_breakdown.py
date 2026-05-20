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
    apply_matplotlib_defaults,
    color_for,
    load_summary,
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
    attr = df["attributed_stall_us"].clip(lower=0)
    unattr = df["unattributed_stall_us"].clip(lower=0)
    labels = df[label_col].astype(str).tolist()

    fig, ax = plt.subplots()
    x = list(range(len(labels)))
    ax.bar(x, compute, color=color_for(0), label="compute")
    ax.bar(x, attr, bottom=compute, color=color_for(1), label="stall (attributed)")
    ax.bar(
        x, unattr, bottom=compute + attr,
        color=color_for(2), label="stall (unattributed)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("time (us)")
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
