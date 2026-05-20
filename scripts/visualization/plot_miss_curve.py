#!/usr/bin/env python3
"""Mattson LRU miss-rate curve from `reuse_distance` output.

Reads `<in_dir>/miss_curve.csv` (count-weighted miss rate by capacity)
and plots miss rate vs capacity on a log-x axis. If a byte-weighted
column (`miss_KB_frac`) is present, plots both.

If `in_dir` is a sweep dir, overlays one curve per cell using each
cell's `reuse_distance/miss_curve.csv`.

Usage:
    python plot_miss_curve.py <dir> [--byte-weighted] [--out PATH]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    apply_matplotlib_defaults,
    color_for,
    parse_out_path_flag,
    read_table,
)


def _is_sweep_dir(p: Path) -> bool:
    return (p / "summary.csv").exists()


def main(
    in_dir: Path,
    *,
    out_path: Path | None = None,
    byte_weighted: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    apply_matplotlib_defaults()
    series: list[tuple[str, "pandas.DataFrame"]] = []  # type: ignore[name-defined]
    in_dir = Path(in_dir)
    if _is_sweep_dir(in_dir):
        for cell in sorted(p for p in in_dir.iterdir() if p.is_dir()):
            try:
                df = read_table(cell / "reuse_distance", "miss_curve")
            except FileNotFoundError:
                continue
            series.append((cell.name, df))
    else:
        df = read_table(in_dir, "miss_curve")
        series.append((in_dir.name, df))

    if not series:
        raise SystemExit("no miss_curve.csv data found")

    y_col = "miss_KB_frac" if byte_weighted else "miss_rate"
    fig, ax = plt.subplots()
    for i, (label, df) in enumerate(series):
        ax.plot(
            df["capacity_tensors"],
            df[y_col],
            marker="o",
            color=color_for(i),
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("LRU capacity (tensors)")
    ax.set_ylabel("byte-weighted miss rate" if byte_weighted else "miss rate")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{in_dir.name}: Mattson miss curve")
    if len(series) > 1:
        ax.legend(frameon=False)

    if out_path is None:
        out_path = in_dir / "plot_miss_curve.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    byte_weighted = False
    if "--byte-weighted" in argv:
        argv = [a for a in argv if a != "--byte-weighted"]
        byte_weighted = True
    if not argv:
        print(
            "Usage: python plot_miss_curve.py <dir> [--byte-weighted] [--out PATH]"
        )
        sys.exit(2)
    in_dir = Path(argv[0])
    rest, out = parse_out_path_flag(argv[1:], __file__, in_dir)
    main(in_dir, out_path=out, byte_weighted=byte_weighted)
