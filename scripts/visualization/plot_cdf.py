#!/usr/bin/env python3
"""CDF plot of a numeric column from a single analysis CSV.

The natural use cases are stall-duration CDF
(`<dir>/stalls.csv`, column `dur_us`) and prefetch-slack CDF
(`<dir>/slack.csv`, column `slack_us`). Pass any
(analysis_dir, csv_name, column) combination.

If `in_dir` is a sweep dir (contains `summary.csv`) and `table` exists
inside each cell, overlays one CDF per cell. Otherwise treats `in_dir`
as a single analysis run.

Usage:
    python plot_cdf.py <dir> <table_name> <column>
        [--log-x] [--out PATH]

Example (single run):
    python plot_cdf.py tmp/analysis/prefetch_quality/python-eager-llama-8B \\
        stalls dur_us --log-x

Example (sweep overlay):
    python plot_cdf.py tmp/sweeps/scheduler_compare prefetch_quality/slack \\
        slack_us
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


def _cdf_xy(values):
    import numpy as np

    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    v.sort()
    if len(v) == 0:
        return v, v
    y = (1 + np.arange(len(v))) / len(v)
    return v, y


def main(
    in_dir: Path,
    table: str,
    column: str,
    *,
    out_path: Path | None = None,
    log_x: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    apply_matplotlib_defaults()

    series: list[tuple[str, "pandas.Series"]] = []  # type: ignore[name-defined]
    in_dir = Path(in_dir)
    if _is_sweep_dir(in_dir):
        cells = sorted(p for p in in_dir.iterdir() if p.is_dir())
        for c in cells:
            try:
                df = read_table(c / Path(table).parent, Path(table).name) \
                    if "/" in table else read_table(c, table)
            except FileNotFoundError:
                continue
            if column not in df.columns:
                continue
            series.append((c.name, df[column]))
        title = f"{in_dir.name}: CDF of {column} from {table}"
    else:
        df = read_table(in_dir, table)
        if column not in df.columns:
            raise SystemExit(
                f"{table}.csv has no column {column!r}; "
                f"columns: {list(df.columns)}"
            )
        series.append((in_dir.name, df[column]))
        title = f"{in_dir.name}: CDF of {column} from {table}"

    if not series:
        raise SystemExit(f"no data found for column {column!r} in any cell")

    fig, ax = plt.subplots()
    for i, (label, vals) in enumerate(series):
        x, y = _cdf_xy(vals)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=color_for(i), label=label)
    ax.set_xlabel(column)
    ax.set_ylabel("CDF")
    if log_x:
        ax.set_xscale("symlog", linthresh=1.0)
    if len(series) > 1:
        ax.legend(frameon=False)
    ax.set_title(title)
    ax.set_ylim(0, 1.02)

    if out_path is None:
        out_path = in_dir / "plot_cdf.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    # `--log-x` is a script-specific flag handled before parse_out_path_flag.
    log_x = False
    if "--log-x" in argv:
        argv = [a for a in argv if a != "--log-x"]
        log_x = True
    if len(argv) < 3:
        print(
            "Usage: python plot_cdf.py <dir> <table_name> <column> "
            "[--log-x] [--out PATH]"
        )
        sys.exit(2)
    in_dir = Path(argv[0])
    rest, out = parse_out_path_flag(argv[1:], __file__, in_dir)
    if len(rest) != 2:
        print(
            "Usage: python plot_cdf.py <dir> <table_name> <column> "
            "[--log-x] [--out PATH]"
        )
        sys.exit(2)
    main(in_dir, rest[0], rest[1], out_path=out, log_x=log_x)
