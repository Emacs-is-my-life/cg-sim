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
    axis_label,
    best_time_scale,
    color_for,
    display_name,
    minsec_formatter,
    parse_out_path_flag,
    read_table,
    unit_of,
)


def _is_sweep_dir(p: Path) -> bool:
    return (p / "summary.csv").exists()


def _iter_sweep_cell_analysis_dirs(sweep_dir: Path) -> list[tuple[str, Path]]:
    """Return `(run_id, analysis_dir)` pairs for each cell of a sweep.

    New layout: `<sweep_dir>/analysis/<run_id>/`. Falls back to the
    legacy layout `<sweep_dir>/<run_id>/` if `analysis/` is missing,
    so old experiment dirs keep rendering.
    """
    new = sweep_dir / "analysis"
    if new.is_dir():
        return [(c.name, c) for c in sorted(new.iterdir()) if c.is_dir()]
    return [
        (c.name, c) for c in sorted(sweep_dir.iterdir())
        if c.is_dir() and c.name not in {"sim_results", "plots", "analysis"}
    ]


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
        for run_id, cell_analysis in _iter_sweep_cell_analysis_dirs(in_dir):
            try:
                df = (
                    read_table(cell_analysis / Path(table).parent, Path(table).name)
                    if "/" in table else read_table(cell_analysis, table)
                )
            except FileNotFoundError:
                continue
            if column not in df.columns:
                continue
            series.append((run_id, df[column]))
        title = f"{in_dir.name}: CDF of {display_name(column)} from {table}"
    else:
        df = read_table(in_dir, table)
        if column not in df.columns:
            raise SystemExit(
                f"{table}.csv has no column {column!r}; "
                f"columns: {list(df.columns)}"
            )
        series.append((in_dir.name, df[column]))
        title = f"{in_dir.name}: CDF of {display_name(column)} from {table}"

    if not series:
        raise SystemExit(f"no data found for column {column!r} in any cell")

    # Auto-scale microsecond-valued columns so the X axis reads "10 s"
    # instead of "1e7 us". Other units pass through unscaled.
    col_unit = unit_of(column)
    divisor = 1.0
    use_minsec = False
    if col_unit == "us":
        import numpy as np
        max_abs = max(
            float(np.nanmax(np.abs(np.asarray(s, dtype=float))))
            for _, s in series
            if len(s) > 0
        )
        divisor, display_unit = best_time_scale(max_abs)
        x_label = f"{display_name(column)} [{display_unit}]"
        use_minsec = display_unit == "s" and (max_abs / 1_000_000.0) >= 60.0
    else:
        x_label = axis_label(column)

    fig, ax = plt.subplots()
    for i, (label, vals) in enumerate(series):
        x, y = _cdf_xy(vals)
        if len(x) == 0:
            continue
        ax.plot(x / divisor, y, color=color_for(i), label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("CDF")
    if use_minsec:
        ax.xaxis.set_major_formatter(minsec_formatter())
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
