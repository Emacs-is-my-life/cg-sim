"""I/O helpers for visualization scripts.

Mirrors `scripts/analysis/common/io.py` from the other side: instead of
*writing* analysis output, these helpers *read* it. The directory
layout produced by analysis scripts (`<out_dir>/meta.json` + one
`<table>.csv` per table) and by sweep runners
(`<sweep_dir>/summary.csv` + one `<cell>/...` per cell) is the contract.
"""
from __future__ import annotations

import json
from pathlib import Path


def default_viz_out_path(script_path: str | Path, in_dir: str | Path) -> Path:
    """`<in_dir>/<script_stem>.png` — convention default."""
    return Path(in_dir) / f"{Path(script_path).stem}.png"


def read_meta(run_dir: Path) -> dict:
    """Load `<run_dir>/meta.json` as a dict (raises if missing)."""
    p = Path(run_dir) / "meta.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text())


def read_table(run_dir: Path, table_name: str) -> "pandas.DataFrame":  # type: ignore[name-defined]
    """Load `<run_dir>/<table_name>.csv` as a pandas DataFrame."""
    import pandas as pd

    name = table_name if table_name.endswith(".csv") else f"{table_name}.csv"
    p = Path(run_dir) / name
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return pd.read_csv(p)


def load_summary(sweep_dir: Path) -> "pandas.DataFrame":  # type: ignore[name-defined]
    """Load `<sweep_dir>/summary.csv` as a pandas DataFrame."""
    import pandas as pd

    p = Path(sweep_dir) / "summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return pd.read_csv(p)


def glob_run_dirs(sweep_dir: Path, subdir: str | None = None) -> list[Path]:
    """List cell directories inside a sweep dir.

    If `subdir` is given (e.g. "prefetch_quality"), returns the matching
    analysis subdirs across cells; otherwise returns cell roots.
    Cells are any direct child dir of `sweep_dir`.
    """
    sweep_dir = Path(sweep_dir)
    cells = sorted(p for p in sweep_dir.iterdir() if p.is_dir())
    if subdir is None:
        return cells
    return [c / subdir for c in cells if (c / subdir).is_dir()]


def parse_out_path_flag(
    argv: list[str], script_path: str | Path, in_dir: str | Path
) -> tuple[list[str], Path]:
    """Strip `--out [PATH]` from argv and resolve to a Path.

    Mirrors the analysis-side `parse_out_flag` but returns a single
    file path (not a directory). Defaults to
    `default_viz_out_path(script_path, in_dir)`.
    """
    rest: list[str] = []
    out_path: Path | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out_path = Path(argv[i + 1])
                i += 2
            else:
                out_path = default_viz_out_path(script_path, in_dir)
                i += 1
            continue
        if a.startswith("--out="):
            out_path = Path(a.split("=", 1)[1])
            i += 1
            continue
        rest.append(a)
        i += 1
    if out_path is None:
        out_path = default_viz_out_path(script_path, in_dir)
    return rest, out_path
