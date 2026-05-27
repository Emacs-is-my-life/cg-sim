"""I/O helpers for structured analysis output.

Per the README convention (`Convention for scripts/analysis/*.py`),
scripts that produce tabular results MAY write CSVs + a `meta.json` to
an `out_dir`. These helpers enforce the directory layout and column
naming so the standard is encoded, not just documented.

Directory layout per analysis run:

    <out_dir>/
        meta.json        # config: log_path, hw names, runtime_start_us, ...
        <table>.csv      # one file per logical table

Default `out_dir` (when CLI passes a bare `--out` with no value, or when
the script is asked to write but no path is supplied) is:

    tmp/analysis/<script_stem>/<log_path.stem>/
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_TMP = Path("tmp/analysis")


def default_out_dir(script_path: str | Path, log_path: str | Path) -> Path:
    """`tmp/analysis/<script_stem>/<log_path.stem>/` — convention default."""
    return REPO_TMP / Path(script_path).stem / Path(log_path).stem


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_meta(out_dir: Path, **fields: Any) -> Path:
    """Write a `meta.json` carrying run config. Paths are stringified.

    Callers SHOULD include at least `log_path`, `runtime_start_us`,
    `runtime_span_us`, and the hardware names the script filtered by.
    Extra fields are passed through.
    """
    out_dir = ensure_out_dir(out_dir)
    serialisable = {}
    for k, v in fields.items():
        if isinstance(v, Path):
            serialisable[k] = str(v)
        elif isinstance(v, (list, tuple)):
            serialisable[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            serialisable[k] = v
    p = out_dir / "meta.json"
    p.write_text(json.dumps(serialisable, indent=2, default=str) + "\n")
    return p


def write_table(
    out_dir: Path,
    name: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> Path:
    """Write `<out_dir>/<name>.csv` with the given header and rows.

    Column-name convention (see README):
      * Times in microseconds → `_us` suffix (`ts_us`, `dur_us`, ...).
      * Sizes → `size_KB`; rates → `rate_KBps`.
      * Identifiers → `node_id`, `tensor_id`, `hw_name`/`src_name`/`dest_name`.
    """
    out_dir = ensure_out_dir(out_dir)
    if not name.endswith(".csv"):
        name = f"{name}.csv"
    p = out_dir / name
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(columns))
        w.writerows(rows)
    return p


def parse_out_flag(
    argv: list[str], script_path: str | Path, log_path: str | Path
) -> tuple[list[str], Path | None]:
    """Strip `--out [PATH]` from argv and resolve it to a Path or None.

    Forms accepted (per README convention):
      * absent              → returns None (no disk output)
      * `--out`             → default_out_dir(script_path, log_path)
      * `--out PATH`        → Path(PATH)
      * `--out=PATH`        → Path(PATH)

    Returns (argv_without_flag, out_dir_or_None).
    """
    rest: list[str] = []
    out_dir: Path | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out_dir = Path(argv[i + 1])
                i += 2
            else:
                out_dir = default_out_dir(script_path, log_path)
                i += 1
            continue
        if a.startswith("--out="):
            out_dir = Path(a.split("=", 1)[1])
            i += 1
            continue
        rest.append(a)
        i += 1
    return rest, out_dir
