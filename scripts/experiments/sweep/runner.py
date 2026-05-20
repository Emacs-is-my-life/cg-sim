"""Subprocess driver for cg-sim sweeps.

A "cell" is one (overrides, label) pair: a single simulation run with a
chosen set of Hydra overrides. The runner invokes `main.py` for each
cell, points the simulator's `logger.args.result_path` into the cell's
own directory, then collects key metrics by importing the analysis
helpers (so the on-disk side-effect of `--out` is reused, not
re-implemented).

Directory layout produced by a sweep:

    tmp/sweeps/<experiment>/
        summary.csv              # one row per cell, joined metrics
        <cell_label>/
            result_log.json      # raw cg-sim trace
            prefetch_quality/    # analysis side-output (--out)
            link_utilization/
            reuse_distance/
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_TMP_SWEEPS = REPO_ROOT / "tmp" / "sweeps"


# Make scripts/analysis importable so we can call analysis helpers directly.
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))


def default_sweep_dir(experiment_name: str) -> Path:
    """`tmp/sweeps/<experiment>/` — convention default."""
    return REPO_TMP_SWEEPS / experiment_name


@dataclass(slots=True)
class Cell:
    """One simulation run in a sweep.

    `label` becomes the cell's subdirectory name; keep it filesystem-safe.
    `params` is what gets written into `summary.csv` (the swept axis
    values). `overrides` are Hydra-style strings handed to `main.py`.
    """

    label: str
    params: dict[str, Any]
    overrides: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SweepResult:
    cell: Cell
    log_path: Path
    cell_dir: Path
    metrics: dict[str, Any]


def _result_log_path(cell_dir: Path) -> Path:
    return cell_dir / "result_log.json"


def run_cell(
    base_yaml: Path,
    cell: Cell,
    sweep_dir: Path,
    *,
    timeout_s: float | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Run cg-sim for one cell. Returns (cell_dir, result_log_path).

    The result log is forced into the cell directory via a Hydra
    override on `logger.args.result_path`, so cells never collide on
    the path baked into the base YAML.
    """
    cell_dir = sweep_dir / cell.label
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_path = _result_log_path(cell_dir)

    overrides = list(cell.overrides) + [
        f"logger.args.result_path={log_path}",
    ]
    cmd = ["python3", str(REPO_ROOT / "main.py"), "-i", str(base_yaml), *overrides]

    env = None
    if extra_env:
        import os
        env = {**os.environ, **extra_env}

    with (cell_dir / "stdout.log").open("w") as out_f, \
         (cell_dir / "stderr.log").open("w") as err_f:
        subprocess.run(
            cmd, stdout=out_f, stderr=err_f,
            cwd=REPO_ROOT, env=env, timeout=timeout_s, check=True,
        )

    if not log_path.exists():
        raise RuntimeError(
            f"cg-sim finished but {log_path} does not exist; "
            f"see {cell_dir / 'stderr.log'}"
        )

    # Stash the actual command + overrides for later forensics.
    (cell_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    return cell_dir, log_path


def collect_metrics(
    log_path: Path,
    cell_dir: Path,
    compute_hw: str,
    memory_hw: str,
    *,
    run_prefetch_quality: bool = True,
    run_link_utilization: bool = True,
    run_reuse_distance: bool = False,
) -> dict[str, Any]:
    """Run selected analyses on the cell's log and return key metrics.

    Each analysis is asked to write its full CSV+meta side-output into
    `<cell_dir>/<analysis>/`. We then re-read the analysis's meta.json
    to pull aggregate numbers into `summary.csv`.
    """
    metrics: dict[str, Any] = {}

    if run_prefetch_quality:
        from prefetch_quality import main as pq_main  # type: ignore[import-not-found]

        out_dir = cell_dir / "prefetch_quality"
        # Silence the stdout summary in sweep mode — it's noisy across N cells.
        # Each cell's per-analysis CSVs are still written.
        _silently(pq_main, log_path, compute_hw, memory_hw, out_dir=out_dir)
        meta = json.loads((out_dir / "meta.json").read_text())
        metrics.update(
            runtime_span_us=meta["runtime_span_us"],
            total_stall_us=meta["total_stall_us"],
            attributed_stall_us=meta["attributed_stall_us"],
            unattributed_stall_us=meta["unattributed_stall_us"],
            num_compute_jobs=meta["num_compute_jobs"],
            num_transfer_jobs=meta["num_transfer_jobs"],
        )

    if run_link_utilization:
        from link_utilization import main as lu_main  # type: ignore[import-not-found]

        out_dir = cell_dir / "link_utilization"
        _silently(lu_main, log_path, memory_hw, out_dir=out_dir)
        meta = json.loads((out_dir / "meta.json").read_text())
        metrics.update(
            transfer_busy_us=meta["busy_us"],
            transfer_total_KB=meta["total_size_KB"],
            achieved_aggregate_KBps=meta["achieved_aggregate_KBps"],
            num_transfers=meta["num_transfers"],
        )

    if run_reuse_distance:
        from reuse_distance import main as rd_main  # type: ignore[import-not-found]

        out_dir = cell_dir / "reuse_distance"
        _silently(rd_main, log_path, memory_hw, out_dir=out_dir)
        meta = json.loads((out_dir / "meta.json").read_text())
        metrics.update(
            unique_tensors=meta["unique_tensors"],
            total_access_KB=meta["total_KB"],
        )

    return metrics


def _silently(fn, *args, **kwargs) -> None:
    """Call `fn` with stdout swallowed; reraise on error."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)


def write_summary(
    sweep_dir: Path,
    results: Sequence[SweepResult],
    param_keys: Iterable[str],
) -> Path:
    """Write `summary.csv` joining swept params + collected metrics.

    Columns: `cell_label`, *param_keys, then every metric key seen
    across cells (union, missing → blank).
    """
    sweep_dir.mkdir(parents=True, exist_ok=True)
    param_keys = list(param_keys)
    metric_keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r.metrics:
            if k not in seen:
                seen.add(k)
                metric_keys.append(k)

    p = sweep_dir / "summary.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_label", *param_keys, *metric_keys])
        for r in results:
            row: list[Any] = [r.cell.label]
            for k in param_keys:
                row.append(r.cell.params.get(k, ""))
            for k in metric_keys:
                row.append(r.metrics.get(k, ""))
            w.writerow(row)
    return p
