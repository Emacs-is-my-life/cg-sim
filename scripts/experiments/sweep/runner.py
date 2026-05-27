"""Subprocess driver for cg-sim sweeps.

A "cell" is one (overrides, label) pair: a single simulation run with a
chosen set of Hydra overrides. The runner invokes `main.py` for each
cell, points the simulator's `logger.args.result_path` into the
experiment's `sim_results/` directory, then collects key metrics by
importing the analysis helpers (so the on-disk side-effect of `--out`
is reused, not re-implemented).

Directory layout produced by a sweep — see AGENTS.md "Output directory
conventions" for the canonical spec:

    output/<experiment-setup>/
        experiment.yaml          # manifest (base config, overrides, git SHA, …)
        summary.csv              # one row per cell, joined metrics
        sim_results/
            <run_id>.json        # raw cg-sim trace
            <run_id>.command.txt
            <run_id>.stdout.log
            <run_id>.stderr.log
        analysis/
            <run_id>/
                prefetch_quality/
                link_utilization/
        plots/                   # populated by scripts/visualization/*

The runner produces the `sim_results/` + `analysis/` + `summary.csv` +
`experiment.yaml` triplet; the visualization scripts fill `plots/`.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_OUTPUT = REPO_ROOT / "output"


# Make scripts/analysis importable so we can call analysis helpers directly.
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))


def default_sweep_dir(experiment_name: str) -> Path:
    """`output/<experiment-setup>/` — the canonical artifact root."""
    return REPO_OUTPUT / experiment_name


def sim_results_dir(experiment_dir: Path) -> Path:
    return Path(experiment_dir) / "sim_results"


def analysis_dir(experiment_dir: Path, run_id: str) -> Path:
    return Path(experiment_dir) / "analysis" / run_id


def plots_dir(experiment_dir: Path) -> Path:
    return Path(experiment_dir) / "plots"


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
    cell_analysis_dir: Path
    metrics: dict[str, Any]


def run_cell(
    base_yaml: Path,
    cell: Cell,
    sweep_dir: Path,
    *,
    timeout_s: float | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Run cg-sim for one cell. Returns (cell_analysis_dir, result_log_path).

    The result log is forced into `<sweep_dir>/sim_results/<run_id>.json`
    via a Hydra override on `logger.args.result_path`. Provenance
    sidecars (`.command.txt`, `.stdout.log`, `.stderr.log`) share the
    same `<run_id>` prefix so a tab-complete shows the full set.
    Per-cell analyses land in `<sweep_dir>/analysis/<run_id>/`.
    """
    sweep_dir = Path(sweep_dir)
    results_dir = sim_results_dir(sweep_dir)
    cell_analysis_dir = analysis_dir(sweep_dir, cell.label)
    results_dir.mkdir(parents=True, exist_ok=True)
    cell_analysis_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / f"{cell.label}.json"
    overrides = list(cell.overrides) + [
        f"logger.args.result_path={log_path}",
    ]
    cmd = ["python3", str(REPO_ROOT / "main.py"), "-i", str(base_yaml), *overrides]

    env = None
    if extra_env:
        import os
        env = {**os.environ, **extra_env}

    stdout_path = results_dir / f"{cell.label}.stdout.log"
    stderr_path = results_dir / f"{cell.label}.stderr.log"
    with stdout_path.open("w") as out_f, stderr_path.open("w") as err_f:
        subprocess.run(
            cmd, stdout=out_f, stderr=err_f,
            cwd=REPO_ROOT, env=env, timeout=timeout_s, check=True,
        )

    if not log_path.exists():
        raise RuntimeError(
            f"cg-sim finished but {log_path} does not exist; "
            f"see {stderr_path}"
        )

    (results_dir / f"{cell.label}.command.txt").write_text(" ".join(cmd) + "\n")
    return cell_analysis_dir, log_path


def collect_metrics(
    log_path: Path,
    cell_analysis_dir: Path,
    compute_hw: str,
    memory_hw: str,
    *,
    run_prefetch_quality: bool = True,
    run_link_utilization: bool = True,
) -> dict[str, Any]:
    """Run selected analyses on the cell's log and return key metrics.

    Each analysis is asked to write its full CSV+meta side-output into
    `<cell_analysis_dir>/<analysis>/`. We then re-read the analysis's
    meta.json to pull aggregate numbers into `summary.csv`.
    """
    metrics: dict[str, Any] = {}

    if run_prefetch_quality:
        from prefetch_quality import main as pq_main  # type: ignore[import-not-found]

        out_dir = cell_analysis_dir / "prefetch_quality"
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
            unique_tensors_accessed=meta["unique_tensors_accessed"],
        )

    if run_link_utilization:
        from link_utilization import main as lu_main  # type: ignore[import-not-found]

        out_dir = cell_analysis_dir / "link_utilization"
        _silently(lu_main, log_path, memory_hw, out_dir=out_dir)
        meta = json.loads((out_dir / "meta.json").read_text())
        metrics.update(
            transfer_busy_us=meta["busy_us"],
            transfer_total_KB=meta["total_size_KB"],
            achieved_aggregate_KBps=meta["achieved_aggregate_KBps"],
            num_transfers=meta["num_transfers"],
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
    across cells (union, missing → blank). Also ensures the `plots/`
    subdir exists so viz scripts have a tidy default home.
    """
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    plots_dir(sweep_dir).mkdir(parents=True, exist_ok=True)

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


def write_manifest(
    sweep_dir: Path,
    *,
    experiment_name: str,
    base_yaml: Path,
    cells: Sequence[Cell],
    description: str = "",
    common_overrides: Iterable[str] = (),
) -> Path:
    """Write a `experiment.yaml` manifest at the experiment root.

    Captures everything needed to re-run the sweep: base config,
    common overrides applied to every cell, per-cell overrides,
    timestamp (UTC), git SHA. Plain YAML rather than JSON so a
    human can `cat` it.
    """
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"name: {experiment_name}")
    if description:
        lines.append("description: |")
        for ln in description.splitlines() or [description]:
            lines.append(f"  {ln}")
    lines.append(f"created_at: {_dt.datetime.now(_dt.timezone.utc).isoformat()}")
    sha = _git_sha()
    if sha:
        lines.append(f"git_sha: {sha}")
    lines.append(f"base_config: {base_yaml}")
    common = list(common_overrides)
    if common:
        lines.append("common_overrides:")
        for ov in common:
            lines.append(f"  - {ov}")
    lines.append("runs:")
    for c in cells:
        lines.append(f"  - run_id: {c.label}")
        if c.overrides:
            lines.append("    overrides:")
            for ov in c.overrides:
                lines.append(f"      - {ov}")

    p = sweep_dir / "experiment.yaml"
    p.write_text("\n".join(lines) + "\n")
    return p


def update_latest_symlink(sweep_dir: Path) -> None:
    """Repoint `output/latest` at the most recent experiment dir.

    Best-effort: silently no-ops if the filesystem rejects the
    symlink (e.g. Windows without admin).
    """
    link = REPO_OUTPUT / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(sweep_dir).name, target_is_directory=True)
    except OSError:
        pass


def _git_sha() -> str | None:
    """Short git SHA of `HEAD`, or `None` if not in a repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None
