#!/usr/bin/env python3
"""Scheduler comparison at a fixed memory budget.

Each scheduler is described by a JSON file listing overrides — e.g.

    [
      {"label": "vanilla",   "overrides": ["scheduler.type=Vanilla"]},
      {"label": "flexinfer", "overrides": ["scheduler.type=FlexInfer",
                                            "scheduler.args.prefetch_window=5"]}
    ]

For each entry, runs cg-sim with the supplied overrides + any
common-overrides passed on the CLI, then collects metrics.

`summary.csv` carries the bar-chart payload directly:
`runtime_span_us`, `total_stall_us`, `attributed_stall_us`,
`unattributed_stall_us`, `transfer_busy_us`. `plot_time_breakdown.py`
consumes it.

Usage:
    python compare_schedulers.py <base_yaml> <compute_hw> <memory_hw>
        <schedulers_json> [experiment_name] [--with KEY=VAL ...]

Extra common overrides via repeated `--with KEY=VAL` apply to every
cell (e.g. fix the memory budget across schedulers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import (  # noqa: E402
    Cell,
    SweepResult,
    collect_metrics,
    default_sweep_dir,
    run_cell,
    write_summary,
)


def main(
    base_yaml: Path,
    compute_hw: str,
    memory_hw: str,
    schedulers_json: Path,
    experiment_name: str = "scheduler_compare",
    common_overrides: list[str] | None = None,
) -> None:
    common_overrides = list(common_overrides or [])
    spec = json.loads(Path(schedulers_json).read_text())
    if not isinstance(spec, list):
        raise SystemExit("schedulers_json must be a JSON array of {label,overrides}")

    sweep_dir = default_sweep_dir(experiment_name)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"experiment dir : {sweep_dir}")
    print(f"base config    : {base_yaml}")
    print(f"compute/memory : {compute_hw} / {memory_hw}")
    print(f"schedulers     : {[c['label'] for c in spec]}")
    if common_overrides:
        print(f"common -overrides: {common_overrides}")

    results: list[SweepResult] = []
    for entry in spec:
        label = entry["label"]
        per_cell_overrides = list(entry.get("overrides", []))
        cell = Cell(
            label=label,
            params={
                "scheduler": label,
                "overrides": " ".join(per_cell_overrides),
            },
            overrides=common_overrides + per_cell_overrides,
        )
        print(f"\n[cell {label}] launching cg-sim …")
        try:
            cell_dir, log_path = run_cell(base_yaml, cell, sweep_dir)
        except Exception as e:
            print(f"[cell {label}] FAILED to run: {e}")
            results.append(
                SweepResult(cell=cell, log_path=Path(), cell_dir=sweep_dir / label,
                            metrics={"status": "run_failed", "error": str(e)})
            )
            continue
        print(f"[cell {label}] log → {log_path}")

        try:
            metrics = collect_metrics(log_path, cell_dir, compute_hw, memory_hw)
        except Exception as e:
            print(f"[cell {label}] analyses FAILED: {e}")
            metrics = {"status": "analysis_failed", "error": str(e)}
        else:
            metrics["status"] = "ok"

        results.append(
            SweepResult(cell=cell, log_path=log_path, cell_dir=cell_dir, metrics=metrics)
        )

    summary = write_summary(
        sweep_dir, results, param_keys=["scheduler", "overrides"]
    )
    print(f"\nSweep complete. summary: {summary}")
    print(f"Plot with: python scripts/visualization/plot_time_breakdown.py {sweep_dir}")


def _parse_with(argv: list[str]) -> tuple[list[str], list[str]]:
    """Strip repeated `--with K=V` from argv, return (rest, withs)."""
    rest: list[str] = []
    withs: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--with" and i + 1 < len(argv):
            withs.append(argv[i + 1])
            i += 2
            continue
        if a.startswith("--with="):
            withs.append(a.split("=", 1)[1])
            i += 1
            continue
        rest.append(a)
        i += 1
    return rest, withs


if __name__ == "__main__":
    argv, withs = _parse_with(sys.argv[1:])
    if len(argv) not in (4, 5):
        print(
            "Usage: python compare_schedulers.py <base_yaml> <compute_hw> "
            "<memory_hw> <schedulers_json> [experiment_name] "
            "[--with KEY=VAL ...]"
        )
        sys.exit(2)
    base = Path(argv[0])
    compute = argv[1]
    mem = argv[2]
    spec = Path(argv[3])
    name = argv[4] if len(argv) == 5 else "scheduler_compare"
    main(base, compute, mem, spec, name, common_overrides=withs)
