#!/usr/bin/env python3
"""Memory-budget sweep: vary the offload-tier capacity, hold everything else.

For each memory size in `sizes_GB`, runs cg-sim with the override

    hardware.memory.<memory_idx>.args.memory_size_KB=<size_KB>

records the result log under `tmp/sweeps/<experiment>/<size>GB/`, runs
the `prefetch_quality` and `link_utilization` analyses, and aggregates
per-cell metrics into `summary.csv`.

The `summary.csv` schema makes the canonical "stall vs memory" plot a
one-liner via `scripts/visualization/plot_metric_vs_param.py`:

    cell_label, memory_GB, scheduler, runtime_span_us, total_stall_us,
    attributed_stall_us, transfer_busy_us, transfer_total_KB, ...

Usage:
    python sweep_memory.py <base_yaml> <compute_hw> <memory_hw>
        <memory_idx> <sizes_csv_GB> [experiment_name]

Example:
    python sweep_memory.py examples/run/llamacpp_llama-3-8B_flexinfer.yaml \\
        cpu ram 0 4,5,6,7,8 flexinfer_mem_sweep
"""
from __future__ import annotations

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
    memory_idx: int,
    sizes_GB: list[float],
    experiment_name: str = "memory_sweep",
) -> None:
    sweep_dir = default_sweep_dir(experiment_name)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"experiment dir : {sweep_dir}")
    print(f"base config    : {base_yaml}")
    print(f"compute/memory : {compute_hw} / {memory_hw}")
    print(f"sizes (GB)     : {sizes_GB}")

    results: list[SweepResult] = []
    for size_GB in sizes_GB:
        size_KB = int(size_GB * 1024 * 1024)
        label = f"{size_GB:g}GB"
        cell = Cell(
            label=label,
            params={"memory_GB": size_GB, "memory_hw": memory_hw},
            overrides=[
                f"hardware.memory.{memory_idx}.args.memory_size_KB={size_KB}",
            ],
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
        sweep_dir, results, param_keys=["memory_GB", "memory_hw"]
    )
    print(f"\nSweep complete. summary: {summary}")
    print(f"Plot with: python scripts/visualization/plot_metric_vs_param.py "
          f"{sweep_dir} memory_GB total_stall_us")


def _parse_sizes(token: str) -> list[float]:
    return [float(s) for s in token.split(",") if s.strip()]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) not in (5, 6):
        print(
            "Usage: python sweep_memory.py <base_yaml> <compute_hw> <memory_hw> "
            "<memory_idx> <sizes_csv_GB> [experiment_name]"
        )
        sys.exit(2)
    base = Path(argv[0])
    compute = argv[1]
    mem = argv[2]
    idx = int(argv[3])
    sizes = _parse_sizes(argv[4])
    name = argv[5] if len(argv) == 6 else "memory_sweep"
    main(base, compute, mem, idx, sizes, name)
