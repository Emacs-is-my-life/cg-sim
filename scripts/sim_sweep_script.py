#!/usr/bin/env python3
"""Simulation sweep over 0521 profiling results.

For each of 4 models × 2 variants (eager, inductor):
  1. vanilla  — simulate the trace with no offload schedule
     compare sim e2e/peak vs real noprofile timing

For each of 4 models (eager trace only):
  2. hf_<mode> — for each HF offload mode (model, module, sequential, group):
       a. run hf_accelerate scheduler to synthesize schedule.json
       b. simulate with inject_eager_schedule_path

Output layout under --output-root (default /data/cg-sim/exp_results/0521_sweep):
  <model_label>_<variant>/vanilla/input.yaml
  <model_label>_<variant>/vanilla/sim_result.json
  <model_label>_eager/hf_<mode>/schedule.json
  <model_label>_eager/hf_<mode>/input.yaml
  <model_label>_eager/hf_<mode>/sim_result.json
  summary.json  — all metrics in one place

At the end prints two tables:
  Table 1: vanilla  — model / variant / sim_e2e_ms / real_e2e_ms / sim_peak_mb / real_peak_mb
  Table 2: offload  — sim_e2e_ms / sim_peak_vram_mb per model × mode
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import yaml


CG_SIM_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

PROFILE_BASE = Path("/data/pytorch-source/exp_results/0521_sweep")

# (model_label, variant, profile_dir)
# variant is "eager" or "inductor"; offload sims only run on "eager"
MODELS = [
    ("sdxl_turbo", "eager",    PROFILE_BASE / "sdxl_eager"),
    ("sdxl_turbo", "inductor", PROFILE_BASE / "sdxl_inductor"),
    ("sd3_med",    "eager",    PROFILE_BASE / "sd3_eager"),
    ("sd3_med",    "inductor", PROFILE_BASE / "sd3_inductor"),
    ("llama3b",    "eager",    PROFILE_BASE / "llama3b_eager"),
    ("llama3b",    "inductor", PROFILE_BASE / "llama3b_inductor"),
    ("llama8b",    "eager",    PROFILE_BASE / "llama8b_eager"),
    ("llama8b",    "inductor", PROFILE_BASE / "llama8b_inductor"),
]

HF_OFFLOAD_MODES = ("model", "module", "sequential", "group")

# PCIe 4.0 x16 one-way = 16 GT/s × 128b/130b × 1 direction ≈ 15.75 GB/s
# Practical measured: ~12.5 GB/s sustained H2D.
PCIE_BW_KBps = 12_500_000

HARDWARE_BLOCK = {
    "compute": [
        {"name": "cpu", "type": "SimpleCPU", "args": {"memory": "ram", "modifier": 1}},
        {"name": "gpu0", "type": "SimpleGPU",
         "args": {"memory": "vram0", "modifier": 1, "max_concurrent_jobs": 1}},
    ],
    "memory": [
        {"name": "ram", "type": "SimpleRAM",
         "args": {"custom_dep_tag": "TARGET_RAM",
                  "memory_size_KB": 33554432,
                  "memory_bandwidth_KBps": PCIE_BW_KBps}},
        {"name": "vram0", "type": "SimpleVRAM",
         "args": {"custom_dep_tag": "TARGET_VRAM",
                  "memory_size_KB": 24624320,
                  "memory_bandwidth_KBps": PCIE_BW_KBps}},
    ],
    "storage": [
        {"name": "ssd", "type": "SimpleSSD", "args": {
            "custom_dep_tag": "TARGET_SSD",
            "fixed_latency_micros": 50,
            "read_io_curve_KBps": [
                [4, 80000], [8, 160000], [16, 320000], [32, 800000],
                [64, 1600000], [128, 3000000], [256, 5000000],
                [512, 6500000], [1024, 7000000],
            ],
            "write_io_curve_KBps": [
                [4, 60000], [8, 120000], [16, 250000], [32, 600000],
                [64, 1200000], [128, 2200000], [256, 3500000],
                [512, 4500000], [1024, 5000000],
            ],
        }},
    ],
}

SCHEDULER_VANILLA = {
    "type": "DeviceAwareVanillaAsync",
    "args": {
        "cpu_compute": "cpu",
        "cuda_compute": "gpu0",
        "cuda_device": "cuda:0",
        "h2d_streams": 1,
    },
}


# ---------------------------------------------------------------------------
# Bundle discovery
# ---------------------------------------------------------------------------

def find_bundle_dir(profile_dir: Path) -> Path | None:
    """Return the directory containing manifest.json under profile_dir."""
    canonical = profile_dir / "llama_bundle" / "manifest.json"
    if canonical.is_file():
        return canonical.parent
    for candidate in sorted(profile_dir.rglob("manifest.json")):
        return candidate.parent
    return None


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)


def build_vanilla_yaml(
    result_path: Path,
    profile_dir: Path,
    bundle_manifest_rel: str,
) -> dict:
    return {
        "logger": {"args": {"result_path": str(result_path), "log_level": 0}},
        "trace": {
            "type": "PytorchProfile",
            "args": {
                "profile_dir": str(profile_dir),
                "bundle_manifest": bundle_manifest_rel,
                "skip_zero_byte_tensors": True,
                "zero_wait_nodes": True,
                "add_temporal_data_control_edges": True,
            },
        },
        "hardware": HARDWARE_BLOCK,
        "scheduler": SCHEDULER_VANILLA,
    }


def build_offload_yaml(
    result_path: Path,
    profile_dir: Path,
    bundle_manifest_rel: str,
    schedule_path: Path,
) -> dict:
    doc = build_vanilla_yaml(result_path, profile_dir, bundle_manifest_rel)
    doc["trace"]["args"]["inject_eager_schedule_path"] = str(schedule_path)
    return doc


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], cwd: Path, log_path: Path | None = None) -> int:
    log_path and log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    start = time.perf_counter()
    with subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    ) as proc:
        lines = []
        for line in proc.stdout:
            print(f"    {line}", end="", flush=True)
            lines.append(line)
        proc.wait()
    elapsed = time.perf_counter() - start
    if log_path is not None:
        log_path.write_text("".join(lines) + f"\n# exit_code: {proc.returncode}\n# elapsed_seconds: {elapsed:.3f}\n")
    status = "ok" if proc.returncode == 0 else f"FAILED({proc.returncode})"
    print(f"  [{status} in {elapsed:.1f}s]", flush=True)
    return proc.returncode


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_sim_result(path: Path) -> dict[str, float | None]:
    try:
        events = json.loads(path.read_text())
        if isinstance(events, dict) and "traceEvents" in events:
            events = events["traceEvents"]
        for ev in reversed(events):
            if ev.get("name") == "SIMULATION_RESULT":
                p = ev.get("args", {})
                sim = p.get("simulation", {})
                mems = p.get("memory", [])
                peak_kb = next(
                    (m.get("peak_memory_usage_KB") for m in mems if m.get("name") == "vram0"),
                    None,
                )
                ok = sim.get("success") == "True" or sim.get("success") is True
                return {
                    "e2e_ms": float(sim.get("time", 0)) / 1000.0 if ok else None,
                    "peak_vram_mb": float(peak_kb) / 1024.0 if (peak_kb is not None and ok) else None,
                    "success": ok,
                }
    except Exception as exc:
        print(f"  [WARN] could not parse {path}: {exc}", flush=True)
    return {"e2e_ms": None, "peak_vram_mb": None, "success": False}


def parse_noprofile_log(log_path: Path) -> dict[str, float | None]:
    result: dict[str, float | None] = {"e2e_ms": None, "peak_vram_mb": None}
    if not log_path.exists():
        return result
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("inference_seconds:"):
            try:
                result["e2e_ms"] = float(line.split(":", 1)[1]) * 1000.0
            except ValueError:
                pass
        if line.startswith("GPU memory:"):
            # "GPU memory: 6624.5 MB allocated, 8128.6 MB peak"
            try:
                parts = line.split(",")
                peak_part = next(p for p in parts if "peak" in p)
                result["peak_vram_mb"] = float(peak_part.split()[0])
            except (StopIteration, ValueError, IndexError):
                pass
    return result


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------

def run_vanilla(
    model_label: str,
    profile_dir: Path,
    output_root: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    bundle_dir = find_bundle_dir(profile_dir)
    if bundle_dir is None:
        print(f"  [SKIP] no bundle found in {profile_dir}", flush=True)
        return {}

    manifest_rel = str(bundle_dir.relative_to(profile_dir) / "manifest.json")
    out_dir = output_root / model_label / "vanilla"
    result_path = out_dir / "sim_result.json"
    yaml_path = out_dir / "input.yaml"

    doc = build_vanilla_yaml(result_path.resolve(), profile_dir.resolve(), manifest_rel)
    _write_yaml(yaml_path, doc)
    rc = run_cmd(
        [PYTHON, "main.py", "-i", str(yaml_path.resolve())],
        cwd=CG_SIM_ROOT,
        log_path=out_dir / "run.log",
    )
    if rc != 0:
        return {"error": f"sim exit {rc}"}

    return parse_sim_result(result_path)


def run_hf_offload(
    model_label: str,
    profile_dir: Path,
    output_root: Path,
    mode: str,
    *,
    force: bool,
) -> dict[str, Any]:
    bundle_dir = find_bundle_dir(profile_dir)
    if bundle_dir is None:
        print(f"  [SKIP] no bundle found in {profile_dir}", flush=True)
        return {}

    manifest_rel = str(bundle_dir.relative_to(profile_dir) / "manifest.json")
    out_dir = output_root / model_label / f"hf_{mode}"
    schedule_path = out_dir / "schedule.json"
    result_path = out_dir / "sim_result.json"
    yaml_path = out_dir / "input.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Harness-specific mode→granularity override.
    #
    # The hf_accelerate solver's default mapping (model/module-hook →
    # depth:1, sequential/module → leaf) assumes the SDXL-style diffusers
    # harness where `--offload-mode model` calls
    # `pipe.enable_model_cpu_offload()` (chained `cpu_offload_with_hook`,
    # per-component .to(device)).
    #
    # The llama harness (run_llama_accelerate_cpu_offload.py) is different:
    #   --mode model      → accelerate.cpu_offload(pipe.model)       (per-leaf)
    #   --mode module     → accelerate.cpu_offload per top-level mod (per-leaf)
    #   --mode sequential → accelerate.cpu_offload in forward order  (per-leaf)
    #   --mode module-hook→ cpu_offload_with_hook chained            (per-component)
    #   --mode group      → diffusers.apply_group_offloading         (per-block)
    #
    # All three of (model, module, sequential) on llama use AlignDevicesHook
    # per leaf — no per-param .to("cpu") D2H. The solver default would
    # incorrectly pick depth:1 for `--mode model`, putting the whole
    # LlamaModel resident in VRAM at every burst and emitting phantom
    # D2H bytes. Force granularity=leaf + no-d2h-evict to match the
    # harness semantics.
    extra_solver_args: list[str] = []
    is_llama = model_label.startswith("llama")
    if is_llama and mode in ("model", "module", "sequential"):
        extra_solver_args = ["--granularity", "leaf", "--no-d2h-evict"]

    rc = run_cmd(
        [
            PYTHON, "-m", "graph_modifiers.schedulers.hf_accelerate.main",
            str(bundle_dir),
            "--mode", mode,
            "--lookahead", "1",
            *extra_solver_args,
            "--output", str(schedule_path),
        ],
        cwd=CG_SIM_ROOT,
        log_path=out_dir / "scheduler.log",
    )
    if rc != 0:
        return {"error": f"scheduler exit {rc}"}

    doc = build_offload_yaml(
        result_path.resolve(), profile_dir.resolve(), manifest_rel, schedule_path.resolve()
    )
    _write_yaml(yaml_path, doc)
    rc = run_cmd(
        [PYTHON, "main.py", "-i", str(yaml_path.resolve())],
        cwd=CG_SIM_ROOT,
        log_path=out_dir / "run.log",
    )
    if rc != 0:
        return {"error": f"sim exit {rc}"}

    return parse_sim_result(result_path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_FMT_F = "{:>10.1f}"
_FMT_S = "{:>10}"
_NA = "       N/A"


def _fmt(val: float | None, fmt: str = _FMT_F) -> str:
    return fmt.format(val) if val is not None else _NA


def print_vanilla_table(rows: list[dict]) -> None:
    header = f"{'Model':<14} {'Variant':<10} {'sim_e2e_ms':>10} {'real_e2e_ms':>11} {'sim_peak_mb':>11} {'real_peak_mb':>12}"
    sep = "-" * len(header)
    print("\n=== Vanilla: sim vs real ===")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['model']:<14}"
            f" {r['variant']:<10}"
            f" {_fmt(r.get('sim_e2e_ms')):>10}"
            f" {_fmt(r.get('real_e2e_ms')):>11}"
            f" {_fmt(r.get('sim_peak_mb')):>11}"
            f" {_fmt(r.get('real_peak_mb')):>12}"
        )


def print_offload_table(rows: list[dict]) -> None:
    header = f"{'Model':<14} {'Mode':<12} {'sim_e2e_ms':>10} {'sim_peak_mb':>11}"
    sep = "-" * len(header)
    print("\n=== HF Offload: sim results ===")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['model']:<14}"
            f" {r['mode']:<12}"
            f" {_fmt(r.get('sim_e2e_ms')):>10}"
            f" {_fmt(r.get('sim_peak_mb')):>11}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default="/data/cg-sim/exp_results/0521_sweep",
                   help="Root directory for sim outputs.")
    p.add_argument("--force", action="store_true", help="Re-run even if outputs exist.")
    p.add_argument("--only", default=None,
                   help="Comma-separated model_label substrings to run (e.g. 'sdxl,llama3b').")
    p.add_argument("--modes", default=",".join(HF_OFFLOAD_MODES),
                   help="Comma-separated HF offload modes to simulate.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    only_filters = [f.strip() for f in args.only.split(",")] if args.only else None

    models = MODELS
    if only_filters:
        models = [m for m in models if any(f in m[0] for f in only_filters)]

    vanilla_rows: list[dict] = []
    offload_rows: list[dict] = []
    summary: dict[str, Any] = {}

    for model_label, variant, profile_dir in models:
        sim_label = f"{model_label}_{variant}"
        print(f"\n{'='*60}")
        print(f"Model: {sim_label}  ({profile_dir})")
        print(f"{'='*60}", flush=True)

        noprofile_log = profile_dir / "noprofile" / "terminal_output.log"
        real = parse_noprofile_log(noprofile_log)

        print(f"\n--- vanilla ---", flush=True)
        vsim = run_vanilla(sim_label, profile_dir, output_root, force=args.force)
        vanilla_rows.append({
            "model": model_label,
            "variant": variant,
            "sim_e2e_ms": vsim.get("e2e_ms"),
            "real_e2e_ms": real.get("e2e_ms"),
            "sim_peak_mb": vsim.get("peak_vram_mb"),
            "real_peak_mb": real.get("peak_vram_mb"),
        })
        summary.setdefault(sim_label, {})["vanilla"] = {**vsim, "real": real}

        if variant == "eager":
            for mode in modes:
                print(f"\n--- hf_{mode} ---", flush=True)
                osim = run_hf_offload(sim_label, profile_dir, output_root, mode, force=args.force)
                offload_rows.append({
                    "model": model_label,
                    "mode": mode,
                    "sim_e2e_ms": osim.get("e2e_ms"),
                    "sim_peak_mb": osim.get("peak_vram_mb"),
                })
                summary[sim_label][f"hf_{mode}"] = osim

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written: {summary_path}", flush=True)

    print_vanilla_table(vanilla_rows)
    print_offload_table(offload_rows)


if __name__ == "__main__":
    main()
