#!/usr/bin/env python3
"""
kineto_probe_microbench.py — per-op kineto observer overhead microbenchmark.

Calibrates the pure (isolated) per-op cost of the top cpu_leaf op_names
recorded in cg-sim's PyTorch profile traces, on the system that produced
those traces. The output feeds into probe_effect_table.csv generation —
see docs/eager-lazy-probing-effect.md for the full pipeline.

For each of the top-6 cpu_leaf op_names, measures per-call wall time
  (a) with kineto profiler OFF      → baseline_ns  (true unprofiled cost)
  (b) with kineto profiler ON       → probed_ns    (isolated profiled cost)
Per-op observer overhead = probed_ns − baseline_ns.

Compares against trace_mean duration_ns extracted from existing eager
trace files (hard-coded below). If probed_ns ≈ trace_mean, the trace's
duration_ns is faithful to the in-context per-op cost. If trace_mean ≫
probed_ns, the trace has workload-context overhead (cache/allocator
pressure, kineto buffer contention) that this microbench cannot capture.

Usage:
    python3 scripts/tool/kineto_probe_microbench.py

Output:
    Prints progress to stdout, writes overhead_results.txt next to this
    script. Copy that file back to the cg-sim repo for probe_effect_table
    generation if running on a different host.
"""

import gc
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity


N_REPS = 5
N_ITER = 200_000  # per repetition

# Reference: trace-derived mean duration_ns for eager (from
# op_distribution_output.txt — top-6 cpu_leaf ops)
TRACE_MEAN_EAGER_NS = {
    "aten::view":          7370,
    "aten::as_strided":    8190,
    "aten::empty":        10610,
    "aten::empty_strided": 13560,
    "aten::_unsafe_view":  7400,
    "aten::to":            6860,
}


def build_ops(device: str):
    """Build (name, callable) pairs. Each callable invokes exactly one
    aten dispatch when called once. Tensors and shapes are kept small to
    avoid allocator pressure across N_ITER iterations."""
    X = torch.empty(1024, 1024, device=device)
    EMPTY_SIZE = [16, 16]
    EMPTY_STRIDE = [16, 1]

    return X, [
        # Pure metadata ops
        ("aten::view",          lambda: X.view([-1])),
        ("aten::as_strided",    lambda: torch.as_strided(X, [1024, 1024], [1024, 1])),
        # Internal view variant — used heavily in eager
        ("aten::_unsafe_view",  lambda: torch.ops.aten._unsafe_view(X, [-1])),
        # Allocation ops — small size to keep allocator caching trivial
        ("aten::empty",         lambda: torch.empty(EMPTY_SIZE, device=device)),
        ("aten::empty_strided", lambda: torch.empty_strided(EMPTY_SIZE, EMPTY_STRIDE, device=device)),
        # No-op .to() — matches the cpu_leaf usage pattern (same dtype/device)
        ("aten::to",            lambda: X.to(X.dtype)),
    ]


def empty_loop_ns(n: int) -> int:
    t0 = time.perf_counter_ns()
    for _ in range(n):
        pass
    return time.perf_counter_ns() - t0


def measure(fn, n: int, with_profiler: bool) -> int:
    """Return total ns for n calls to fn, with or without profiler context.
    GC is disabled across the measurement window."""
    ctx = (profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
                with_modules=False,
            )
            if with_profiler else nullcontext())
    gc.disable()
    try:
        with ctx:
            t0 = time.perf_counter_ns()
            for _ in range(n):
                fn()
            return time.perf_counter_ns() - t0
    finally:
        gc.enable()
        gc.collect()


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run on the RTX4090 host.", file=sys.stderr)
        sys.exit(1)

    torch.cuda.init()
    torch.set_grad_enabled(False)

    device_name = torch.cuda.get_device_name()
    print(f"device:   {device_name}")
    print(f"pytorch:  {torch.__version__}")
    print(f"cuda:     {torch.version.cuda}")
    print(f"python:   {sys.version.split()[0]}")
    print(f"n_iter:   {N_ITER}  reps: {N_REPS}\n")

    X, OPS = build_ops("cuda")

    # Warm everything: CUDA context, allocator caches, dispatcher
    print("warming up...", flush=True)
    for _ in range(1000):
        for _, fn in OPS:
            fn()
    torch.cuda.synchronize()
    gc.collect()

    # Calibrate Python empty-loop overhead
    loop_ns = [empty_loop_ns(N_ITER) for _ in range(N_REPS)]
    loop_per_iter = sum(loop_ns) / N_REPS / N_ITER
    print(f"empty Python loop: {loop_per_iter:.1f} ns/iter (will be subtracted)\n", flush=True)

    print(f"{'op_name':<22} {'baseline_ns':>14} {'probed_ns':>14} {'overhead_ns':>14}")
    print("-" * 72)

    results = []
    for name, fn in OPS:
        baseline_ns = []
        probed_ns = []
        for _ in range(N_REPS):
            t = measure(fn, N_ITER, with_profiler=False)
            baseline_ns.append(t / N_ITER - loop_per_iter)
        for _ in range(N_REPS):
            t = measure(fn, N_ITER, with_profiler=True)
            probed_ns.append(t / N_ITER - loop_per_iter)

        b_mean = statistics.mean(baseline_ns)
        b_std  = statistics.stdev(baseline_ns) if len(baseline_ns) > 1 else 0.0
        p_mean = statistics.mean(probed_ns)
        p_std  = statistics.stdev(probed_ns) if len(probed_ns) > 1 else 0.0
        overhead = p_mean - b_mean
        results.append((name, b_mean, b_std, p_mean, p_std, overhead))
        print(f"{name:<22} {b_mean:>10.0f}±{b_std:<4.0f} {p_mean:>10.0f}±{p_std:<4.0f} {overhead:>+14.0f}", flush=True)

    # Write a parseable summary next to this script
    out_path = Path(__file__).resolve().parent / "overhead_results.txt"
    with open(out_path, "w") as f:
        f.write(f"# pytorch={torch.__version__}  cuda={torch.version.cuda}  device={device_name}\n")
        f.write(f"# python={sys.version.split()[0]}\n")
        f.write(f"# n_iter_per_rep={N_ITER}  n_reps={N_REPS}\n")
        f.write(f"# empty_python_loop_ns_per_iter={loop_per_iter:.1f}\n")
        f.write("# All values in nanoseconds, per single op call.\n")
        f.write("# overhead_ns = probed_ns - baseline_ns = pure per-op kineto observer overhead.\n")
        f.write("# trace_mean_eager_ns = mean duration_ns from existing eager trace files (for comparison).\n")
        f.write("# If overhead_ns ≈ trace_mean_eager_ns - baseline_ns, the trace is faithful and clipping=overhead.\n")
        f.write("# If trace_mean_eager_ns ≫ probed_ns, the trace has workload-context cost beyond raw observer.\n\n")
        f.write(f"{'op_name':<22} {'baseline_ns':>12} {'baseline_std':>13} "
                f"{'probed_ns':>12} {'probed_std':>11} "
                f"{'overhead_ns':>13} {'trace_mean_eager_ns':>21} "
                f"{'trace - probed':>16}\n")
        for name, b_mean, b_std, p_mean, p_std, overhead in results:
            t = TRACE_MEAN_EAGER_NS.get(name, 0)
            extra = t - p_mean if t else 0
            f.write(f"{name:<22} {b_mean:>12.0f} {b_std:>13.0f} "
                    f"{p_mean:>12.0f} {p_std:>11.0f} "
                    f"{overhead:>13.0f} {t:>21} {extra:>+16.0f}\n")

    print(f"\nWrote: {out_path}")
    print("\nCopy overhead_results.txt back to the cg-sim repo for analysis.")


if __name__ == "__main__":
    main()
