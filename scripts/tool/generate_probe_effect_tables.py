"""
generate_probe_effect_tables.py — produce per-trace probe_effect_table.csv files
that the cg-sim PyTorchProfile loader picks up automatically.

For each trace under examples/trace/, computes per-op
    probe_effect_ns = trace_median(duration_ns) − microbench_probed_ns

where the trace_median is over all cpu_leaf occurrences of that op_name in
that trace, and microbench_probed_ns comes from running
scripts/tool/kineto_probe_microbench.py on the same host that produced the
trace bundles. Median (not mean) is used to be robust against startup
outliers in lazy-mode traces.

The resulting probe_effect_table.csv is written to each trace's root dir
(sibling of the bundle dir). The loader will subtract probe_effect_ns from
each cpu_leaf's duration_ns at load time, with max(0, ...) clamping.

Usage:
    # 1. Run kineto_probe_microbench.py on the profiling host
    # 2. Copy overhead_results.txt next to this script (or update MICROBENCH)
    # 3. python3 scripts/tool/generate_probe_effect_tables.py

See docs/eager-lazy-probing-effect.md for the methodology.
"""
import csv
import re
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]
TRACE_ROOT = REPO_ROOT / "examples" / "trace"
# Default: look next to this script. Override by editing if needed.
MICROBENCH = Path(__file__).resolve().parent / "overhead_results.txt"
if not MICROBENCH.exists():
    # Fallback to historical tmp location used during initial development.
    MICROBENCH = REPO_ROOT / "tmp" / "eager_overhead_investigation" / "overhead_results.txt"

TRACES = [
    "pytorch-eager__llama-3-3B__RTX4090",
    "pytorch-eager__llama-3-8B__RTX4090",
    "pytorch-eager__sd3__RTX4090",
    "pytorch-eager__sdxl-turbo__RTX4090",
    "pytorch-lazy__llama-3-3B__RTX4090",
    "pytorch-lazy__llama-3-8B__RTX4090",
    "pytorch-lazy__sd3__RTX4090",
    "pytorch-lazy__sdxl-turbo__RTX4090",
]


def parse_microbench(path: Path) -> dict[str, dict[str, float]]:
    """Returns {op_name: {'baseline_ns': float, 'probed_ns': float, 'overhead_ns': float}}."""
    result = {}
    with path.open() as f:
        in_header = True
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            cols = re.split(r"\s+", line)
            if in_header:
                # First non-comment row is the column header
                in_header = False
                # Header has spaces inside it but we look for op_name col 0
                if cols[0] != "op_name":
                    in_header = True  # keep looking
                continue
            # data row
            op = cols[0]
            try:
                baseline = float(cols[1])
                probed = float(cols[3])
                overhead = float(cols[5])
            except (IndexError, ValueError):
                continue
            result[op] = {"baseline_ns": baseline, "probed_ns": probed, "overhead_ns": overhead}
    return result


def trace_median_per_op(node_csv: Path) -> tuple[dict[str, float], dict[str, int]]:
    """Median duration_ns per cpu_leaf op_name in this trace. Median is robust
    to startup-init outliers (e.g. lazy traces have isolated 1.3 ms first-call
    aten::empty_strided that would swamp a mean)."""
    per_op: dict[str, list[float]] = {}
    with node_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["runtime_role"] != "cpu_leaf":
                continue
            op = r["op_name"] or ""
            try:
                d = float(r["duration_ns"])
            except (TypeError, ValueError):
                continue
            per_op.setdefault(op, []).append(d)
    medians: dict[str, float] = {}
    counts: dict[str, int] = {}
    for op, ds in per_op.items():
        ds.sort()
        n = len(ds)
        counts[op] = n
        mid = n // 2
        medians[op] = (ds[mid] if n % 2 == 1 else (ds[mid - 1] + ds[mid]) / 2)
    return medians, counts


def main():
    microbench = parse_microbench(MICROBENCH)
    print(f"Microbench ops: {list(microbench.keys())}")
    print(f"Microbench device/setup: see {MICROBENCH}\n")

    for tag in TRACES:
        trace_dir = TRACE_ROOT / tag
        bundle = trace_dir / "llama_bundle"
        node_csv = bundle / "runtime_nodes.csv"
        if not node_csv.exists():
            print(f"[skip] {tag} (no runtime_nodes.csv)")
            continue

        medians, counts = trace_median_per_op(node_csv)

        rows = []
        for op, mb in microbench.items():
            tmed = medians.get(op)
            n = counts.get(op, 0)
            MIN_N = 20  # below this, median is unreliable (one startup outlier swamps)
            if tmed is None or n == 0:
                rows.append({
                    "op_name": op,
                    "probe_effect_ns": 0,
                    "trace_median_ns": 0,
                    "occurrences": 0,
                    "microbench_probed_ns": int(round(mb["probed_ns"])),
                    "note": "op absent in this trace",
                })
                continue
            if n < MIN_N:
                rows.append({
                    "op_name": op,
                    "probe_effect_ns": 0,
                    "trace_median_ns": int(round(tmed)),
                    "occurrences": n,
                    "microbench_probed_ns": int(round(mb["probed_ns"])),
                    "note": f"too few occurrences (n<{MIN_N}); calibration skipped",
                })
                continue
            probe_effect = int(round(tmed - mb["probed_ns"]))
            if probe_effect < 0:
                probe_effect = 0
            rows.append({
                "op_name": op,
                "probe_effect_ns": probe_effect,
                "trace_median_ns": int(round(tmed)),
                "occurrences": n,
                "microbench_probed_ns": int(round(mb["probed_ns"])),
                "note": "trace_median - microbench_probed_ns (RTX4090, kineto record_shapes=True)",
            })

        out = trace_dir / "probe_effect_table.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["op_name", "probe_effect_ns",
                                              "trace_median_ns", "occurrences",
                                              "microbench_probed_ns", "note"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[ok] {tag}: wrote {out.relative_to(TRACE_ROOT.parent)}")
        for r in rows:
            print(f"     {r['op_name']:<24} probe_effect={r['probe_effect_ns']:>6} ns "
                  f"(median={r['trace_median_ns']:>6}, n={r['occurrences']:>6}, probed={r['microbench_probed_ns']:>5})")


if __name__ == "__main__":
    main()
