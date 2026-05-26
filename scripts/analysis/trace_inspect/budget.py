"""
Decompose pytorch-profile traces into CPU vs GPU duration budgets.

For each trace bundle:
  1) sum-of-CPU-leaf duration (the leak candidate)
  2) sum-of-GPU-kernel duration (real work, profiler-blind)
  3) DAG critical-path length (lower bound on sim e2e)
  4) wall span (max(end_ns) - min(start_ns))

Then compare to Normal Run / Profiled Run / cg-sim Replay numbers.
"""

import csv
from collections import defaultdict
from pathlib import Path

TRACES = [
    ("pytorch-eager__llama-3-3B",  0.159, 1.160, 0.428),
    ("pytorch-eager__llama-3-8B",  0.307, 1.382, 0.529),
    ("pytorch-eager__sd3",         1.193, 3.271, 1.967),
    ("pytorch-eager__sdxl-turbo",  0.185, 1.023, 0.343),
    ("pytorch-lazy__llama-3-3B",   0.137, 0.587, 0.130),
    ("pytorch-lazy__llama-3-8B",   0.283, 0.673, 0.266),
    ("pytorch-lazy__sd3",          0.889, 1.658, 0.988),
    ("pytorch-lazy__sdxl-turbo",   0.180, 0.798, 0.167),
]
REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "examples" / "trace"


def _parse_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def load(bundle: Path):
    nodes = {}
    with (bundle / "runtime_nodes.csv").open() as f:
        for r in csv.DictReader(f):
            nodes[r["node_id"]] = {
                "role": r["runtime_role"],
                "dev":  r["device_type"],
                "dur":  _parse_int(r["duration_ns"]),
                "start": _parse_int(r["start_ns"]),
                "end":   _parse_int(r["end_ns"]),
            }
    edges = []
    with (bundle / "runtime_edges.csv").open() as f:
        for r in csv.DictReader(f):
            edges.append((r["src_node_id"], r["dst_node_id"], r["edge_kind"]))
    return nodes, edges


def critical_path(nodes, edges):
    """Longest-path DP over the DAG using duration_ns as node weight.
    Edges are control edges in trace order (already topo). Returns CP length in ns."""
    children = defaultdict(list)
    indeg = defaultdict(int)
    for s, d, _ in edges:
        if s in nodes and d in nodes:
            children[s].append(d)
            indeg[d] += 1
    # topo via Kahn — but trace insertion order is already topological per kineto.
    order = list(nodes.keys())
    longest = {nid: nodes[nid]["dur"] for nid in nodes}
    for nid in order:
        base = longest[nid]
        for c in children.get(nid, ()):
            cand = base + nodes[c]["dur"]
            if cand > longest[c]:
                longest[c] = cand
    return max(longest.values()) if longest else 0


def summarize(tag, normal_s, profiled_s, sim_s):
    bundle = ROOT / f"{tag}__RTX4090" / "llama_bundle"
    if not (bundle / "runtime_nodes.csv").exists():
        return None
    nodes, edges = load(bundle)

    cpu_leaf_sum = sum(n["dur"] for n in nodes.values() if n["role"] == "cpu_leaf")
    gpu_run_sum  = sum(n["dur"] for n in nodes.values() if n["role"] == "gpu_runtime")
    gpu_leaf_sum = sum(n["dur"] for n in nodes.values() if n["role"] == "gpu_leaf")
    other_sum    = sum(n["dur"] for n in nodes.values()
                       if n["role"] not in ("cpu_leaf", "gpu_runtime", "gpu_leaf"))
    wall_span    = max(n["end"] for n in nodes.values()) - min(n["start"] for n in nodes.values())
    cp_ns        = critical_path(nodes, edges)

    n_cpu  = sum(1 for n in nodes.values() if n["role"] == "cpu_leaf")
    n_gpu_run = sum(1 for n in nodes.values() if n["role"] == "gpu_runtime")
    n_gpu_leaf = sum(1 for n in nodes.values() if n["role"] == "gpu_leaf")

    print(f"\n== {tag} ==")
    print(f"  Normal:    {normal_s*1e6:>11.0f} µs")
    print(f"  Profiled:  {profiled_s*1e6:>11.0f} µs")
    print(f"  cg-sim:    {sim_s*1e6:>11.0f} µs")
    print(f"  Trace wall:{wall_span/1e3:>11.0f} µs  (max(end) - min(start))")
    print(f"  CP (sum of node-durs on longest DAG path):  {cp_ns/1e3:>11.0f} µs")
    print(f"  Σ cpu_leaf:   {cpu_leaf_sum/1e3:>11.0f} µs  (n={n_cpu})")
    print(f"  Σ gpu_runtime:{gpu_run_sum/1e3:>11.0f} µs  (n={n_gpu_run})")
    print(f"  Σ gpu_leaf:   {gpu_leaf_sum/1e3:>11.0f} µs  (n={n_gpu_leaf})")
    print(f"  Σ other:      {other_sum/1e3:>11.0f} µs")
    if n_cpu:
        print(f"  Mean cpu_leaf dur: {cpu_leaf_sum/n_cpu/1e3:.2f} µs/node")
    if n_gpu_leaf:
        print(f"  Mean gpu_leaf dur: {gpu_leaf_sum/n_gpu_leaf/1e3:.2f} µs/node")

    return {
        "tag": tag,
        "normal_us": normal_s*1e6,
        "profiled_us": profiled_s*1e6,
        "sim_us": sim_s*1e6,
        "wall_us": wall_span/1e3,
        "cp_us": cp_ns/1e3,
        "cpu_leaf_sum_us": cpu_leaf_sum/1e3,
        "gpu_runtime_sum_us": gpu_run_sum/1e3,
        "gpu_leaf_sum_us": gpu_leaf_sum/1e3,
        "n_cpu_leaf": n_cpu,
        "n_gpu_leaf": n_gpu_leaf,
    }


def main():
    rows = []
    for tag, normal, profiled, sim in TRACES:
        r = summarize(tag, normal, profiled, sim)
        if r:
            rows.append(r)

    # Comparative table
    print("\n\n=== Comparative table (all µs) ===")
    hdr = ["tag", "normal", "profiled", "cg-sim", "wall", "CP", "Σ cpu_leaf", "Σ gpu_leaf", "Σ gpu_runtime"]
    print("| " + " | ".join(f"{h:>15}" for h in hdr) + " |")
    print("|" + "|".join("-"*17 for _ in hdr) + "|")
    for r in rows:
        print("| " + " | ".join(f"{r['tag']:>15}" if i == 0 else f"{v:>15.0f}"
            for i, v in enumerate([r["tag"], r["normal_us"], r["profiled_us"], r["sim_us"],
                                    r["wall_us"], r["cp_us"], r["cpu_leaf_sum_us"],
                                    r["gpu_leaf_sum_us"], r["gpu_runtime_sum_us"]])) + " |")

    # Hypothesis check: gap_per_cpu_node
    print("\n=== Per-cpu_leaf overhead (sim - normal) / n_cpu_leaf ===")
    for r in rows:
        gap = r["sim_us"] - r["normal_us"]
        per = gap / r["n_cpu_leaf"] if r["n_cpu_leaf"] else 0
        print(f"  {r['tag']:>30}: gap={gap:>9.0f}µs  n_cpu_leaf={r['n_cpu_leaf']:>6}  → {per:.2f} µs/node")


if __name__ == "__main__":
    main()
