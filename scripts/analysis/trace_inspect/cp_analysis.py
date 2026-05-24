"""
Compute critical path on the full node-DAG (control + data edges), count
cpu_leaf nodes on the CP, and test the hypothesis:

    gap / n_cpu_leaf_on_CP   should cluster across workloads

If it does, probe overhead is proportional to exposed-node-count and the
correction is just `subtract α from every cpu_leaf duration_ns` with a
single workload-independent α.

Notes on what this CP captures vs the sim's actual CP:
  - Control edges (thread_order, stream_order, submit, wait) — included
  - Data edges (producer→tensor→consumer composed into producer→consumer) — included
  - Memcpys inserted by the sim for cross-device tensors — NOT included
  - GPU resource serialization (one job at a time) — NOT included
So this is a lower-bound CP. Adequate for testing the proportionality
hypothesis; if numbers cluster, hypothesis is supported.
"""

import csv
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = REPO_ROOT / "tmp" / "eager_overhead_investigation" / "cp_output.txt"
_OUT_FH = None

def out(msg=""):
    _OUT_FH.write(msg + "\n")
    _OUT_FH.flush()

TRACES = [
    ("pytorch-eager__llama-3-3B",  0.159, 0.428),
    ("pytorch-eager__llama-3-8B",  0.307, 0.529),
    ("pytorch-eager__sd3",         1.193, 1.967),
    ("pytorch-eager__sdxl-turbo",  0.185, 0.343),
    ("pytorch-lazy__llama-3-3B",   0.137, 0.130),
    ("pytorch-lazy__llama-3-8B",   0.283, 0.266),
    ("pytorch-lazy__sd3",          0.889, 0.988),
    ("pytorch-lazy__sdxl-turbo",   0.180, 0.167),
]
ROOT = REPO_ROOT / "examples" / "trace"


def _parse_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def load_bundle(bundle: Path):
    """Returns (nodes, control_edges, tensor_producer, tensor_consumers).
    nodes: dict node_id -> {role, dev, dur}
    control_edges: list of (src, dst)
    tensor_producer: dict tensor_id -> producer node_id (or None)
    tensor_consumers: dict tensor_id -> list of consumer node_id
    """
    nodes = {}
    with (bundle / "runtime_nodes.csv").open() as f:
        for r in csv.DictReader(f):
            nodes[r["node_id"]] = {
                "role": r["runtime_role"],
                "dev":  r["device_type"],
                "dur":  _parse_int(r["duration_ns"]),
            }

    control_edges = []
    tensor_producer = {}
    tensor_consumers = defaultdict(list)

    with (bundle / "runtime_edges.csv").open() as f:
        for r in csv.DictReader(f):
            kind = r["edge_kind"]
            src = r["src_node_id"]
            dst = r["dst_node_id"]
            if kind in ("thread_order", "stream_order", "submit", "wait"):
                control_edges.append((src, dst))
            elif kind == "data_input":
                # src is tensor_id, dst is node_id (tensor → consumer)
                tensor_consumers[src].append(dst)
            elif kind == "data_output":
                # src is node_id, dst is tensor_id (producer → tensor)
                tensor_producer[dst] = src
    return nodes, control_edges, tensor_producer, tensor_consumers


def build_dag(nodes, control_edges, tensor_producer, tensor_consumers):
    """Compose data edges into direct node→node edges and union with
    control edges. Drops back-edges (dst earlier than src in trace order)
    to preserve DAG-ness — back-edges arise from in-place ops and view
    aliasing where producer/consumer relationships are non-causal.
    Returns adjacency dict."""
    order_idx = {nid: i for i, nid in enumerate(nodes)}
    children = defaultdict(set)
    parents  = defaultdict(set)
    n_dropped = 0
    def _add(s, d):
        nonlocal n_dropped
        if s == d or s not in nodes or d not in nodes:
            return
        if order_idx[d] <= order_idx[s]:
            n_dropped += 1
            return
        children[s].add(d)
        parents[d].add(s)
    for s, d in control_edges:
        _add(s, d)
    for tid, prod in tensor_producer.items():
        for cons in tensor_consumers.get(tid, ()):
            _add(prod, cons)
    return children, parents, n_dropped


def longest_path_with_predecessor(nodes, children):
    """Longest path by sum of node durations, in trace-insertion topo order.
    Returns (longest_to_node, pred_node)."""
    order = list(nodes.keys())  # trace order is topological per kineto
    longest = {nid: nodes[nid]["dur"] for nid in nodes}
    pred = {nid: None for nid in nodes}
    for nid in order:
        base = longest[nid]
        for c in children.get(nid, ()):
            cand = base + nodes[c]["dur"]
            if cand > longest[c]:
                longest[c] = cand
                pred[c] = nid
    return longest, pred


def trace_cp(nodes, longest, pred):
    """Recover the CP node sequence by walking back from argmax(longest)."""
    end = max(longest, key=lambda n: longest[n])
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = pred[cur]
    path.reverse()
    return path, longest[end]


def summarize(tag, normal_s, sim_s):
    bundle = ROOT / f"{tag}__RTX4090" / "llama_bundle"
    if not (bundle / "runtime_nodes.csv").exists():
        return None

    nodes, ctrl, prod, cons = load_bundle(bundle)
    children, parents, n_dropped = build_dag(nodes, ctrl, prod, cons)
    longest, pred = longest_path_with_predecessor(nodes, children)
    path, cp_ns = trace_cp(nodes, longest, pred)

    n_cpu_leaf_total = sum(1 for nid in nodes if nodes[nid]["role"] == "cpu_leaf")
    n_cpu_leaf_on_cp = sum(1 for nid in path if nodes[nid]["role"] == "cpu_leaf")
    n_gpu_run_on_cp  = sum(1 for nid in path if nodes[nid]["role"] == "gpu_runtime")
    n_other_on_cp    = len(path) - n_cpu_leaf_on_cp - n_gpu_run_on_cp

    cpu_leaf_dur_on_cp = sum(nodes[nid]["dur"] for nid in path if nodes[nid]["role"] == "cpu_leaf")
    gpu_run_dur_on_cp  = sum(nodes[nid]["dur"] for nid in path if nodes[nid]["role"] == "gpu_runtime")

    gap_us = (sim_s - normal_s) * 1e6

    out(f"\n== {tag} ==")
    out(f"  normal:      {normal_s*1e6:>10.0f} µs")
    out(f"  cg-sim:      {sim_s*1e6:>10.0f} µs")
    out(f"  gap:         {gap_us:>10.0f} µs")
    out(f"  CP length (DAG, dur-sum):   {cp_ns/1e3:>10.0f} µs")
    out(f"  edges dropped (back-edges): {n_dropped}")
    out(f"  CP nodes total: {len(path)}")
    out(f"  CP nodes: cpu_leaf={n_cpu_leaf_on_cp}  gpu_runtime={n_gpu_run_on_cp}  other={n_other_on_cp}")
    out(f"  cpu_leaf total: {n_cpu_leaf_total}  (on CP: {n_cpu_leaf_on_cp},  {100*n_cpu_leaf_on_cp/max(n_cpu_leaf_total,1):.1f}%)")
    out(f"  Σ cpu_leaf dur on CP:    {cpu_leaf_dur_on_cp/1e3:>10.0f} µs")
    out(f"  Σ gpu_runtime dur on CP: {gpu_run_dur_on_cp/1e3:>10.0f} µs")
    if n_cpu_leaf_on_cp:
        per_op = gap_us / n_cpu_leaf_on_cp
        out(f"  >>> gap / n_cpu_leaf_on_CP = {per_op:.2f} µs/op  <<<")

    return {
        "tag": tag,
        "normal_us": normal_s*1e6,
        "sim_us": sim_s*1e6,
        "gap_us": gap_us,
        "cp_us": cp_ns/1e3,
        "n_cpu_leaf_total": n_cpu_leaf_total,
        "n_cpu_leaf_on_cp": n_cpu_leaf_on_cp,
        "n_gpu_run_on_cp": n_gpu_run_on_cp,
        "cpu_leaf_dur_on_cp_us": cpu_leaf_dur_on_cp/1e3,
        "gpu_run_dur_on_cp_us":  gpu_run_dur_on_cp/1e3,
    }


def main():
    rows = []
    for tag, normal, sim in TRACES:
        r = summarize(tag, normal, sim)
        if r:
            rows.append(r)

    out("\n\n=== Hypothesis test: gap / n_cpu_leaf_on_CP ===")
    out("  If the refined hypothesis holds, eager values cluster.")
    out()
    out(f"{'workload':<32} {'gap µs':>10} {'n_CP_cpu':>10} {'µs/op_CP':>10} {'µs/op_total':>14}")
    out("-"*82)
    for r in rows:
        per_cp = r["gap_us"] / max(r["n_cpu_leaf_on_cp"], 1)
        per_total = r["gap_us"] / max(r["n_cpu_leaf_total"], 1)
        out(f"{r['tag']:<32} {r['gap_us']:>10.0f} {r['n_cpu_leaf_on_cp']:>10} {per_cp:>10.2f} {per_total:>14.2f}")


if __name__ == "__main__":
    with open(OUT_PATH, "w") as f:
        globals()["_OUT_FH"] = f
        main()
