"""
Enumerate cpu_leaf op_name (the natural notion of "type") across all 8 traces.
Report:
  - per-trace unique op_name count
  - union across all traces
  - top op_names by total occurrence
  - top op_names by total duration

Output: cpu_leaf_types_output.txt
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "examples" / "trace"
OUT = REPO_ROOT / "tmp" / "eager_overhead_investigation" / "cpu_leaf_types_output.txt"

TAGS = [
    "pytorch-eager__llama-3-3B",
    "pytorch-eager__llama-3-8B",
    "pytorch-eager__sd3",
    "pytorch-eager__sdxl-turbo",
    "pytorch-lazy__llama-3-3B",
    "pytorch-lazy__llama-3-8B",
    "pytorch-lazy__sd3",
    "pytorch-lazy__sdxl-turbo",
]


def _parse_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def scan(bundle: Path):
    """Returns (count_by_op, dur_by_op) for cpu_leaf nodes only."""
    cnt = Counter()
    dur = Counter()
    with (bundle / "runtime_nodes.csv").open() as f:
        for r in csv.DictReader(f):
            if r["runtime_role"] != "cpu_leaf":
                continue
            op = r["op_name"] or "<unknown>"
            cnt[op] += 1
            dur[op] += _parse_int(r["duration_ns"])
    return cnt, dur


def main():
    per_trace_cnt = {}   # tag -> Counter op_name
    per_trace_dur = {}   # tag -> Counter op_name
    union_ops = set()
    eager_ops = set()
    lazy_ops = set()

    for tag in TAGS:
        bundle = ROOT / f"{tag}__RTX4090" / "llama_bundle"
        if not (bundle / "runtime_nodes.csv").exists():
            continue
        c, d = scan(bundle)
        per_trace_cnt[tag] = c
        per_trace_dur[tag] = d
        union_ops |= set(c.keys())
        if tag.startswith("pytorch-eager"):
            eager_ops |= set(c.keys())
        else:
            lazy_ops |= set(c.keys())

    with open(OUT, "w") as f:
        f.write("=== cpu_leaf op_name diversity ===\n\n")
        f.write(f"Union across all 8 traces:    {len(union_ops)} distinct op_names\n")
        f.write(f"Union across 4 eager traces:  {len(eager_ops)} distinct op_names\n")
        f.write(f"Union across 4 lazy  traces:  {len(lazy_ops)} distinct op_names\n")
        f.write(f"In both modes (intersection): {len(eager_ops & lazy_ops)} op_names\n")
        f.write(f"Eager-only:                   {len(eager_ops - lazy_ops)} op_names\n")
        f.write(f"Lazy-only:                    {len(lazy_ops - eager_ops)} op_names\n\n")

        f.write("--- Per-trace cpu_leaf op_name count ---\n")
        f.write(f"{'trace':<32} {'n_cpu_leaf':>12} {'unique_ops':>12}\n")
        for tag in TAGS:
            if tag not in per_trace_cnt:
                continue
            c = per_trace_cnt[tag]
            f.write(f"{tag:<32} {sum(c.values()):>12} {len(c):>12}\n")

        # Aggregate counts and durations across all traces
        total_cnt = Counter()
        total_dur = Counter()
        for c in per_trace_cnt.values():
            total_cnt.update(c)
        for d in per_trace_dur.values():
            total_dur.update(d)

        f.write("\n--- Top 50 op_names by TOTAL count (all 8 traces) ---\n")
        f.write(f"{'rank':>4} {'op_name':<55} {'count':>10} {'cum %':>8}\n")
        cum = 0
        total = sum(total_cnt.values())
        for i, (op, c) in enumerate(total_cnt.most_common(50), 1):
            cum += c
            f.write(f"{i:>4} {op:<55} {c:>10} {100*cum/total:>7.1f}%\n")

        f.write("\n--- Top 50 op_names by TOTAL duration_ns (all 8 traces) ---\n")
        f.write(f"{'rank':>4} {'op_name':<55} {'µs':>14} {'cum %':>8}\n")
        cum = 0
        total = sum(total_dur.values())
        for i, (op, d) in enumerate(total_dur.most_common(50), 1):
            cum += d
            f.write(f"{i:>4} {op:<55} {d/1e3:>14.1f} {100*cum/total:>7.1f}%\n")

        f.write("\n--- Eager-only op_names (top 30 by count) ---\n")
        eager_only_cnt = Counter()
        for tag in TAGS:
            if tag.startswith("pytorch-eager"):
                for op, n in per_trace_cnt.get(tag, {}).items():
                    if op not in lazy_ops:
                        eager_only_cnt[op] += n
        f.write(f"{'op_name':<55} {'count':>10}\n")
        for op, n in eager_only_cnt.most_common(30):
            f.write(f"{op:<55} {n:>10}\n")

        f.write("\n--- Lazy-only op_names (top 30 by count) ---\n")
        lazy_only_cnt = Counter()
        for tag in TAGS:
            if not tag.startswith("pytorch-eager"):
                for op, n in per_trace_cnt.get(tag, {}).items():
                    if op not in eager_ops:
                        lazy_only_cnt[op] += n
        f.write(f"{'op_name':<55} {'count':>10}\n")
        for op, n in lazy_only_cnt.most_common(30):
            f.write(f"{op:<55} {n:>10}\n")


if __name__ == "__main__":
    main()
