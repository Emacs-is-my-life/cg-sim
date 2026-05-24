"""
Within-trace distribution analysis of cpu_leaf duration_ns by op_name.

Goal: for each op_name, see whether the per-call duration is
  (a) tight across thousands of calls → per-op overhead is stable
  (b) tight across modes (lazy vs eager) → overhead is mode-invariant; the
      mode-dependence of e2e is purely structural (off-CP hiding)
  (c) different across modes → there's a mode-correlated overhead (cache /
      allocator / buffer-state) on top of per-op overhead

We extract for each (op_name, mode):
    count, mean, p10, p25, p50, p75, p90, max, CV = stdev/mean

Per-mode aggregation: we sum count and recompute mean over (lazy union all
4 lazy traces, eager union all 4 eager traces). Percentiles are computed
on the pooled raw samples.

Output: op_distribution_output.txt
"""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "examples" / "trace"
OUT  = REPO_ROOT / "tmp" / "eager_overhead_investigation" / "op_distribution_output.txt"

TAGS = [
    ("pytorch-eager__llama-3-3B",  "eager"),
    ("pytorch-eager__llama-3-8B",  "eager"),
    ("pytorch-eager__sd3",         "eager"),
    ("pytorch-eager__sdxl-turbo",  "eager"),
    ("pytorch-lazy__llama-3-3B",   "lazy"),
    ("pytorch-lazy__llama-3-8B",   "lazy"),
    ("pytorch-lazy__sd3",          "lazy"),
    ("pytorch-lazy__sdxl-turbo",   "lazy"),
]

# Limit deep distribution analysis to the top N op_names by total count
# (covers ~99% of cpu_leaf instances)
TOP_N = 12


def _parse_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def percentiles(xs, ps):
    if not xs:
        return [float("nan")] * len(ps)
    xs = sorted(xs)
    n = len(xs)
    out = []
    for p in ps:
        if n == 1:
            out.append(xs[0])
            continue
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            out.append(xs[int(k)])
        else:
            out.append(xs[f] * (c - k) + xs[c] * (k - f))
    return out


def scan_durs(bundle: Path):
    """Returns dict op_name -> list[duration_ns] for cpu_leaf nodes."""
    out = defaultdict(list)
    with (bundle / "runtime_nodes.csv").open() as f:
        for r in csv.DictReader(f):
            if r["runtime_role"] != "cpu_leaf":
                continue
            op = r["op_name"] or "<unknown>"
            out[op].append(_parse_int(r["duration_ns"]))
    return out


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "    -"
    return f"{x/1000:7.2f}"


def main():
    per_mode = {"eager": defaultdict(list), "lazy": defaultdict(list)}
    per_trace = {tag: defaultdict(list) for tag, _ in TAGS}

    for tag, mode in TAGS:
        bundle = ROOT / f"{tag}__RTX4090" / "llama_bundle"
        durs = scan_durs(bundle)
        for op, ds in durs.items():
            per_mode[mode][op].extend(ds)
            per_trace[tag][op].extend(ds)

    # Pick top-N ops across all 8 traces by total count
    total_cnt = defaultdict(int)
    for mode in per_mode:
        for op, ds in per_mode[mode].items():
            total_cnt[op] += len(ds)
    top_ops = [op for op, _ in sorted(total_cnt.items(), key=lambda x: -x[1])[:TOP_N]]

    with open(OUT, "w") as f:
        f.write("=== Per-op_name duration distribution: lazy vs eager ===\n")
        f.write(f"\nTop {TOP_N} cpu_leaf op_names by total occurrence:\n")

        # Header
        f.write("\n  All durations in µs. CV = stdev / mean.\n")
        f.write("\n")
        hdr = f"{'op_name':<32} {'mode':<6} {'n':>8} {'mean':>7} {'p10':>7} {'p50':>7} {'p90':>7} {'max':>7} {'CV':>5}\n"
        f.write(hdr)
        f.write("-" * (len(hdr) - 1) + "\n")

        for op in top_ops:
            for mode in ("eager", "lazy"):
                ds = per_mode[mode].get(op, [])
                if not ds:
                    f.write(f"{op:<32} {mode:<6} {'0':>8}  (absent)\n")
                    continue
                p10, p50, p90 = percentiles(ds, [0.10, 0.50, 0.90])
                mean = sum(ds) / len(ds)
                stdev = statistics.pstdev(ds) if len(ds) > 1 else 0.0
                cv = (stdev / mean) if mean > 0 else float("nan")
                f.write(f"{op:<32} {mode:<6} {len(ds):>8} "
                        f"{fmt(mean)} {fmt(p10)} {fmt(p50)} {fmt(p90)} {fmt(max(ds))} {cv:>5.2f}\n")
            f.write("\n")

        # Same-op lazy/eager comparison: ratio of means
        f.write("\n=== Same-op lazy vs eager mean comparison ===\n")
        f.write("If overhead is mode-invariant, eager_mean / lazy_mean ≈ 1.\n")
        f.write("If overhead is mode-correlated (cache/allocator/profiler-buffer state), ratio > 1.\n\n")
        f.write(f"{'op_name':<32} {'lazy_mean':>10} {'eager_mean':>11} {'ratio':>7} {'delta_us':>9}\n")
        f.write("-" * 73 + "\n")
        for op in top_ops:
            le = per_mode["eager"].get(op, [])
            ll = per_mode["lazy"].get(op, [])
            if not le or not ll:
                continue
            me = sum(le) / len(le) / 1000  # µs
            ml = sum(ll) / len(ll) / 1000  # µs
            ratio = me / ml if ml > 0 else float("nan")
            f.write(f"{op:<32} {ml:>10.2f} {me:>11.2f} {ratio:>7.2f} {me-ml:>+9.2f}\n")

        # Per-trace mean for the very top ops (sanity: does it vary within a mode?)
        f.write("\n\n=== Per-trace mean for top-6 ops (sanity check) ===\n")
        f.write("If within-mode means are tight, op-level overhead is workload-independent.\n\n")
        top6 = top_ops[:6]
        f.write(f"{'op_name':<32} " + " ".join(f"{tag:>14}" for tag, _ in TAGS) + "\n")
        f.write("-" * (33 + 15 * len(TAGS)) + "\n")
        for op in top6:
            cells = []
            for tag, _ in TAGS:
                ds = per_trace[tag].get(op, [])
                if not ds:
                    cells.append(f"{'-':>14}")
                else:
                    cells.append(f"{sum(ds)/len(ds)/1000:>14.2f}")
            f.write(f"{op:<32} " + " ".join(cells) + "\n")


if __name__ == "__main__":
    main()
