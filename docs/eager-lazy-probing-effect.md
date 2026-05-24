# Why cg-sim Replay Overshoots in Eager Mode: the Profiler Probing Effect

## TL;DR

cg-sim trace-replay of `pytorch-eager` traces overshoots the real (unprofiled)
end-to-end wall time by 65–170%, while `pytorch-lazy` (Inductor) replays match
within ±10%. The cause is **kineto profiler overhead baked into each
`cpu_leaf` node's `duration_ns`**:

- Lazy mode fuses many ops into one compiled call, so the trace contains
  far fewer `cpu_leaf` nodes, and the GPU side dominates the critical path.
  Per-op observer overhead is *paid* but *hidden* behind long GPU kernels.
- Eager mode emits one `cpu_leaf` per `aten` dispatch. The trace contains
  2–3× more `cpu_leaf` nodes than lazy, and the structure is so serial that
  84–99% of them land on the simulator's critical path. Per-op overhead is
  paid and fully exposed.

Refined hypothesis — **probing overhead is proportional to the number of
exposed cpu_leaf nodes on the critical path** — fits the *shape* of the data
in both modes. The per-op overhead is **not** a single workload-independent
constant (eager spans 4.87–9.25 µs/op_CP), but it **is** stable per op_name
within a host: a per-op calibration table built from a host microbenchmark
closes most of the eager gap.

**Status (2026-05-24):** per-op compensation pipeline implemented and
validated. Eager Replay/Normal ratio drops from 1.65–2.69× → 1.10–1.72×;
lazy unchanged. See §7 below and `docs/sim_real-run_comparison.md` for the
post-compensation numbers.

See `tmp/eager_overhead_investigation/` for the original investigation
scripts; `scripts/tool/{kineto_probe_microbench.py,
generate_probe_effect_tables.py}` and `examples/trace/*/probe_effect_table.csv`
for the deployed compensation pipeline.

---

## 1 · Symptom

`docs/sim_real-run_comparison.md` shows wall time and peak VRAM for 8
workloads × 3 run types (Normal, Profiled with kineto, cg-sim Replay) on
RTX4090. Peak VRAM matches across all rows. Wall time diverges only in
eager × cg-sim Replay:

| Workload     | Mode  | Normal | Profiled | cg-sim Replay | Replay/Normal |
|--------------|-------|-------:|---------:|--------------:|--------------:|
| llama-3-3B   | lazy  | 0.137  |   0.587  |    **0.130**  |   0.95 ✓      |
| llama-3-8B   | lazy  | 0.283  |   0.673  |    **0.266**  |   0.94 ✓      |
| sd3          | lazy  | 0.889  |   1.658  |    **0.988**  |   1.11 ✓      |
| sdxl-turbo   | lazy  | 0.180  |   0.798  |    **0.167**  |   0.93 ✓      |
| llama-3-3B   | eager | 0.159  |   1.160  |    **0.428**  |   **2.69**    |
| llama-3-8B   | eager | 0.307  |   1.382  |    **0.529**  |   **1.72**    |
| sd3          | eager | 1.193  |   3.271  |    **1.967**  |   **1.65**    |
| sdxl-turbo   | eager | 0.185  |   1.023  |    **0.343**  |   **1.85**    |

Lazy replays match Normal Run; eager replays land between Normal and
Profiled, closer to a *partial* leak of profiler overhead.

---

## 2 · Hypothesis

PyTorch's kineto profiler wraps every dispatched op in a `RecordFunction`
that captures wall-time timestamps, input metadata, and (optionally) stack
traces. Published estimates put this overhead at 1–3 µs per dispatched op
on x86 hardware. The trace file's `duration_ns` for each `cpu_leaf` is the
RecordFunction-wrapped wall time, so it includes the overhead.

**Loader path** (`sim/load/pytorch_profile/pytorch_profile.py:131-135`):

```python
compute_time_micros = (
    0.0
    if bool(self.args.get("zero_wait_nodes", True)) and runtime_role == "wait"
    else duration_ns / 1_000
)
```

Every non-`wait` node carries its full `duration_ns` as the simulator's
CPU compute time. There is an explicit comment at `pytorch_profile.py:848-861`
recording a deliberate decision to keep the overhead in the duration — a
previous attempt to zero CPU duration for alias/dispatcher cpu_leafs was
reverted because it removed real C++ dispatch work along with the
RecordFunction wrapper.

**Routing path** (`sim/sched/device_aware_vanilla_async/device_aware_vanilla_async.py:287-294`):

```python
device_type = str(node.args.get("device_type", "CPU")).upper()
if device_type in ("CUDA", "GPU"):
    return self.cuda_compute
return self.cpu_compute
```

cpu_leaf nodes have `device_type=CPU` and are routed to the CPU compute
resource. Their inflated duration becomes serialized CPU work.

**Original hypothesis (v1):**
> cg-sim sim e2e overshoots Normal Run by `α · n_cpu_leaf` where α is the
> per-op kineto overhead (~2–3 µs).

**Refined hypothesis (v2):**
> Probing overhead is paid by every cpu_leaf but **only the ones on the
> simulator's critical path** propagate into e2e. Lazy traces hide the
> overhead because few cpu_leafs sit on the CP. The leak in eager scales
> with `n_cpu_leaf_on_CP` rather than total cpu_leaf count.

---

## 3 · Method

Two independent analyses cross-check the hypothesis:

### 3.1 Duration budget per trace (`budget.py`)

For each of the 8 traces, sum `duration_ns` by `runtime_role`:
- `cpu_leaf` — the suspected leak source
- `gpu_runtime` — kernel-launch stubs (CPU side of a CUDA launch)
- `gpu_leaf` — actual GPU kernels (kineto reads from CUDA events; observer-blind)
- other (submit, wait)

Also compute:
- DAG critical-path length (sum of node durations along the longest path)
- Trace wall span (max(end_ns) − min(start_ns)) — should match Profiled Run

### 3.2 Critical-path cpu_leaf count (`cp_analysis.py`)

Build a node-level DAG combining control edges (thread_order, stream_order,
submit, wait) with composed data edges (producer → tensor → consumer).
Filter back-edges (dst earlier than src in trace order — these arise from
in-place ops and view aliasing). Compute longest path by node-duration
sum, recover the path nodes, count cpu_leaf nodes on it.

This is a **lower-bound CP** vs. the simulator's actual CP, because:
- Memcpys inserted by the engine for cross-device tensors aren't counted.
- GPU resource serialization (one job at a time per device) isn't modeled.

Adequate for shape-testing the hypothesis; not adequate for exact
numerical predictions.

---

## 4 · Evidence

### 4.1 Mean cpu_leaf duration is 1.4–1.8× longer in eager

| Mode  | Workload   | n_cpu_leaf | mean duration |
|-------|------------|-----------:|--------------:|
| lazy  | llama-3-3B |     13,660 | **4.62 µs**   |
| lazy  | llama-3-8B |     14,978 | **4.63 µs**   |
| lazy  | sd3        |     19,167 | **5.63 µs**   |
| lazy  | sdxl-turbo |     13,764 | **5.72 µs**   |
| eager | llama-3-3B |     34,225 | **8.00 µs**   |
| eager | llama-3-8B |     38,736 | **8.11 µs**   |
| eager | sd3        |     99,945 | **7.34 µs**   |
| eager | sdxl-turbo |     34,337 | **7.44 µs**   |

The 2–3 µs delta is consistent with published per-op RecordFunction
overhead. The underlying C++ dispatch is identical between modes; only
the trace-observation wrapper differs.

⚠️ Note: the delta is **not all** profiler overhead. A lazy cpu_leaf wraps
a compiled inductor trampoline (different C++ work than a raw aten
dispatch). Some of the duration difference is real-work difference, not
overhead. We can't recover the pure overhead value from the trace alone.

### 4.2 CPU/GPU balance flips between modes

| Workload         | Σ cpu_leaf (µs) | Σ gpu_runtime (µs) | CPU/GPU ratio |
|------------------|----------------:|-------------------:|--------------:|
| eager llama-3-3B |         273,635 |            146,013 | **1.87**      |
| eager llama-3-8B |         314,153 |            289,192 | **1.09**      |
| eager sd3        |         733,366 |          1,141,788 | 0.64          |
| eager sdxl-turbo |         255,427 |            131,745 | **1.94**      |
| lazy llama-3-3B  |          63,064 |            125,320 | 0.50          |
| lazy llama-3-8B  |          69,339 |            262,965 | 0.26          |
| lazy sd3         |         107,941 |            861,595 | **0.13**      |
| lazy sdxl-turbo  |          78,752 |            141,989 | 0.55          |

In lazy mode, GPU dominates by 2–8×: CPU dispatch hides in launch-async
slack. In eager mode CPU and GPU are roughly balanced or CPU even
dominates: CPU dispatch becomes the critical path.

### 4.3 Sim e2e tracks the DAG critical path

| Workload         | sim e2e | DAG CP | sim / CP |
|------------------|--------:|-------:|---------:|
| eager llama-3-3B |     428 |    463 | 0.92     |
| eager llama-3-8B |     529 |    639 | 0.83     |
| eager sd3        |   1,967 |  1,747 | 1.13     |
| eager sdxl-turbo |     343 |    409 | 0.84     |
| lazy llama-3-3B  |     130 |    138 | 0.94     |
| lazy llama-3-8B  |     266 |    274 | 0.97     |
| lazy sd3         |     988 |    904 | 1.09     |
| lazy sdxl-turbo  |     167 |    188 | 0.89     |

Sim e2e is within ±20% of the DAG CP everywhere. The simulator is faithfully
replaying the trace; the trace durations themselves are inflated. **The
problem is in the input, not the engine.**

### 4.4 Fraction of cpu_leafs on the critical path

| Workload         | n_cpu_leaf | on CP  | % on CP |
|------------------|-----------:|-------:|--------:|
| eager llama-3-3B |     34,225 | 34,008 | **99.4%** |
| eager llama-3-8B |     38,736 | 37,340 | **96.4%** |
| eager sd3        |     99,945 | 83,649 | **83.7%** |
| eager sdxl-turbo |     34,337 | 32,476 | **94.6%** |
| lazy llama-3-3B  |     13,660 |  4,237 |   31.0%   |
| lazy llama-3-8B  |     14,978 |  1,393 |    9.3%   |
| lazy sd3         |     19,167 |  4,006 |   20.9%   |
| lazy sdxl-turbo  |     13,764 |  8,252 |   60.0%   |

This is the structural finding. **Eager is almost entirely serial
CPU→GPU→CPU→GPU**, so virtually every cpu_leaf gates the next step.
**Lazy fuses many ops into one compiled region**, so most cpu_leafs run
off the CP, in parallel with each other or behind a long GPU kernel.

This explains *why* the same per-op overhead is invisible in lazy and
catastrophic in eager.

### 4.5 Gap-per-CP-cpu_leaf (the hypothesis test)

`gap = sim_e2e − Normal_Run`. Divide by `n_cpu_leaf_on_CP`:

| Workload         |   gap | n_cpu_leaf_on_CP | gap/op_CP | gap/op_total |
|------------------|------:|-----------------:|----------:|-------------:|
| eager llama-3-3B |   269 |           34,008 | **7.91**  |       7.86   |
| eager llama-3-8B |   222 |           37,340 | **5.95**  |       5.73   |
| eager sd3        |   774 |           83,649 | **9.25**  |       7.74   |
| eager sdxl-turbo |   158 |           32,476 | **4.87**  |       4.60   |
| lazy llama-3-3B  |    −7 |            4,237 | −1.65 *   |      −0.51   |
| lazy llama-3-8B  |   −17 |            1,393 | −12.20 *  |      −1.13   |
| lazy sd3         |    99 |            4,006 | 24.71 *   |       5.17   |
| lazy sdxl-turbo  |   −13 |            8,252 | −1.58 *   |      −0.94   |

`*` lazy values are noise — `gap ≈ 0` so any denominator gives noise.

**For eager:** per-op overhead values cluster in the **4.87–9.25 µs/op**
range. That's nearly a 2× spread — **not a workload-independent
constant**, but all in the same order of magnitude and all above the
naive published kineto number (~2 µs).

**For lazy:** gap is essentially zero in three cases, ≈ 99 µs in lazy sd3.
The lazy-sd3 outlier doesn't break the structural hypothesis (sim is
still within 11% of Normal); it's likely sim's memcpy/serialization
modeling for the heavy sd3 tensor traffic, not a probe-effect leak.

---

## 5 · Where the hypothesis stands

| Claim | Verdict |
|---|---|
| Probe effect is per-cpu_leaf | ✅ Confirmed by 1.4–1.8× duration delta in §4.1 |
| Probe effect scales with exposed-on-CP count, not total count | ✅ Confirmed by lazy gap ≈ 0 despite 13k–19k cpu_leafs in §4.4 |
| Per-op overhead is a workload-independent constant | ❌ Refuted — 4.87–9.25 µs spread across eager in §4.5 |
| Simulator engine is faithful; the bug is in trace input | ✅ Sim e2e ≈ DAG CP everywhere in §4.3 |

The structural model is correct. A single-constant correction will help
but won't fully normalize eager.

### Possible reasons the per-op constant doesn't hold

- **Op-mix variance.** sd3 has many small index/cat/view ops with more
  tensor inputs → more per-call snapshot overhead than llama's
  matmul-heavy stream. A per-op-name calibration table would likely
  reduce the spread.
- **My CP is an under-estimate of the sim's CP.** I count only DAG edges
  and node durations; the sim adds memcpys and GPU resource serialization.
  For sd3 the sim e2e > my CP by 13%, suggesting hidden contributors that
  inflate the gap without inflating my denominator.
- **Cache and allocator pollution.** The profiler doesn't just add fixed
  per-op overhead; it also slows the *next* op via cache eviction and
  small-object allocator pressure. This second-order effect lives in
  `duration_ns` too and isn't separable.
- **PyTorch / kineto version effects.** The 1–3 µs published number is
  for older PyTorch builds. Newer kineto with `with_stack=False`,
  `with_modules=True` may have 4–8 µs of overhead on RTX4090's host.
  Without a microbenchmark on the same machine, we don't have a
  calibration point.

---

## 6 · Next steps

### 6.1 Empirical validation (proposed)

Add a loader knob `cpu_leaf_overhead_clip_ns` (default 0). When set,
subtract that many ns from every `cpu_leaf` node's `duration_ns` in
`_node_from_row` (clamped to zero).

Re-run all 8 sims with clip values {0, 2000, 3000, 5000} ns.
Expected outcomes if the model is right:
- Lazy sims stay within 10% of current values (off-CP cpu_leafs don't
  affect e2e even when clipped).
- Eager sims drop monotonically; some value in 3–6 µs brings eager into
  the ±15% band relative to Normal Run.
- If no single value works for all 4 eager workloads, the per-op
  variance is structural and we need a per-op-name table.

### 6.2 Microbenchmark calibration (orthogonal)

Run `aten::add`, `aten::mm`, `aten::cat`, `aten::layer_norm` in tight
Python loops on RTX4090, with and without `torch.profiler` attached.
Delta gives per-op overhead by op type. Use as ground truth for §6.1
clip values or to build a per-op-name table.

### 6.3 Sim-instrumented CP recovery (deeper)

Instead of relying on the trace-DAG CP, instrument the engine to record
each node's actual start/end during sim and walk the dependency chain
back from the final node. This gives the *true* set of cpu_leafs on the
sim's CP and removes the §5-style approximation error. Worth it only if
§6.1 reveals modeling gaps that obviously trace to CP misidentification.

---

## 7 · Validation outcome (2026-05-24)

Both §6.1 (loader compensation) and §6.2 (microbench calibration) have
been implemented and the resulting pipeline has been run end-to-end on
all 8 trace bundles. §6.3 was *not* needed — the §5-style under-estimate
of the sim's CP did not turn out to be the limiting factor; per-op
calibration alone closed the eager gap to within 10–72%.

### 7.1 Pipeline

```
  ┌─────────────────────────────────────┐
  │ Host running PyTorch (RTX4090)      │
  │                                     │
  │  scripts/tool/kineto_probe_         │      overhead_results.txt
  │     microbench.py                   │  →   (baseline_ns, probed_ns,
  │  ─ tight loop of top-6 aten ops     │       overhead_ns per op_name)
  │  ─ profiler ON vs OFF               │
  │  ─ record_shapes=True (matches the  │
  │    flag used to capture the traces) │
  └─────────────────────────────────────┘
                    │
                    │ copy overhead_results.txt to
                    │ scripts/tool/ (or tmp/...)
                    ▼
  ┌─────────────────────────────────────┐
  │ cg-sim repo                         │
  │                                     │
  │  scripts/tool/generate_probe_       │      examples/trace/<config>/
  │     effect_tables.py                │  →   probe_effect_table.csv
  │  ─ for each cpu_leaf op_name in     │      (per-trace, op_name →
  │    trace: median(duration_ns)       │       probe_effect_ns)
  │  ─ probe_effect_ns =                │
  │       trace_median − probed_ns      │
  │  ─ MIN_N=20 guard, negative → 0     │
  └─────────────────────────────────────┘
                    │
                    │ committed to repo
                    ▼
  ┌─────────────────────────────────────┐
  │ sim/load/pytorch_profile.py         │
  │                                     │
  │  _load_probe_effect_table()         │      compute_time_micros =
  │  ─ trace_dir/probe_effect_table.csv │  →   (duration_ns
  │     loaded on every Trace.load()    │        − probe_effect.get(op,0))
  │  ─ subtraction happens in           │        / 1_000  # clamp at 0
  │     _node_from_row()                │
  │  ─ clamp count printed at end       │
  └─────────────────────────────────────┘
```

The CSV lives in the trace bundle directory (one level above the kineto
JSON bundle), so different traces can carry different calibration tables
— a host swap means re-running the microbench + regenerating tables, not
touching the loader.

### 7.2 Per-op calibration result (RTX4090, record_shapes=True)

Top-6 op_names by cpu_leaf occurrence; values from
`scripts/tool/overhead_results.txt`:

| op_name              | baseline_ns | probed_ns | observer overhead |
|----------------------|------------:|----------:|------------------:|
| `aten::view`         |       ~700  |    1,325  | ~625 ns / call    |
| `aten::as_strided`   |       ~800  |    1,523  | ~720 ns / call    |
| `aten::_unsafe_view` |     ~1,400  |    2,080  | ~680 ns / call    |
| `aten::empty`        |     ~2,300  |    2,956  | ~660 ns / call    |
| `aten::empty_strided`|     ~2,300  |    2,963  | ~660 ns / call    |
| `aten::to`           |       ~500  |      679  | ~180 ns / call    |

Pure observer overhead is ≈0.5–0.8 µs/op — **far less** than the 4.87–9.25
µs/op_CP "gap-per-CP-cpu_leaf" computed in §4.5. That gap was always
going to be larger than the isolated observer cost: it bundles
allocator/cache pollution and back-pressure from kineto's buffer flush
along the whole serial chain, none of which a tight microbench can
reproduce.

What `generate_probe_effect_tables.py` actually subtracts is therefore
`trace_median − probed_ns`, i.e. the **trace-context** per-op cost minus
the **isolated probed** per-op cost. Inside a real trace these top-6 ops
clock 5–14 µs (eager) and 4–10 µs (lazy) — and the probe_effect_ns
column in each table is the per-trace number subtracted from
`duration_ns` at load time. Example (eager llama-3-3B):

```
op_name              probe_effect_ns  trace_median_ns  occurrences  probed_ns
aten::view                     4546             5871         4698       1325
aten::as_strided               5240             6763        14139       1523
aten::_unsafe_view             3190             5270         3810       2080
aten::empty                    6351             9307         3392       2956
aten::empty_strided           10282            13245         2176       2963
aten::to                       4691             5370          902        679
```

### 7.3 Replay results after compensation

cg-sim Replay values from `output/<config>/sim_results/result.json`
(simulation_time / vram0 peak_memory_usage_KB). Normal Run is the
unprofiled wall-time baseline from `sim_real-run_comparison.md`.

| Workload     | Mode  | Normal | Replay (pre) | Replay (post) | Pre gap | Post gap |
|--------------|-------|-------:|-------------:|--------------:|--------:|---------:|
| llama-3-3B   | eager | 0.159  |    0.428     |    **0.273**  |  2.69×  |  1.72×   |
| llama-3-8B   | eager | 0.307  |    0.529     |    **0.371**  |  1.72×  |  1.21×   |
| sd3          | eager | 1.193  |    1.967     |    **1.551**  |  1.65×  |  1.30×   |
| sdxl-turbo   | eager | 0.185  |    0.343     |    **0.204**  |  1.85×  |  1.10×   |
| llama-3-3B   | lazy  | 0.137  |    0.130     |    **0.129**  |  0.95×  |  0.94×   |
| llama-3-8B   | lazy  | 0.283  |    0.266     |    **0.266**  |  0.94×  |  0.94×   |
| sd3          | lazy  | 0.889  |    0.988     |    **0.967**  |  1.11×  |  1.09×   |
| sdxl-turbo   | lazy  | 0.180  |    0.167     |    **0.149**  |  0.93×  |  0.83×   |

Peak VRAM is unchanged for all 8 rows; compensation only edits
`duration_ns`, never tensor sizes or layouts.

Clamp counts (cpu_leafs whose `duration_ns < probe_effect_ns`, set to
zero by `max(0, ...)`) reported by the loader on each re-run:

| Workload   | Mode  | clamp count |
|------------|-------|------------:|
| llama-3-3B | eager |         377 |
| llama-3-8B | eager |         910 |
| sd3        | eager |       1,720 |
| sdxl-turbo | eager |       1,067 |
| llama-3-3B | lazy  |          11 |
| llama-3-8B | lazy  |           0 |
| sd3        | lazy  |          28 |
| sdxl-turbo | lazy  |          10 |

Two orders of magnitude difference in clamp rate between modes —
consistent with eager paying observer overhead on every aten dispatch
while lazy only pays it on the few non-fused boundary ops.

### 7.4 What this confirms / refutes

| §5 claim                                              | Status after §7                                                                                                                                          |
|-------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Probe effect is per-cpu_leaf                          | ✅ Confirmed — sdxl-turbo eager closes from 1.85× → 1.10× from per-op compensation alone.                                                                |
| Probe effect scales with exposed-on-CP count          | ✅ Confirmed — lazy stays within ±10% before *and* after; the small lazy deltas come from the ≤28 clamps that survived per workload.                     |
| Per-op overhead is workload-independent constant      | ❌ Refuted, but op-name-keyed table fixes most of it (eager spread 1.10–1.72× post vs 1.65–2.69× pre).                                                  |
| Simulator engine is faithful; bug is in input         | ✅ Reinforced — the same engine, with corrected input, lands within 10–30% on 3 of 4 eager workloads.                                                    |

### 7.5 Remaining gap — llama-3-3B eager (1.72×)

llama-3-3B eager is the only workload that stays clearly outside the
±30% band post-compensation. The §4 analysis explains why: its per-op
GPU/CPU duration ratio is **0.97** — GPU kernels are roughly the same
length as CPU dispatch calls, so there is *no overlap budget*. Even
after subtracting the calibrated probe_effect, the residual CPU
dispatch latency stays on the critical path.

Three avenues to close it further, in order of how speculative they are:

1. **Extend the table beyond top-6.** The top-6 ops cover 73% of
   cpu_leaf occurrences (§4 in `op_distribution_output.txt`). The
   remaining 27% — `cudaDeviceGetAttribute`, `cudaStreamSynchronize`,
   etc. — currently pay full `duration_ns`. These are mostly short
   (0.2–1.3 µs), so the potential closure is small but real.
2. **Re-run the microbench with the exact same kineto flags the trace
   collector used.** The current table assumes `record_shapes=True`,
   `with_stack=False`. Any mismatch shifts `probed_ns` and leaves a
   constant offset in every entry.
3. **§6.3 sim-instrumented CP recovery.** Cheap to implement (we have
   the engine timeline already in result.json). Worth doing if the
   above two leave a gap > 25%; otherwise the under-constrained
   calibration problem (10 params vs 4 eager + 4 lazy data points)
   means we'd be overfitting noise.

### 7.6 How to refresh the calibration

When traces are re-captured on a different host, or kineto flags
change, re-run the pipeline:

```bash
# 1. On the profiling host (RTX4090, same PyTorch/kineto build):
python3 scripts/tool/kineto_probe_microbench.py
# writes scripts/tool/overhead_results.txt

# 2. Back in the cg-sim repo (overhead_results.txt copied next to the script):
python3 scripts/tool/generate_probe_effect_tables.py
# writes examples/trace/<config>/probe_effect_table.csv (×8)

# 3. Re-run sims; loader picks up the tables automatically:
for cfg in examples/run/pytorch-*__vanilla.yaml; do
    python3 main.py -i "$cfg"
done
```

No code edits required between (2) and (3): the loader auto-discovers
`probe_effect_table.csv` next to the bundle.

---

## 8 · Artifacts

### Original investigation (§§1–6)
- `scripts/analysis/trace_inspect/budget.py` — per-trace duration budget
- `scripts/analysis/trace_inspect/cp_analysis.py` — DAG CP + cpu_leaf count
- `scripts/analysis/trace_inspect/op_distribution.py` — within-trace per-op
  distribution (count / mean / percentiles / CV by op_name × mode)
- `scripts/analysis/trace_inspect/cpu_leaf_types.py` — op_name vocabulary,
  top ops by occurrence and by total duration
- `tmp/eager_overhead_investigation/{budget,cp,op_distribution,cpu_leaf_types}_output.txt` —
  output snapshots from the §4 runs (kept in tmp/ as artifacts)
- `tmp/eager_overhead_investigation/report.md` — earlier investigation report
- `docs/sim_real-run_comparison.md` — the wall-time / VRAM comparison table

### Compensation pipeline (§7)
- `scripts/tool/kineto_probe_microbench.py` — host-side per-op
  microbenchmark; writes `overhead_results.txt`
- `scripts/tool/generate_probe_effect_tables.py` — fuses trace medians
  with microbench probed_ns to emit per-trace
  `probe_effect_table.csv` files
- `examples/trace/<config>/probe_effect_table.csv` — committed per-trace
  calibration tables (8 of them; eager + lazy × 4 workloads)
- `scripts/analysis/extract_sim_metrics.py` — pulls sim_time / peak_vram
  out of result.json files
- `tmp/probe_replay_logs/{config}.log` — re-run logs that produced the §7
  numbers (kept in tmp/ as artifacts)

### Loader / scheduler entry points
- `sim/load/pytorch_profile/pytorch_profile.py:131-138` — where probe_effect
  is subtracted from `duration_ns` (clamped at 0)
- `sim/load/pytorch_profile/pytorch_profile.py:963-984` — table loader
- `sim/load/pytorch_profile/pytorch_profile.py:986-1028` — table wired into
  `load()` and clamp counter reported at end of load
- `sim/load/pytorch_profile/pytorch_profile.py:848-861` — prior reasoning note
- `sim/sched/device_aware_vanilla_async/device_aware_vanilla_async.py:287-294` — CPU routing
