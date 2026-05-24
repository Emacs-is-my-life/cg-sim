# RTX4090 Environment (e2e time, peak VRAM)
## PyTorch Mode
- Lazy (Inductor): Does compute graph optimizations, like kernel fusing
- Eager: Runs PyTorch in eager mode.

## Run Type
- Normal Run: Ordinary pytorch execution
- Profiled Run: Pytorch run with profiler(kineto, trace observer) attached. Causes large overhead.
- cg-sim Replay: Using pytorch traces obtained from Profiled Run, replay that trace in cg-sim

## Result
| PyTorch Mode    | Run Type      | llama-3-3B       | llama-3-8B       | sd3              | sdxl-turbo       |
|-----------------|---------------|------------------|------------------|------------------|------------------|
| Lazy (Inductor) | Normal Run    | 0.137s, 6.1 GB   | 0.283s, 15 GB    | 0.889s, 15 GB    | 0.180s, 6.6 GB   |
| Lazy (Inductor) | Profiled Run  | 0.587s, 6.2 GB   | 0.673s, 15 GB    | 1.658s, 17 GB    | 0.798s, 8 GB     |
| Lazy (Inductor) | cg-sim Replay | 0.129s, 6.09 GB  | 0.266s, 15.16 GB | 0.967s, 16.54 GB | 0.149s, 7.55 GB  |
| Eager           | Normal Run    | 0.159s, 6.1 GB   | 0.307s, 15 GB    | 1.193s, 15 GB    | 0.185s, 6.6 GB   |
| Eager           | Profiled Run  | 1.16s, 6.2 GB    | 1.382s, 15 GB    | 3.271s, 16 GB    | 1.023s, 7 GB     |
| Eager           | cg-sim Replay | 0.273s, 5.99 GB  | 0.371s, 14.96 GB | 1.551s, 15.48 GB | 0.204s, 6.78 GB  |

cg-sim Replay numbers above are with the per-op probe-effect compensation
table applied (`examples/trace/*/probe_effect_table.csv`, loaded by
`sim/load/pytorch_profile`). The eager rows previously read 0.428 /
0.529 / 1.967 / 0.343s — pre-compensation.

## Findings: why eager Replay overshoots Normal Run

**Symptom (pre-compensation).** Peak VRAM matches across all rows. Wall time
matches for lazy Replay (within ±10% of Normal). Eager Replay overshoots
Normal by 65–169%.

**After per-op probe-effect compensation.** Eager gap shrinks across the
board:

| Eager workload | Normal | Replay (pre) | Replay (post) | Pre gap | Post gap |
|----------------|-------:|-------------:|--------------:|--------:|---------:|
| llama-3-3B     | 0.159s |   0.428s     |    0.273s     |  2.69×  |  1.72×   |
| llama-3-8B     | 0.307s |   0.529s     |    0.371s     |  1.72×  |  1.21×   |
| sd3            | 1.193s |   1.967s     |    1.551s     |  1.65×  |  1.30×   |
| sdxl-turbo     | 0.185s |   0.343s     |    0.204s     |  1.85×  |  1.10×   |

Compensation subtracts the per-op `probe_effect_ns` (trace_median − microbench
probed cost; see `docs/eager-lazy-probing-effect.md`) from every cpu_leaf
`duration_ns` at load time, clamped at 0. Workloads where the dispatch path
itself dominates (sdxl-turbo, llama-3-8B) close to within 10–21%; llama-3-3B
still overshoots because its per-op GPU/CPU ratio is 0.97 — even compensated
CPU dispatch stays on the critical path.

**Thesis.** The kineto profiler used to record the trace embeds per-op observer
overhead (RecordFunction wrapper, input snapshotting, callback chain) into
every `cpu_leaf` node's `duration_ns`. cg-sim faithfully replays that inflated
duration. The overhead surfaces in sim e2e **only when those cpu_leaf nodes sit
on the simulator's critical path**.

In lazy mode, Inductor fuses many ops into one compiled call → few cpu_leafs,
most hidden behind long GPU kernels (9–60% on CP). Overhead is paid but
hidden.

In eager mode, every aten dispatch is its own cpu_leaf → 2–3× more of them,
and 84–99% sit on the CP. Overhead is paid and fully exposed.

**Per-workload divergence is set by the per-op GPU/CPU duration ratio.**

| Eager workload | mean gpu_runtime | mean cpu_leaf | GPU/CPU | sim/Normal |
|----------------|-----------------:|--------------:|--------:|-----------:|
| llama-3-3B     |       7.77 µs    |     8.00 µs   |  0.97   |  **2.69×** |
| llama-3-8B     |      13.30 µs    |     8.11 µs   |  1.64   |    1.72×   |
| sd3            |      20.80 µs    |     7.34 µs   |  2.84   |    1.65×   |
| sdxl-turbo     |      11.96 µs    |     7.44 µs   |  1.61   |    1.85×   |

When per-op GPU kernel time ≈ per-op CPU dispatch time (llama-3-3B), CPU
has zero overlap budget → divergence is highest. As GPU kernels lengthen
with model size, CPU hides behind GPU and divergence drops.

**Key evidence supporting the thesis:**

1. **Engine is faithful.** cg-sim e2e tracks the trace's DAG critical-path
   length within ±20% on all 8 traces. The bug is in the input
   (`duration_ns`), not in the simulator.
2. **Per-op duration is mode-invariant.** Same op_name (e.g. `aten::view`)
   has mean duration 6.09 µs in lazy and 7.37 µs in eager — ratio 1.21.
   Lazy doesn't have cheaper per-op cost; it has structurally hidden cost.
3. **CPU-leaf op vocabulary is small.** Only 38 distinct op_names across
   all 8 traces; the top 4 (`aten::as_strided`, `aten::view`, `aten::empty`,
   `cudaDeviceGetAttribute`) cover 73% of cpu_leaf occurrences. Bimodal
   duration structure: aten metadata ops 5–15 µs, cuda* introspection ops
   0.2–1.3 µs.
4. **CP cpu_leaf fraction explains lazy vs eager flip:** 9–60% on CP in
   lazy → leak hidden; 84–99% on CP in eager → leak exposed.

**Open question (gated on microbenchmark).** Per-op observer overhead α is
not directly measurable from existing traces — every `duration_ns` we have
was recorded with kineto attached. A microbenchmark on the RTX4090 host
will measure α by comparing each top-6 aten op with profiler ON vs OFF.
Until α is known we can't predict how much of the eager gap is closable by
per-op clipping (Reading A: ~15%; Reading B: ~50%; details in
`docs/eager-lazy-probing-effect.md`).

See `docs/eager-lazy-probing-effect.md` and
`tmp/eager_overhead_investigation/` for the full investigation.
