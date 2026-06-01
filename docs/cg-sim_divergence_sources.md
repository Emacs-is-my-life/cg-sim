# Sources of Divergence between cg-sim and Real Offload Runs

This document catalogs **why** cg-sim's wall-time predictions diverge from
real-world offload runs (accelerate cpu_offload, diffusers group_offload),
and lays out the hierarchy we use to decide **where** to close each gap.

The catalog and hierarchy here are the durable working model the team
relies on when deciding what to fix and how. Verification targets and
the bandwidth-calibration compromise are documented separately in
`docs/cg-sim_bandwidth_calibration.md` and the verification table in
`CLAUDE.md`.

> **DIRECTION DECISION (current): approach A, trace-driven.** We have stopped
> trying to make the scheduler *synthesize* offload behavior from the eager
> trace (approach B). Two evidence-backed reasons: (1) the accelerate/diffusers
> documentation is inaccurate (record_stream, matched-block eviction D6, the
> stream split), so a doc-derived policy is wrong; (2) e2e time is dominated by
> microscopic CPU/sync overhead (~1 ms/leaf, D1/D2) that simply isn't in the
> eager trace. **Conclusion: without the offload run's own trace — the most
> accurate description of the scheduler's real behavior — the simulator cannot
> match the real run's e2e time.** So we feed the simulator the offload trace
> and have the loader recognize+mark its real transfers (approach A), which
> turns most of D1-D8 from "guess the policy" into "read the policy." The full
> A plan, design options, and first steps are in `TODO.md` ("DIRECTION
> CHANGE"). The 4-step hierarchy and the profile-setup recipe below remain
> valid reference but describe the superseded B path.

## D1–D8: divergence sources

| ID | Source | Affects | Status |
|---|---|---|---|
| **D1** | Eager trace input lacks the offload scheduler's hook CPU work. Eager run had no hooks installed, so `aten::to` / `set_module_tensor_to_device` / `cudaStreamSynchronize` / `cuMemMap/Create/Release/Unmap` per leaf is absent. | Both schedulers | Compensated (Step 4 phantom) |
| **D2** | Eager profiler did not capture Python frames (`with_stack=True` was off at record time). ~700 µs/leaf of interpreter time is invisible to the loader. | Both schedulers | Rolled into D1 phantom (empirical) |
| **D3** | Single global memory bandwidth in cg-sim. Real systems have multiple paths: pinned HtoD (~25 GB/s), pageable HtoD (~13 GB/s), pageable DtoH (workload-dependent). cg-sim's `memory_bandwidth_KBps` is one knob per memory hardware. | Diffusers (accelerate is all-pageable) | Compromise via 13 GB/s flat |
| **D4** | No CUDA stream model in cg-sim. Real diffusers runs matched-group H2D on pinned side stream 13 and pageable D2H on compute stream 7 — they are physically independent. cg-sim's water-filling forces all transfers to share one bandwidth pool. | Diffusers | Compromise via 13 GB/s flat |
| **D5** | Batched TransferJob vs per-tensor `cudaMemcpyAsync`. Real diffusers fires one Memcpy per parameter (22,317 for SD3); cg-sim batches into one TransferJob per group (651 for SD3). Per-event characteristics get coalesced — a 100-250× event-count collapse. | Both, worse for diffusers | Not addressed |
| **D6** | `record_stream=False` matched-group eviction emits **real D2H bytes** in practice, not pointer-swap as documented. Verified empirically: SDXL UNet 3.36 GB / 4 steps, SD3 transformer 18.9 GB / 4 steps, all kind=WEIGHT events. Our scheduler models pointer-swap. | Diffusers | Not addressed (fix interacts with D3/D4) |
| **D7** | SD3 per-event D2H wall is ~7.5× larger than `bytes / bandwidth` predicts. Real SD3 D2H ≈ 2.83 GB/s effective vs H2D 21.4 GB/s. SDXL doesn't show this asymmetry. | SD3 specifically | Not addressed |
| **D8** | Tied weights collapsed by cg-sim's tid-per-storage model. e.g. Llama-3.2-3B's `embed_tokens.weight` ⇔ `lm_head.weight` share storage; accelerate transfers each separately; cg-sim's `_resolve_paged_leaves` was attributing the single tid to only the first owner module. | Accelerate (any tied-weight model) | **Fixed** in scheduler |

## The 4-step correction hierarchy

When closing a divergence source, prefer the highest (cheapest) step
that does the job. Each step adds more complexity to the simulator
side; lower steps mean less risk and broader applicability.

1. **Profile setup** — use real-run settings that produce a trace
   cg-sim can faithfully model (`with_stack=True`, `record_stream`,
   `leaf_level` vs `block_level`, `offload_buffers=True`, etc.). No
   simulator code changes. Cheapest. Limitation: only verifies the
   chosen profile mode, not arbitrary configurations.
2. **Simulator core** (`sim/core/`) — architectural changes:
   per-direction bandwidth, multi-stream lanes, per-tensor TransferJob,
   etc. Most expensive and highest risk; foundational.
3. **Trace loader** (`sim/load/pytorch_profile/`) — correctness in
   translating PyTorch trace into cg-sim's representation (tied
   storage, alias detection, weight vs context classification,
   Python-frame mapping when `with_stack=True`).
4. **Scheduler** (`sim/sched/`) — faithful model of the real
   offload runtime (per-leaf hook emission, eviction semantics,
   prefetch chains). Most downstream; can become per-workload
   tuning if not disciplined.

**Default rule**: try Step 1 first, close as much gap as possible at
each level, then move to the next. Allow short-circuit when a
downstream fix is much cheaper *and* the upstream fix is genuinely
expensive (e.g., D8 was easier to fix in the scheduler than the
loader, even though loader is "more upstream" — pragmatic exception).

**Boundary statement**: Step 1 changes the verification target, not
just the modeling. Re-profiling diffusers with `leaf_level` means we
verify "cg-sim predicts wall time for `leaf_level`" — not "cg-sim
predicts wall time for any diffusers config." That scope limitation
needs to be visible wherever the verification result is claimed.

## D-to-step mapping

| Source | Step 1 (re-profile) | Step 2 (sim core) | Step 3 (loader) | Step 4 (scheduler) |
|---|---|---|---|---|
| D1 | Re-profile with config the simulator can model with minimal scheduler help | — | — | Phantom CPU emission (current workaround) |
| D2 | `with_stack=True` at profile time | — | Parse Python frames into cpu_leaf nodes | Rolled into D1 phantom otherwise |
| D3 | Use single-path config (leaf_level + no stream, or accelerate-like) | Per-path bandwidth pools | `tensor.args["transfer_path"]` tag | Scheduler tags tids per path |
| D4 | Use `use_stream=False`, or leaf_level with one active stream at a time | Stream/lane model | — | Per-scheduler stream assignment |
| D5 | — | Per-tensor TransferJob, or per-batch event multiplier | — | Scheduler emits per-tensor jobs |
| D6 | `--record-stream` flag makes intent explicit; or leaf_level may sidestep | — | — | Scheduler emits matched-block D2H (requires D3/D4 fix to not break SDXL) |
| D7 | Likely disappears with `leaf_level`/no-stream; need re-profile to confirm | Per-direction bandwidth | — | — |
| D8 | — | — | Better tied-storage modeling (future) | **Fixed** — `_resolve_paged_leaves` accumulates all leaf consumers |

## Current state (as of `c6071fd` on `sim-test`)

### Compensations applied
- **Phantom CPU hook nodes** (Step 4): emitted by both `AccelerateCpuOffload`
  and `DiffusersGroupOffload` in `compile()`. Calibration table on
  `DeviceAwareVanillaAsync._post_xfer_cpu_us`:
  - `pageable`: 1090 µs/transfer (from Llama 3B/8B inter-Memcpy gap;
    two-trace agreement Δ<1%)
  - `pinned`: 70 µs/transfer (SDXL inter-Memcpy gap; single-trace
    calibration)
  This absorbs D1+D2 for accelerate. For diffusers it does partial
  work; the rest is absorbed by D3/D4 bandwidth compromise.
- **Tied-weight fix** (Step 3, scheduler-as-workaround): `_resolve_paged_leaves`
  in `AccelerateCpuOffload` accumulates all leaf modules that consume a
  WEIGHT tid, not just the first one. Closes D8.
- **PCIe bandwidth compromise** (Step 4-ish via YAML): SDXL/SD3 YAMLs
  use 13 GB/s flat instead of per-path 25 + 13. Documented in
  `docs/cg-sim_bandwidth_calibration.md`. Absorbs D3/D4/partial D6.

### Verification status

| Workload | Sim wall | Real wall | Δ wall | Δ VRAM | Bound |
|---|---|---|---|---|---|
| Llama 3B accelerate | 11.87 s | 11.84 s | +0.3% | −8.6% | ✅ |
| Llama 8B accelerate | 21.85 s | 21.85 s | +0.02% | −8.4% | ✅ |
| SDXL diffusers (block_level + use_stream) | 2.11 s | 2.60 s | −19.0% | +1.8% | ⚠️ in bound by D3⇄D6 cancellation |
| SD3 diffusers (block_level + use_stream) | 11.51 s | 25.33 s | −54.5% | −6.0% | ❌ |

Bound = ±20% e2e wall, ±10% peak VRAM. SDXL passes by coincidence;
SD3's byte ratios break the same cancellation.

## Recommended next step

Re-profile diffusers SDXL and SD3 with **`leaf_level + use_stream=True + with_stack=True`**. This is the maximum Step-1 leverage available:

- Collapses two transfer paths into one (helps D3, D4).
- Pointer-swap eviction expected at leaf level (helps D6).
- May resolve D7 (SD3 D2H asymmetry likely tied to block_level + matched-D2H semantics).
- `with_stack=True` captures Python frames so D2 can be loaded directly into trace nodes; phantom can be reduced or removed.
- Keeps the compute/transfer overlap demonstration that `use_stream=True` provides — and cg-sim handles that overlap correctly today (TransferJob and ComputeJob touch different hardware, so water-filling doesn't make them compete).
- Trade-off: more transfer events (per-leaf vs per-block). Analysis showed the per-event calibration noise is far smaller than the current ~250× event-count under-count in block_level.

After re-profile, verify both schedulers against the new traces with
the existing scheduler logic (a small leaf-level variant may be useful
in `DiffusersGroupOffload`, or AccelerateCpuOffload may apply directly).
If any workload still misses bound, the remaining gap will have
isolated, named sources from the D1-D8 catalog — and we can decide
whether the next step is core, loader, or scheduler-level fix on
specific evidence.

For accelerate, also re-profile with `offload_buffers=True` so the
RoPE `inv_freq` buffers are transferred through the same offload
mechanism as parameters (eliminates an asymmetry between sim and
real).

## Profiling-setup recommendation in detail

What to set when re-profiling, and what each flag buys.

### Diffusers SDXL and SD3

```
--group-offload-type leaf_level     # one transfer path, simulator-modelable
--group-offload-use-stream          # demonstrate compute/transfer overlap
                                    # (cg-sim handles it natively: TransferJob
                                    #  on memory hw runs concurrently with
                                    #  ComputeJob on gpu hw — no shared
                                    #  running_on, no water-filling collision)
--with-stack                        # capture Python frames so D2's ~700 µs/leaf
                                    # interpreter time appears in the trace at
                                    # loader level (eliminates need for scheduler
                                    # phantom to absorb it empirically)
--profile-memory                    # for VRAM peak verification
--record-shapes                     # tensor metadata
--with-modules                      # module_path attribution
```

Leave at defaults:
- `record_stream` (default False) — empirical behavior of False is already documented from prior traces; True is an unknown the simulator hasn't been calibrated against. If leaf_level+False still emits matched-block D2H bytes (we'd see it in the new trace), revisit then.
- `non_blocking` (default False) — use_stream=True already gives async-side-stream semantics; this knob matters more when use_stream is False.

### Accelerate Llama 3B / 8B

Current config (Llama 3B/8B `accelerate.cpu_offload`) is already at <1% sim-vs-real; re-profile isn't strictly required. If re-profiling anyway, add:

```
--offload-buffers                   # offload RoPE inv_freq buffer (currently
                                    # stays on GPU permanently). Tiny perf
                                    # cost; removes a buffer/parameter
                                    # asymmetry between sim and real.
--with-stack                        # captures Python frames; lets future
                                    # work move the per-leaf phantom from
                                    # scheduler-emitted to loader-loaded.
```

### What this configuration buys

| Source | How it gets closed by Step 1 |
|---|---|
| D1 hook CPU chain | `with_stack=True` records the chain into the trace; loader can map it to cpu_leaf nodes; scheduler phantom can be reduced or removed. |
| D2 Python frames | `with_stack=True` captures them directly. |
| D3 single bandwidth | `leaf_level` collapses two transfer paths into one — one cg-sim bandwidth knob is now physically correct. |
| D4 no stream model | `leaf_level + use_stream=True` has one active transfer at a time; cg-sim's hardware-resource model handles compute/transfer overlap natively. |
| D6 `record_stream=False` D2H bytes | Likely sidestepped at leaf-level (one-leaf-at-a-time eviction is structurally simpler than block-level). Verify against new trace. |
| D7 SD3 D2H asymmetry | Hypothesis: tied to block_level matched-block eviction pattern. Likely disappears at leaf_level. Verify against new trace. |

### What this configuration gives up

1. **Doesn't verify the production-optimized `block_level` config.** Production users running diffusers prefer block_level (fewer hook fires per step → better perf). The verification result is explicitly bounded to `leaf_level + use_stream=True`. Anyone reading the verification needs to see this scope statement.
2. **Slightly slower real wall time** at recording time (more per-event hook overhead × more events). Irrelevant for simulator verification (we're checking prediction accuracy, not optimizing perf).
3. **D5 (batched TransferJob vs per-tensor) doesn't disappear** — but its impact shrinks because leaf_level groups are size 1, so sim's "one tid per TransferJob" matches reality's one cudaMemcpyAsync per tid. The 100-250× collapse becomes ~1×.

### What is unknown until the new traces arrive

1. **Does `leaf_level + record_stream=False` actually pointer-swap?** Block-level didn't (D6). If leaf-level also emits real D2H bytes per leaf, the trace will show it and we decide between (a) emitting matched D2H in the scheduler, (b) re-profile with `record_stream=True`, or (c) document and accept residual gap.
2. **Is D7 truly block_level-specific?** Hypothesis above. Only confirmable by inspecting the new SD3 D2H per-event bandwidth.
3. **How much of the per-leaf phantom (1090 µs pageable / 70 µs pinned) is already in `with_stack=True` traces?** First new trace tells us how much the scheduler phantom can be reduced.

### Verification expectations after re-profile

If the hypothesis above is right:
- All four workloads (Llama 3B + 8B accelerate, SDXL + SD3 diffusers-leaf-level) should land within ±10-20% e2e and ±10% VRAM with the existing scheduler logic (possibly a small leaf-level scheduler variant for diffusers — even reusing `AccelerateCpuOffload` directly might work).
- One calibration knob (pageable per-transfer phantom ~1090 µs), one bandwidth (literal per-Memcpy measurement, no compromise), one scheduler logic.

If a workload still misses bound after re-profile, the gap is now from named sources (D5, residual D6/D7) on isolated evidence, and a focused step-2/3/4 decision can follow.
