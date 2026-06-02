# Diffusers `group_offload` in cg-sim

How cg-sim reproduces a HuggingFace `diffusers` **`group_offload`** run (SDXL-turbo,
SD3), the decisions that made a diffusion-transformer trace replayable in cg-sim,
and the **honest sources of the remaining VRAM-peak gap.** Like the accelerate doc,
this is **not** a byte-for-byte clone of the recorded run; every modelling decision
and its residual error are spelled out below.

- Loader: `sim/load/pytorch_offload_loader/` (`PytorchOffloadLoader`, shared with
  accelerate; `offload_variant` auto-detected, `offload_reconstruct` arg).
- Scheduler: `sim/sched/diffusers_group_offload/` (`DiffusersGroupOffload`) — a thin
  subclass of `AccelerateCPUOffload`.
- Configs: `examples/run/diffusers-group-offload__{sdxl-turbo,sd3}__diffusers.yaml`.
- Companion: `docs/accelerate_cpu_offload.md` (the shared mechanism + engine
  constraints; this doc covers only the diffusers-specific delta).

---

## 1. What the real policy does

`diffusers` `group_offload(offload_type="leaf_level", use_stream=True,
non_blocking=True, record_stream=False, low_cpu_mem_usage=False)` offloads the model
in **leaf-module groups**: each leaf's parameters live pinned in CPU RAM and are
streamed **H2D just before the leaf computes**, then **dropped** (a pointer-swap back
to the CPU master — *no* device→host copy, since inference weights are read-only).

The one execution difference from accelerate's synchronous `cpu_offload`: with
`use_stream=True` the onloads run on a **dedicated side CUDA stream and overlap
compute** — while leaf *i* computes, leaf *i+1*'s weights are already streaming. This
prefetch is why group_offload is much faster than accelerate cpu_offload at similar
VRAM.

cg-sim's goal: reproduce the **inference-scope VRAM peak**, the **H2D re-stream
volume** (each weight streamed once per diffusion step), and the **end-to-end time**
on true (non-profiled) hardware.

Measured from the traces (do not re-measure — see `tmp/measure_diffusers.py`):

| | sdxl-turbo | sd3 |
|---|---|---|
| H2D events (`Pinned→Device`) | 7 435 | 20 050 |
| D2H events (eviction) | 3 | 2 |  ← pointer-swap, **no copy-back** |
| total H2D volume | 22.18 GB | 121.10 GB |
| **pinned bandwidth (aggregate)** | **26.37 GB/s** | **26.57 GB/s** |
| masters (cpu-source storage) | 2 400 | 1 851 |
| compute on stream 7; H2D on side streams | 13/17/21 | 13/17/21/25 |

## 2. Relation to accelerate: same mechanism, overlap from the trace

The offloaded-weight machinery is **identical** to accelerate cpu_offload (one RAM
master per weight, re-streamed per use, freed after use, activations refcount-freed).
The *only* execution-axis difference — H2D∥compute overlap — is **carried by the
trace, not the scheduler**:

- The loader translates each leaf's `Memcpy HtoD (Pinned → Device)` (recorded on a
  **side stream**, distinct from compute stream 7) into a transfer trigger whose
  `stream_order` parent is the **previous side-stream Memcpy**, not the compute
  kernel. So consecutive H2D serialize among themselves (one side stream ⇒ full
  pinned bandwidth) while the compute kernels form their own chain. The engine runs
  a `ComputeJob` (gpu) and the next `TransferJob` (memory bandwidth) concurrently —
  **the prefetch overlap emerges for free.** (Contrast accelerate: every node is on
  stream 7, so each Memcpy's `stream_order` parent is the prior compute ⇒ serial.)

Because the mechanism is shared and the overlap is in the trace, **`DiffusersGroupOffload`
is a thin subclass of `AccelerateCPUOffload`** — it changes exactly one behaviour
(§4.4) and toggles one knob (§5.4). Everything else is inherited.

## 3. How cg-sim models it (architecture)

- **One loader, auto-detected variant.** `PytorchOffloadLoader` serves both accelerate
  and diffusers; `offload_variant` is auto-detected from the
  `cudaMemcpyAsync`/`Memcpy HtoD` duration ratio (≥0.5 ⇒ "accelerate", a blocking
  copy; <0.5 ⇒ "diffusers", an async side-stream copy). The accelerate path is
  byte-identical after this unification (verified).
- **Shared executor.** Loader emits `trace.args["offload"]={master_tids,
  evict_after_node}` + per-Memcpy `offload_transfer` tags; the scheduler fires one
  same-tid RAM→VRAM `TransferJob` per trigger, schedule-evicts masters, refcount-frees
  intermediates. (Full mechanics: `docs/accelerate_cpu_offload.md` §3.)

---

## 4. Decisions that make the diffusers trace replayable

These make the trace **run** correctly (no deadlock, no wrong reads). The VRAM-peak
fidelity decisions are in §5.

### 4.1 Variant auto-detection (cpu-scalar kernel-args dropped)
Tiny (≤ 4096 B) non-master **cpu** tensors read by gpu kernels are **by-value kernel
arguments**, not VRAM reads — e.g. `AUnaryFunctor<MulFunctor<float>>` captures a
scalar (diffusion sigmas/scales from `aten::sub`/`aten::empty`). The loader drops
them from gpu `input_tensors` (+ a control edge to the producer) so the gpu op isn't
gated on a non-existent VRAM residency. A larger-than-threshold one is left in place
and **WARNs** (possible missing transfer). *(Shared with accelerate; no-op there.)*

### 4.2 HOL residency gate for side-stream prefetch
`job_waiting` is FIFO with head-of-line blocking. A diffusers gpu compute can become
*ready* before its weight's **cross-stream** transfer lands (the H2D is on a side
stream, not a `stream_order` parent of the compute). Parking such a not-yet-runnable
node at the FIFO head would block the very transfer that makes it runnable ⇒
"Deadlock detected". `AccelerateCPUOffload._submit_ready` defers a gpu compute whose
cuda inputs aren't VRAM-resident yet (`_inputs_resident_vram`), skipping `custom_deps`
nodes. **No-op for accelerate** (its gpu nodes are `stream_order` children of their
own Memcpy, so resident once ready).

### 4.3 Skip pre-claiming **orphan** dispatcher outputs
Diffusers eager traces allocate transient activation buffers via
`aten::empty(device=cuda)` on the cpu thread (cross-device "dispatcher" outputs the
base scheduler pre-claims in VRAM). The majority (SDXL: ~4135 of 4780) have **no
consumer** in the rewritten graph — the framework reads/writes them through aliased
views / untraced pointers, and the genuine gpu-produced activation that overwrites
the same bytes is tracked separately. Holding these orphans double-counts VRAM and
leaks (no consumer ⇒ refcount-free never fires). `DiffusersGroupOffload`'s sole
override skips pre-claiming orphan dispatcher outputs (still pre-claims the genuinely-
consumed ones).

### 4.4 Pinned bandwidth, measured per-trace
`SimpleRAM`/`SimpleVRAM` use one linear rate set to each trace's measured aggregate
pinned H2D rate: **26.37 GB/s (SDXL) / 26.57 GB/s (SD3)** — ~2× accelerate's pageable
rate, the real difference between pinned and pageable host memory. (KBps in config =
bytes/s ÷ 1000: 26.37 GB/s → `26370000`.) Measured, never back-fit to the e2e target.

---

## 5. Decisions that make the VRAM peak accurate

The VRAM target is the **inference-scope** peak — the kineto trace's own running
`Total Allocated` high-water, **not** `manifest.vram_peak_allocated_bytes` (1513 MiB,
which is whole-process incl. the out-of-scope VAE decode + warmup). Confirmed targets
(`tmp/kineto_vram_timeline.py`): **SDXL 129.2 MiB, SD3 464.4 MiB**. The naive
last-consumer residency over-shot these by ~35 % / ~278 %; the fixes below close
SDXL and bring SD3 to +14.5 %.

### 5.1 Stale-duplicate reclamation (double-claim)
A tid produced twice (a dispatcher pre-claim **and** a normal producer) gets a second
VRAM region when the first is still `BEING_READ` at the second claim; the second
write `invalidate()`s the first, leaving it `IDLE` + `is_latest=False` and **never
freed**. The scheduler registers such tids at the claim sites and releases the dead
(not-latest, IDLE) duplicates each tick — provably safe (no compute reads a non-latest
region; the only transfer source-select requires `is_latest`). **SD3 −295 MiB.**

### 5.2 `MemorySpace` tid-index (core performance)
`MemorySpace.get_by_tensor_id` was an O(all-regions) linear scan, called ~10×/node;
SD3 holds ~6 000 VRAM regions, so a full run took ~8 min. A `tid → list[region]`
secondary index (maintained in `claim`/`release`; `get_by_tensor_id` returns the
per-tid list sorted by page) makes it O(1)-ish and behaviour-identical (same region
order → deterministic placement). **SD3 ~8 min → ~1–2 min.** Core change, verified
byte-identical across accelerate / diffusers / a vanilla (DeviceAwareVanillaAsync)
config.

### 5.3 Producer-less `birth` fix (loader aliasing over-merge)
`_apply_storage_aliasing` clusters same-`storage_id` tensors by `[birth, death]`
overlap (concurrent views merge; sequential reincarnations stay separate). But
**producer-less** tensors — cuda activations whose producer was a cross-device
dispatcher / machinery node the reconstruction stripped — defaulted to `birth=0`,
so every reincarnation of a reused slot "started at 0" and the clustering collapsed
**thousands of 0.1 ms buffers into one union-lifetime mega-tid held for ~6 s** (one
24 MiB tid absorbed 1143 tensors / 1621 consumers). Fix: a producer-less tensor is
born at **first consumption**, not 0 (track producer-births and first-uses separately).
Clusters ≥50 members: **84 → 1.** **SD3 1460 → 719 MiB; SDXL 175 → 125 (solved).**
*(Shared loader code; accelerate stays byte-identical — its masters all have real
producers.)*

### 5.4 Dead-intermediate GC (diffusers-gated)
The consumer-triggered refcount free only fires when a consumer *retires*. It misses
(a) **orphans** (produced with no consumer ⇒ rem==0 from the start) and (b) tids whose
region is finalized after rem already hit 0. SD3/MMDiT generates thousands of these;
they accumulate. A periodic sweep (`dead_gc_period`, default 32 ticks) frees any
INTERMEDIATE region with `remaining_consumers==0` + IDLE (skip masters/initials) —
provably dead. **SD3 719 → 580 MiB.** **Gated to the diffusers variant**
(`DiffusersGroupOffload.__init__` sets the period; the accelerate base defaults it
**off**) because in accelerate the real allocator *holds* its few such buffers at the
peak — sweeping them made accelerate *less* accurate (768.55 → 763.3 vs the real
772.8). The phenomenon is diffusion-transformer-specific.

### 5.5 Master evict-fix (cpu-consumed masters)
`evict_after_node` (the per-epoch free schedule) counted only `gpu_runtime` consumers.
SD3's 432 MiB master (`[1,147456,1536]`, consumed by 20 **`cpu_leaf`** launchers, not
gpu kernels) therefore got **0 evict points** → held the whole run, sitting under
every other master's residency window. Fix: per master, use its gpu_runtime consumers
if any, **else its cpu-side consumers**, for the per-epoch last-use eviction. The 432
is now freed per-step. **SD3 580 → 531.95 MiB** (master-at-peak 499.5 → 432.8 = just
the legit 432). Volume unchanged (the 20 re-streams already fire); **accelerate
byte-identical** (its masters all have gpu_runtime uses, so the fallback never
triggers — evict points 3826/4366 exact).

---

## 6. Results (vs the true-hardware no-profiler run)

Targets: noprofile `inference_seconds` (±20 % e2e) and the **kineto traced-inference
peak** (±10 % VRAM). e2e is **exact-step** throughout the fidelity work (the §5 fixes
only change residency, not the critical path).

| Workload | e2e sim | noprofile | Δ e2e | VRAM sim | kineto peak | Δ VRAM |
|---|--:|--:|--:|--:|--:|--:|
| sdxl-turbo | 1.225 s | 1.151 s | +6.5 % ✅ | **125.2 MiB** | 129.2 | **−3.2 % ✅** |
| sd3 | 5.318 s | 5.389 s | −1.3 % ✅ | **531.95 MiB** | 464.4 | **+14.5 %** ⚠ |

SD3 VRAM arc: 1756 (naive) → 1460 (§5.1) → 719 (§5.3) → 580 (§5.4) → **531.95
(§5.5) = −70 %**, e2e exact. **SDXL is solved**; SD3 is just outside ±10 % — see §7.

## 7. Sources of the remaining SD3 VRAM gap (+14.5 %, ~68 MiB over 464)

Investigated in detail; the peak occurs **early** (sim-time ~0.92 s, ~9 % of nodes
done — the step-0/1 boundary), and it is **not** over-hold or a scheduling bug
(two earlier hypotheses — "late alias execution" and "refcount-stuck" — were
**disproven**: the consumer of the dominant 36 MiB activation *dispatches at sim-time
0.9209 s*, right at the peak; truly-stuck = 0 MiB). The residual decomposes as:

1. **~18.8 MiB — page-quantization artifact (clean lever).** 4824 cuda **scalar**
   params (`shape []`, 8 bytes, `param_*`), each rounded up to one 4 KB page → 18.84
   MiB; real data is 38.6 KB. They're by-value kernel args
   (`AUnaryFunctor<MulFunctor<float>>` / `CUDAFunctorOnSelf_add`), never streamed →
   real VRAM ≈ 0. The loader's scalar-arg drop (§4.1) covers **cpu** scalars but not
   these **cuda-home** ones. Extending it would reclaim ~18 MiB → SD3 ≈ 513 (≈ ±10 %).
2. **~12.7 MiB — genuinely cross-step intermediates** (44 tids whose consumer is in a
   *later* diffusion step). Possibly legitimate carried-over state; freeing them
   assumes reality re-produces fresh — **unverified**.
3. **~36 MiB — legitimate early-peak working-set concurrency.** At the early peak the
   sim's live set (432 master + ~77 MiB just-produced activations whose consumers are
   *imminent* + 18.8 MiB scalars) exceeds reality's 464. Whether the sim over-concurs
   vs reality at the equivalent instant is **unverifiable**: sim-time and recorded-time
   diverge by modelling (the sim streams the 432 at sim-0.92 s vs recorded 1.21 s), and
   the kineto trace has **no per-instant weight/activation split** (`Addr` is dropped,
   `storage_id` re-indexed). Tuning here would mean fitting an ungrounded target.

So **#1 is a clean ~18 MiB future lever** (reuse the existing scalar-arg drop for
cuda scalars); **#2/#3 are unverifiable** with the current trace and were deliberately
not tuned (no duct-tape). A future trace that retains device `Addr` (a tid↔allocation
bridge) would make #2/#3 verifiable.

## 8. Inherited from accelerate / deliberately NOT changed

Everything in `docs/accelerate_cpu_offload.md` §4 applies (reincarnation→master,
storage-based weight↔consumer linkage, H2D-as-TransferJob, imposed eviction,
probe-effect compensation, …). The diffusers path adds only the §4–§5 items above.
Op durations, the non-weight control/data graph, and the stock RTX-4090 hardware
model (bandwidth set to the measured pinned rate, §4.4) are unchanged.
