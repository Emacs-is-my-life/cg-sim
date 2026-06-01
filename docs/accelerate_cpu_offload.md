# Accelerate CPU-Offload in cg-sim

This document records the **Trace modifications** the `AccelerateCPUOffload`
scheduler makes (in its `compile()` pass) to replay a HuggingFace
`accelerate` cpu_offload trace, the **reasoning** behind each, and **why each
is a faithful simulation** of the real policy.

Scheduler: `sim/sched/accelerate_cpu_offload/accelerate_cpu_offload.py`.
Configs: `examples/run/accelerate-cpu-offload__llama-3-{3B,8B}__accelerate.yaml`.
Results: `tmp/report/accelerate-cpu-offload.md` (peak VRAM matched to ±0.2 %).

---

## 1. Background — what the real policy does

`accelerate.cpu_offload(model, execution_device)` keeps **one** copy of the
model's state dict on the CPU. Per submodule, an `AlignDevicesHook` copies that
submodule's weights **CPU→GPU just before its forward** (`pre_forward`) and
**frees the GPU copy right after** (`post_forward`, `offload=True`). Because
inference weights are read-only, the "offload" is a *free* of the GPU
allocation, not a device-to-device copy back. The copies run synchronously on
the default stream, so there is **no prefetch overlap** — which is why the
policy trades ~8–15× less VRAM for a ~66–69× slowdown.

The goal of cg-sim here is to reproduce that **VRAM working set** (one
submodule's weights resident at a time) and the **per-forward streaming cost**.

## 2. Why the raw trace cannot be replayed as-is

The bundle is a **profiled** run, and the profiler/loader record the offload
through the lens of the CUDA caching allocator, which produces a trace that is
*not directly simulatable*:

- **There are no clean `WEIGHT` tensors.** Every forward streams each weight to
  a fresh GPU buffer, so the loader records each use as a separate
  `tensor_kind == "CONTEXT"` *reincarnation*. The `WEIGHT`-typed tensors that do
  exist are 8–256-byte scalars (norm γ/β, etc.); the real matmul/embedding
  weights appear as `cache_*`-named `INPUT`/`INTERMEDIATE` CONTEXT tensors.
- **The reincarnations are enormous in aggregate.** For llama-3-3B the loader
  emits ~196 CPU "master" copies (the model resident in RAM once, the real
  state-dict copy) **plus tens of thousands of per-forward cuda reincarnations
  totalling ~151 GB** — they cycle through only ~12 physical GPU storage slots
  in reality, but the loader keeps each reincarnation as a distinct cgsim
  tensor (so peak-memory accounting can reflect slot reuse).
- A vanilla layout places every `INPUT` tensor on its home device, so it tries
  to put 151 GB of cuda-homed reincarnations into 24 GB of VRAM and aborts with
  `LAYOUT_FAILURE` (verified via MCP — the engine fills all 6,156,080 VRAM
  pages and aborts on the next weight).

The real run never holds 151 GB anywhere: it holds **one logical copy per
weight in RAM** (~6 GB) and **one submodule's weights in VRAM** at a time
(~0.77 GB peak). The 151 GB is purely an allocator/profiling artifact of
representing N per-forward reuses as N distinct tensors.

The project already solves this offline in the `graph_modifiers` tool
(`storage_coalesce` + the `hf_accelerate` schedule solver + the injector that
writes marks into the trace). `AccelerateCPUOffload.compile()` performs the
**equivalent transform in-scheduler**, so the scheduler is self-contained and
needs no offline preprocessing. Every modification below mirrors something
`graph_modifiers` does.

---

## 3. Trace modifications made in `compile()`

All four mutations happen once, in `compile()`, before the layout stage.

| # | Mutation | Code |
|---|----------|------|
| A | Coalesce per-forward weight reincarnations onto one logical RAM master | `master_map`, L446–465 |
| B | Retarget a promoted master's `device` cuda→cpu | `t.args["device"] = "cpu"` (L459) |
| C | Redirect every consumer's `input_tensors` to the master | `n.input_tensors = new_in` (L482) |
| D | Retype orphaned reincarnations `INPUT`→`INTERMEDIATE` | `tm[tid].args["tensor_type"] = "INTERMEDIATE"` (L486) |
| E | Write `evict_after_node` + `evictable_tensor_ids` marks into `trace.args` | L498–499 |

A streamable weight is identified as a `CONTEXT` tensor whose shape is
"parameter shaped" — `min(shape) >= offload_min_weight_dim` (default 64). Each
weight's *owner* module is resolved by hopping from the consuming gpu node to
its CPU dispatcher (via `trace.args["start_gated_edges"]`) and walking up the
parent chain to the nearest `module_path`. Reincarnations are grouped by
`(owner, shape)` (shape-only fallback for shared/tied/unresolved weights), and
one CPU master is chosen per group.

### A. Coalesce weight reincarnations onto one logical master

**What.** All per-forward cuda reincarnations of a logical weight are mapped to
a single representative tensor (`master_map[reincarnation] = master`). The
master is the existing CPU `CONTEXT` tensor for that `(owner, shape)`; if a
group has only cuda reincarnations, one is *promoted* to master.

**Why.** The reincarnations are an allocator artifact, not distinct data. The
real run has exactly one logical copy of each weight.

**Why faithful.** Inference weights are **constant** (read-only). Every
reincarnation of weight *W* holds byte-identical contents — *W*'s value. So
collapsing them to one logical tensor changes no value any consumer observes.
The coalesced master count (~196–264) and RAM footprint (~6–8 GB) match the
real "one state-dict copy on CPU" that `accelerate.cpu_offload` keeps, and the
real `dram_peak_rss` (~12.6 GB incl. framework). This is exactly
`graph_modifiers`' `storage_coalesce` step.

### B. Retarget a promoted master's device to CPU

**What.** When a weight appears only as cuda reincarnations (no CPU master in
the trace), the chosen master's `device` is flipped `cuda → cpu`.

**Why.** Its home must be RAM so the layout places it in RAM (not VRAM) and the
runtime streams it on demand.

**Why faithful.** Under cpu_offload the weight's **home is the CPU** — it lives
in the CPU state dict and only transiently visits the GPU. The loader tagged
the reincarnation `cuda` because that is where the profiler observed the copy;
the *logical* residence is CPU. This is identical to the injector's
`streamed_tids` cuda→cpu retarget in `graph_modifiers/inject_schedule`.

### C. Redirect consumers to the master

**What.** Every node's `input_tensors` list has reincarnation ids replaced by
their master id (de-duplicated).

**Why.** So that all weight reads reference the single RAM-homed master, which
the runtime then streams + evicts.

**Why faithful.** Only the **data reference** is rewritten; the control graph
(`parent_nodes`/`children_nodes`) and therefore the execution order are
untouched. Because every reincarnation equals the master in value (point A), a
consumer reading the master reads exactly what it read before. Output tensors
and all non-weight tensors are left unchanged.

### D. Retype orphaned reincarnations INPUT→INTERMEDIATE

**What.** After redirection a reincarnation has no consumers; its
`tensor_type` is changed `INPUT → INTERMEDIATE`.

**Why.** The layout phase initial-places `WEIGHT`/`INPUT`/`LEAF` tensors. An
unconsumed `INPUT` would still be placed (and re-introduce the overflow); an
unconsumed `INTERMEDIATE` with no producer is simply never claimed.

**Why faithful.** These reincarnations were never *simultaneously* resident in
the real run — the allocator reused ~12 GPU slots for thousands of them.
Removing them from initial placement reflects that they do not occupy distinct
memory; their single logical footprint is captured by the master (RAM) plus the
runtime VRAM copy (point E). Neutralising rather than deleting keeps tensor ids
stable for any incidental references.

### E. Emit `evict_after_node` + `evictable_tensor_ids`

**What.** `trace.args["evictable_tensor_ids"]` is set to the master ids, and
`trace.args["evict_after_node"][gpu_node] = [masters it consumes]`. These are
the same marks `graph_modifiers` injects; the inherited DAV runtime reads
`evict_after_node` and frees the listed tensors' **VRAM** regions (keeping the
RAM master) the instant that gpu node retires.

**Why.** To model `AlignDevicesHook.post_forward` freeing the GPU copy after
each submodule's forward.

**Why faithful.** This is a direct analogue of `post_forward` with
`offload=True`: the GPU allocation is released immediately after use; the CPU
master persists (accelerate keeps the one state-dict copy). Combined with
on-demand staging, only the weight(s) of the currently-executing submodule are
VRAM-resident, so the simulated peak equals the largest single weight — which
is why the simulated peak (772.2 / 1022.2 MB) matches the real run (772.8 /
1024.0 MB) to within 0.2 %. VRAM-only eviction (not RAM) mirrors that the
read-only weight needs no write-back.

### Not added: `xfer_arrivals` (synchronous, on-demand staging)

`compile()` deliberately emits **no** `xfer_arrivals` (async prefetch marks).
Real cpu_offload copies in `pre_forward` on the default stream and **blocks** —
there is no overlap of the next submodule's H2D with the current compute. The
inherited `_ensure_inputs_resident` path already models exactly this: when a gpu
node needs a RAM-homed weight it issues a RAM→VRAM `TransferJob` and the
consumer waits for it to complete. The H2D bytes are charged through the normal
`SimpleRAM`/`SimpleVRAM` bandwidth model — necessary because the streamed
weights are `INPUT` tensors with no producer node, so the H2D cost is not
otherwise represented in the trace. Adding async prefetch would *understate*
the real (synchronous) cost.

---

## 4. What is deliberately NOT modified

- **Node compute durations** (`compute_time_micros`). The replay uses the
  trace's recorded durations as-is (see §5 for the separate, loader-level probe
  compensation).
- **The control/data dependency graph** — `parent_nodes`, `children_nodes`,
  `output_tensors`, and all non-weight (activation / KV-cache) tensors. Only
  weight `input_tensors` references, four `args` fields, and two
  `trace.args` side-channel keys change.
- **The hardware model / bandwidths** — stock RTX-4090 config.

This keeps the modification surface minimal and auditable: the simulated
*workload* (op graph, op costs) is the trace's; only the *weight residency
representation* is rewritten from the allocator's per-forward view to the
logical one-copy-in-RAM view.

---

## 5. Loader-side: probe-effect compensation (separate from the Trace mutations)

The trace is a **profiled** run, so its `cpu_leaf` durations are inflated by
kineto observer overhead. This is compensated at **load time** (not in
`compile()`) when `cpu_node_probe_effect_compensate: true` is set in the loader
config — the loader subtracts per-op `probe_effect_ns` from each matching
`cpu_leaf` duration, read from a sibling `probe_effect_table.csv`. Because the
probe effect is per-op and machine-specific, the table copied from the
same-machine `pytorch-eager` traces applies (see
`docs/eager-lazy-probing-effect.md`). This is the standard PyTorch-trace
calibration and is orthogonal to the offload modeling above.

---

## 6. Why the overall result is faithful

- **Peak VRAM is reproduced to ±0.2 %** (772.2 vs 772.8 MB; 1022.2 vs 1024.0
  MB). This is the decisive evidence that the residency rewrite (A–E) is
  correct: the simulated working set is one submodule's weights, exactly as the
  real hook leaves it.
- **The e2e tracks the trace it replays.** Measured against the trace's own
  profiled wall-span, the replay is within −4 % (3B) / +8 % (8B), i.e. the
  scheduler faithfully executes the recorded op/transfer schedule.
- **The offload trade-off direction and magnitude match**: vs the resident
  (`pytorch-lazy`) runs, both real and sim show ~8–15× VRAM reduction at a large
  slowdown.

The residual +20–25 % of simulated e2e over the *no-profiler* run is the kineto
probe effect baked into the trace (profiled is +12–30 % over noprofile), only
partially removable by the 6-op aten compensation table (it covers none of the
offload-specific `aten::copy_` / `cuda*` ops). This is a **data/calibration**
limitation, independent of the Trace modifications, and is discussed in
`tmp/report/accelerate-cpu-offload.md` §"Open item".

---

## 7. Heuristics and limitations

- **Weight identification is shape-based** (`CONTEXT` ∧ `min(shape) ≥ 64`).
  Transformer weights are parameter-shaped while activation/KV buffers carry a
  small sequence/batch dimension (e.g. `[9, 8192]`), so this separates them
  cleanly. A mutated buffer wrongly classified as a read-only master would be
  invalidated on write and then deadlock once its VRAM copy is evicted (this was
  observed for one `[9, 8192]` buffer during bring-up and fixed by the shape
  test). A weight that is genuinely square-and-large *and* mutated could fool
  it; none exist in these Llama traces. Tunable via `offload_min_weight_dim`.
- **Owner resolution** depends on `start_gated_edges` + `module_path` on the CPU
  dispatcher chain; weights whose owner cannot be resolved fall back to
  shape-only grouping, which can over-merge same-shape weights into one master.
  This slightly inflates the RAM master footprint (e.g. ~8 GB vs the ~6 GB true
  model size for 3B) but does not affect the VRAM peak (one weight resident at a
  time regardless) or e2e.
- **Per-use eviction** frees a weight's VRAM after every gpu consumer; a weight
  consumed by two consecutive gpu ops in one forward is re-streamed between
  them. This matches the recorded per-reincarnation structure and the total H2D
  volume; it is not a coarser per-submodule grouping.
