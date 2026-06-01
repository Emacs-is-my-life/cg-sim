# cg-sim Offload Verification — Handoff & Working Doc

Single source of truth for the offload-scheduler verification project. Merges
the former `TODO.md` (state/plan/decisions) and `docs/cg-sim_divergence_sources.md`
(D1–D8 catalog, correction hierarchy). If you are a fresh session, read this
top-to-bottom and you will be caught up.

**Reading order for catching up**:
1. `README.md` — what cg-sim is and how to run it
2. `CLAUDE.md` — durable context (faithfulness principle, bandwidth calibration,
   container quirks, commit hygiene)
3. **This file** (`docs/TODO.md`) — the project's state, the active direction
   (approach A), the divergence catalog, the decision log, and a map of where
   each simulator mechanism lives in the repo.

`docs/cg-sim_bandwidth_calibration.md` has the SDXL/SD3 bandwidth-compromise
rationale (relevant to the superseded approach B).

---

## TL;DR

We are verifying that cg-sim reproduces real offload runs (HuggingFace
`accelerate.cpu_offload` for Llama; diffusers `group_offload` for SDXL/SD3),
matching **e2e wall time (±20%)** and **peak VRAM (±10%)**.

**Current direction: approach A (trace-driven).** Feed the simulator the
*offload run's own trace* and have the loader recognize+mark its real transfer
events, letting the hardware model compute timing. This replaced approach B
(scheduler synthesizes offload behavior from the *eager* trace), which we
proved cannot match e2e time. See the decision below.

---

## DIRECTION DECISION (current) — approach A, trace-driven

We stopped making the scheduler **synthesize** offload behavior from the eager
trace (approach B). Two evidence-backed reasons discovered this session:

1. **The accelerate/diffusers documentation is not accurate.** `record_stream`
   semantics, matched-group eviction (docs imply pointer-swap; real traces show
   real D2H bytes — see D6), and the matched/unmatched stream split all differ
   from the docs. A policy synthesized from docs is simply wrong.
2. **e2e time is dominated by microscopic events** — per-leaf CPU hook overhead
   (~1 ms/leaf), `cudaStreamSynchronize`, `cuMem*` driver calls — far more than
   transfer bytes/bandwidth (the per-Memcpy transfer fits `bytes/bandwidth` at
   R²=0.99; the wall is CPU-bound). These are absent from the eager trace
   (D1, D2), so no synthesis on the eager trace can recover them.

**Conclusion (a real finding, not a retreat):** without the *same* trace — the
offload run's own trace, the most accurate description of the scheduler's real
behavior — the simulator cannot match the real run's e2e time. So we feed the
simulator that trace.

**What approach A is.** Feed the *offload* trace (not eager). Modify the loader
(`sim/load/pytorch_profile/`) to recognize the real transfer events and mark
them so the engine/DAV reproduces the run; let cg-sim's hardware model compute
timing. The policy becomes *ground truth read from the trace*, not a guess.

**Why A dissolves most of D1–D8** (see catalog below for the IDs):
- D1 hook CPU chain — present in the trace as real `cpu_leaf` nodes. Solved.
- D2 Python frames — named CPU ops present; residual slivers small (a
  `with_stack` bottom-edge reconstruction can close the rest later).
- D5 batching — each Memcpy is its own event → per-tensor transfers, ~1:1.
- D6 matched-group D2H — the D2H events are in the trace; no guessing.
- D8 tied weights — both transfers recorded separately; no tid-collapse guess.
- D3/D4 (single bandwidth knob vs concurrent H2D∥D2H streams) — the only
  residual; access pattern is exact, timing model still imperfect if recomputing.

**What A trades away.** It does NOT predict offload for a model you only have an
*eager* trace of — you must already have the offload trace. Accepted cost.

**What A means for the offload schedulers.** `AccelerateCpuOffload` and
`DiffusersGroupOffload` (synthesis logic) become largely unnecessary; the
transfers are explicit in the trace. The phantom-node compensation can be
removed once A lands (the real CPU overhead is in the trace).

### Concrete A plan (grounded in the current loader)

Building blocks already exist; this is extension, not from-scratch:
- The loader already handles HtoD-memcpy `gpu_runtime` nodes, device-crossing,
  storage aliasing, and "transfer-on-input-mismatch" firing
  (`pytorch_profile.py` ~L530, 830–880, 1054).
- There is already an `inject_schedule_path` hook (L1054+) that makes DAV
  replay a weight-streaming schedule via transfer-on-input-mismatch **without**
  a bespoke replay scheduler. Most promising integration point.

Two viable designs (pick after a full read of the loader's node/tensor
construction path — NOT yet verified at line level):

- **D-1. Explicit transfer nodes.** Loader re-types `Memcpy HtoD`/`DtoH`
  `gpu_runtime` nodes into transfer operations (src/dst tensors resolved from
  data edges + device fields — done manually this session in Test E/F),
  preserving submit/wait/data deps. Engine runs them as `TransferJob`s. Most
  faithful to the recorded schedule.
- **D-2. Residency-driven (reuse `inject_schedule_path`/transfer-on-mismatch).**
  Loader marks offloaded weights RAM-resident initially (signal: any tensor that
  is the *source* of an H2D Memcpy) + extracts the eviction schedule (when each
  weight leaves VRAM, from DtoH event timing). DAV's existing
  transfer-on-input-mismatch recreates the H2D; eviction hints drive the D2H.
  Less new code; slight risk of re-introducing policy-guessing for eviction.

Recommended: prototype **D-2** first (least new code, reuses
`inject_schedule_path`); fall back to D-1 if eviction fidelity is insufficient.

### First steps for the next session (approach A)

1. Read `sim/load/pytorch_profile/pytorch_profile.py` node/tensor construction
   end-to-end; confirm the line-level integration point for marking transfers.
   (Only spot-read so far; no grounded LoC estimate yet.)
2. Start with the simplest trace: `examples/trace/llama3b_offload_model/`
   (accelerate, single pageable path — no stream concurrency, so D3/D4 don't
   bite). Get A working there before diffusers.
3. Verify: trace-driven sim e2e + peak VRAM vs the same trace's recorded span.
   Target tighter than B (±10%?) since the policy is now ground truth.
4. Then diffusers (where D3/D4 concurrent-stream timing is the open question).

### Methodological bonus

Once A works, it validates B retroactively: A-sim vs real validates the
**timing model**; B-sim vs A-sim validates **policy synthesis**. Splitting the
two halves localizes residual error (today they're conflated — which is why
SDXL "passes by accident").

---

## D1–D8: divergence sources (catalog)

| ID | Source | Affects | Status |
|---|---|---|---|
| **D1** | Eager trace lacks the offload hook's CPU work — no `aten::to` / `set_module_tensor_to_device` / `cudaStreamSynchronize` / `cuMemMap/Create/Release/Unmap` per leaf. | Both | A: solved (in trace). B: phantom. |
| **D2** | Eager profiler captured no Python frames at loader level. ~700 µs/leaf interpreter time invisible. | Both | A: mostly solved. B: rolled into phantom. |
| **D3** | Single global memory bandwidth knob. Real systems have multiple paths: pinned HtoD (~25 GB/s), pageable HtoD (~13 GB/s), pageable DtoH (workload-dependent). | Diffusers (accelerate is all-pageable) | Open (timing-model limit). |
| **D4** | No CUDA stream model. Real diffusers runs matched H2D on a pinned side stream and pageable D2H on the compute stream — physically independent; cg-sim's water-filling shares one pool. | Diffusers | Open (timing-model limit). |
| **D5** | Batched TransferJob vs per-tensor `cudaMemcpyAsync`. Real diffusers fires one Memcpy per parameter (22,317 for SD3); cg-sim batched into one per group (651). 100–250× event-count collapse. | Both, worse diffusers | A: solved (per-event). |
| **D6** | `record_stream=False` matched-group eviction emits **real D2H bytes**, not pointer-swap as documented. Verified: SDXL UNet 3.36 GB / 4 steps, SD3 transformer 18.9 GB / 4 steps, all kind=WEIGHT. | Diffusers | A: solved (in trace). |
| **D7** | SD3 per-event D2H wall is ~7.5× larger than `bytes/bandwidth` predicts (≈2.83 GB/s effective D2H vs 21.4 GB/s H2D). SDXL is symmetric. | SD3 | A: moot if replaying durations; open if recomputing. |
| **D8** | Tied weights collapsed by cg-sim's tid-per-storage model. Llama-3.2-3B `embed_tokens.weight` ⇔ `lm_head.weight` share storage; accelerate transfers each separately; `_resolve_paged_leaves` attributed the single tid to only the first owner. | Accelerate (tied-weight models) | **Fixed** in scheduler. |

---

## Where to look in the repo — simulator mechanism map

Grounded paths (verified this session). Use this to navigate when implementing
approach A or debugging timing/VRAM.

### Engine & scheduling loop
- `sim/core/engine/engine.py` — discrete-event loop; `_compile()` / `_layout()`
  / `_runtime()` stages; `_runtime_forward()` pops jobs by ETA and does the
  **TransferJob two-phase retire** for fixed-latency (≈L247–258); `submit()`.
- `sim/core/engine/update.py` — `update_running_jobs`: per-tick ETA refresh;
  routes ComputeJobs (`max_work_rate`) and collects TransferJobs.
- `sim/core/engine/update_transfer.py` — **water-filling bandwidth allocation**
  across concurrent TransferJobs. The D3/D4 single-pool limit lives here.
- `sim/core/engine/job_stats.py` — stall-time / job-stats recording.

### Job model
- `sim/core/job/job.py` — `BaseJob`, `update_ETA`, work_total/work_done.
- `sim/core/job/compute_job.py` — `ComputeJob` (wraps a Node).
- `sim/core/job/transfer_job.py` — `TransferJob` (batch of `(src,dst)`
  DataRegions; `work_total` = pages×4KB; `fixed_latency_micros` two-phase
  retire). The rejected per-transfer `fixed_latency` idea would have lived here.
- `sim/core/job/claim_job.py`, `release_job.py` — region claim/free (alloc).
- `sim/core/job/mutation/*.py` — begin/end side effects per job type
  (`transfer_mutation.py` sets BEING_READ/BEING_WRITTEN, frees on end).
- `sim/core/job/assertion/*.py` — runnability predicates
  (`compute_assertion.py`: input residency + control deps; `transfer_assertion.py`).

### System API (how a scheduler drives the engine)
- `sim/core/system.py` — `compute(hw, node)`→ComputeJob; `transfer(batch)`→
  TransferJob; `claim(hw, tensor)` / `release(region)`; `find(hw, tid)`;
  `abort()`. **Approach A's transfer marking ultimately routes through
  `sys.transfer`.**

### Trace model
- `sim/core/trace/node.py` — `Node` (id, name, compute_time_micros, args,
  input/output_tensors, parent/children_nodes, custom_deps, pre/post hooks);
  `NodeStatus`; `NodeHW`; `TerminalNode`.
- `sim/core/trace/tensor.py` — `Tensor` (args: device, tensor_type
  WEIGHT/INPUT/LEAF/CONTEXT, size).
- `sim/core/trace/trace.py` — `Trace` (node_map, tensor_map, `args`
  side-channel that carries scheduler hints like `xfer_arrivals`,
  `evict_after_node`, `start_gated_edges`, `module_hierarchy`).
- `sim/core/trace/custom_dep.py` — `CustomDep` predicates: `NodeDoneDep`,
  `TensorAtHWDep`, `MinTimestampDep`, `LambdaDep` (the bypass-residency hook).
- `sim/core/trace/loader.py` — base loader interface.

### Hardware models
- `sim/hw/common/base_hardware.py` — `BaseHardware` (job_running, run/retire,
  `max_work_rate`).
- `sim/hw/common/data_region.py` — `DataRegion` + `DataRegionAccess`
  (IDLE / BEING_READ / BEING_WRITTEN), `is_ready` / `is_latest`.
- `sim/hw/memory/common/base_memory.py` — `BaseMemory` + `MemorySpace`
  (page allocation, `get_by_tensor_id`).
- `sim/hw/memory/simple_ram/simple_ram.py`, `simple_vram/simple_vram.py` —
  `SimpleRAM` / `SimpleVRAM`: `memory_size_KB`, `memory_bandwidth_KBps`
  (the **single bandwidth knob** — D3 lives here; `can_run` caps at 4 jobs).
- `sim/hw/compute/simple_cpu/`, `simple_gpu/` — `SimpleCPU` / `SimpleGPU`
  (`modifier`, `max_concurrent_jobs`); `compute/common/base_*`.
- `sim/hw/storage/simple_ssd/simple_ssd.py` — `SimpleSSD`: `fixed_latency_micros`
  + read/write IO curves. The **only** hw with fixed latency today; the model
  for disk offload if `offload_to_disk_path` is ever used.
- `sim/hw/storage/common/base_storage.py` — `BaseStorage`.

### Loader (PyTorch profile → cg-sim trace)
- `sim/load/pytorch_profile/pytorch_profile.py` — the trace loader. Key methods:
  - `_read_nodes` / `_node_from_row` — CSV row → `Node`; sets `runtime_role`
    (cpu_leaf / submit / wait / gpu_runtime), `device_type`, `module`.
  - `_apply_storage_aliasing` — collapse tensors sharing storage (D8-adjacent).
  - `_mark_implicit_inputs` — WEIGHT/INPUT/LEAF classification.
  - `_add_temporal_data_control_edges` — temporal ordering edges.
  - `_annotate_alias_dispatcher_deps` — alias / dispatcher custom_deps.
  - `_is_start_gated_edge` — cudaLaunchKernel→kernel async (`start_gated_edges`).
  - `inject_schedule_path` (≈L1054+) — replay a weight-streaming schedule via
    transfer-on-mismatch. **Approach A's transfer-marking goes here / nearby.**

### Schedulers
- `sim/sched/common/base_scheduler.py` — `BaseScheduler` (compile/layout/runtime).
- `sim/sched/device_aware_vanilla_async/device_aware_vanilla_async.py` — **DAV**,
  the base for the offload schedulers. Hint channels (`xfer_arrivals`,
  `d2h_xfer_arrivals`, `evict_after_node`); multi-phase layout (SSD→RAM→VRAM);
  runtime transfer-on-input-mismatch; `_post_xfer_cpu_us` calibration table;
  phantom-node emission helpers.
- `sim/sched/accelerate_cpu_offload/` — `AccelerateCpuOffload` (compile-time hint
  synthesis + phantom; D8 fix in `_resolve_paged_leaves`).
- `sim/sched/diffusers_group_offload/` — `DiffusersGroupOffload`
  (matched/unmatched classification + phantom).
- `sim/sched/llamacpp_vanilla/`, `llamacpp_flexinfer/`, `generic_stub/` —
  other schedulers (reference patterns).

### Entry points & tooling
- `main.py` — `python3 main.py -i <config.yaml>` runs one simulation.
- `main_agent.py` — MCP server for interactive breakpoint debugging (see
  `AGENTS.md` for the breakpoint / `debug.break_lambda` / abort-interception
  surface).
- `scripts/analysis/extract_sim_metrics.py` — e2e + peak VRAM from `result.json`.
- `scripts/analysis/{cpu_offload_timeline,link_utilization,prefetch_quality,sim_summary}.py`
  — analysis helpers.

---

## Decision log (so the next session doesn't re-litigate)

Decided & committed:
- **Approach A over B** (this session's central pivot — see top).
- **Faithfulness principle** (`CLAUDE.md`): scheduler/loader should faithfully
  model the real run; phantom nodes are acceptable only when they represent real
  CPU work the trace doesn't capture; arbitrary tuning knobs are not.
- **No `sim/core/` changes** during the deadline phase. Per-stream bandwidth
  (the proper D3/D4 fix) is deferred — ~200–300 LoC in engine/hardware, too
  risky for this milestone. (Approach A sidesteps needing it for most sources.)
- **Per-transfer phantom node mechanism** (approach B compensation): scheduler
  injects CPU `Node`s into `trace.node_map` during `compile()`, one per
  TransferJob, gating the next prefetch. Calibration: pageable 1090 µs (Llama
  3B/8B agreement Δ<1%), pinned 70 µs (SDXL). To be **removed** once A lands.
- **Tied-weight fix** (`AccelerateCpuOffload._resolve_paged_leaves`): removed the
  `break`; accumulate all leaf modules consuming a WEIGHT tid. Closes D8.

Tried and rolled back:
- **`fixed_latency_micros` on memory hardware** — rejected. Per-Memcpy regression
  intercept ≈0; the overhead is *between* transfers, not transfer setup.
- **Single global per-TransferJob `cpu_overhead_us` knob** — rejected in favor of
  the two-value (pageable/pinned) table for DMA-path semantics.
- **Emit matched-block D2H in `DiffusersGroupOffload.compile()`** — tested, broke
  SDXL (+44%, out of bound). Reverted: without per-stream bandwidth, the new D2H
  competes with concurrent H2D in cg-sim's water-filling but doesn't in reality
  (D3/D4). This is exactly why approach A is the chosen path.

---

## Verification state (approach-B era, for reference)

As of commit `c6071fd` / `7ed0fa3` on `sim-test`, with B compensations applied:

| Workload | Sim e2e | Real e2e | Δ e2e | Δ VRAM | Bound (±20% e2e, ±10% VRAM) |
|---|---|---|---|---|---|
| Llama 3B accelerate | 11.87 s | 11.84 s | +0.3% | −8.6% | ✅ pass |
| Llama 8B accelerate | 21.85 s | 21.85 s | +0.02% | −8.4% | ✅ pass |
| SDXL diffusers (block_level + use_stream) | 2.11 s | 2.60 s | −19.0% | +1.8% | ⚠️ pass by accident (D3⇄D6 cancellation) |
| SD3 diffusers (block_level + use_stream) | 11.51 s | 25.33 s | −54.5% | −6.0% | ❌ fail |

Reference targets come from each offload trace's `runtime_nodes.csv`
(`max(end_ns) − min(start_ns)`) and `manifest.json:vram_peak_allocated_bytes`.

B compensations (all in `sim/sched/`, no `sim/core/`): per-leaf phantom CPU
nodes; tied-weight fix; PCIe bandwidth compromise (13 GB/s flat for SDXL/SD3,
see `docs/cg-sim_bandwidth_calibration.md`). SDXL "passes" only because the slow
flat bandwidth inflates matched-H2D wall by ~the amount of the missing matched
D2H bytes; SD3's byte ratios break that cancellation.

---

## Open questions

For approach A:
- Exact loader integration point for marking transfers (needs full read).
- D-1 vs D-2 design choice (explicit transfer nodes vs residency-driven).
- Do we **replay** recorded transfer durations (exact, but no hardware-variation
  prediction) or **recompute** from bytes/bandwidth (enables hardware studies but
  reintroduces D3/D4/D7 for concurrent diffusers transfers)? Decide per use case.
- Tensor-identity across Memcpy-created `cache_*` tensors — automate the
  storage-aliasing resolution done manually in Test E/F.

Carried over (mostly moot under A, relevant if B is revisited):
- Does `leaf_level + record_stream=False` pointer-swap or emit D2H bytes?
- Is D7 (SD3 D2H asymmetry) block_level-specific?

---

## Superseded — approach B reference (profile-setup recipe)

Kept because the flag analysis is grounded and correct *for B*. If A stalls and
B is revisited, this is the right re-profile recipe. The 4-step correction
hierarchy below also informed B.

### 4-step correction hierarchy
Prefer the highest (cheapest) step that closes a divergence source:
1. **Profile setup** — real-run flags producing a trace cg-sim can model
   (`with_stack`, `record_stream`, `leaf_level` vs `block_level`,
   `offload_buffers`). No sim code change. (Approach A is the logical endpoint
   of this idea: change the input trace entirely.)
2. **Simulator core** (`sim/core/`) — per-direction bandwidth, multi-stream
   lanes, per-tensor TransferJob. Most expensive/foundational.
3. **Trace loader** (`sim/load/pytorch_profile/`) — faithful PyTorch→cg-sim
   translation (tied storage, aliasing, Python-frame mapping). **Approach A is
   primarily Step-3 work.**
4. **Scheduler** (`sim/sched/`) — faithful runtime model. Most downstream; risks
   per-workload tuning.

### B re-profile recipe (diffusers)
Grounded against diffusers `apply_group_offloading` source. Cleanest B config:
```
--offload-mode group --group-offload-type leaf_level
--group-offload-use-stream --group-offload-non-blocking
# record_stream default OFF; low_cpu_mem_usage default OFF; no disk path
--profile-memory --record-shapes --with-stack --with-modules
```
Per-parameter rationale (evidence-based):
- `leaf_level` — removes the matched/unmatched split (the SDXL↔SD3 divergence
  cause); groups size 1 → per-tensor transfers (D5 ~1×). NOTE: does **not**
  eliminate all concurrent paths — onload (side stream) and offload (default
  stream, hardcoded `non_blocking=False`) can still overlap; confirmed in source.
- `use_stream=True` — compute/transfer overlap; cg-sim models it natively.
- `non_blocking=True` — onload honors it (pinned source) → async H2D, clean
  CPU-cost vs transfer-wall separation. Offload is hardcoded `non_blocking=False`
  regardless, so D2H still blocks the host.
- `record_stream=False` — deterministic, trace-visible eviction (explicit sync
  node) + predictable peak VRAM; `True` is allocator-deferred (needs a
  stream/allocator model cg-sim lacks).
- `low_cpu_mem_usage=False` — pre-pin once at init; `True` pins per-transfer
  (injects extra CPU work into the hot path).

### B re-profile recipe (accelerate / Llama)
Already <1% sim-vs-real; re-profile not required. If doing it, the profiler
flags are already defaults (`--with-stack --with-modules --profile-memory
--record-shapes`). `offload_buffers` is auto-computed per module (no CLI flag).

### Important script facts (grounded)
- **Group offload is NOT in the public `royce8636/pytorch-profile` repo** — the
  string `group_offload` appears nowhere in its history. The SDXL/SD3 traces
  were made with a **local/uncommitted** script. The public repo only has
  `none/model/sequential/module/module-hook` accelerate modes.
- `--offload-mode sequential` (public repo) ≈ leaf-level accelerate offload; the
  diffusers docs call leaf_level "equivalent to CPU offloading." Usable fallback.
- Profiler flags `--with-stack`, `--profile-memory`, `--record-shapes` default
  True; `with_modules=True` is hardcoded.

### Filling the ~700 µs/leaf CPU gap with `with_stack` (D2, for either approach)
The current bundle keeps only **bottom-most** ops (`cpu_leaf`); the gap is the
**self-time of parent frames** (hook Python code) the bundle drops. To fill it:
1. Build the per-thread CPU event tree from the flamegraph.
2. Sweep the **bottom edge** (deepest active frame at each instant) → a flat,
   gapless sequence.
3. **Classify** each segment: aten/launch → keep; Python self-time → CPU compute
   node (fills the gap); `cudaStreamSynchronize` / blocking `.to()` self-time →
   transfer-**wait** (gate, NOT CPU work — else double-count).
4. Coalesce consecutive Python slivers per leaf-forward into one measured
   `cpu_overhead` node (≈1–2 nodes/leaf), or emit full bottom-edge for max
   fidelity. This replaces the fitted phantom with a **measured** per-leaf value
   and is the structurally faithful D2 fix (loader-level, Step 3).
Requires piping the stack column through the bundle export
(`runtime_nodes.csv` currently has no stack/frame field).

---

## Verification commands

```bash
# Approach-B configs (current repo state)
python3 main.py -i examples/run/pytorch-eager__llama-3-3B__accelerate_cpu_offload.yaml
python3 main.py -i examples/run/pytorch-eager__llama-3-8B__accelerate_cpu_offload.yaml
python3 main.py -i examples/run/pytorch-eager__sdxl-turbo__diffusers_group_offload.yaml
python3 main.py -i examples/run/pytorch-eager__sd3__diffusers_group_offload.yaml

python3 scripts/analysis/extract_sim_metrics.py output/<config>/sim_results/result.json
```

Offload reference traces (the approach-A inputs):
```
examples/trace/llama3b_offload_model/                          # accelerate Llama 3B (simplest A target)
examples/trace/llama8b_offload_model/                          # accelerate Llama 8B
examples/trace/diffusers-group-offload__sdxl-turbo__RTX4090/   # diffusers SDXL block_level
examples/trace/sd3_offload_group/                              # diffusers SD3 block_level (LFS)
```

---

## Commit / push reminders

- **Never commit anything under `examples/trace/`** (LFS-managed; false-positive
  "modifications" appear after `git lfs pull` — see CLAUDE.md commit-hygiene).
- Stage explicit code/doc paths; never `git add .` / `git add -A`.
- Container has no GitHub credentials by default. PAT-in-chat is the current
  workaround (**revoke immediately after use**); session-level GitHub binding is
  the clean fix. Commit signing: `git -c commit.gpgsign=false commit …` when the
  env signer is unavailable and the user authorized it.
