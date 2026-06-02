# Known Problems — Modeling PyTorch Offload Traces in cg-sim

Goal: faithfully simulate `accelerate` cpu_offload (and later `diffusers`
group_offload) so cg-sim's **e2e and VRAM peak match the real run**. The fix is
**co-designed across a trace loader and the offload scheduler as one set**.

This is a clean restart. The previous `accelerate_cpu_offload` /
`diffusers_group_offload` schedulers (both forks of `device_aware_vanilla_async`,
"DAV") were deleted — they reverse-engineered clean semantics *at runtime* and
that became unwinnable whack-a-mole (§7). New plan: a dedicated loader
(`sim/load/pytorch_loader2`, currently a verbatim copy of `pytorch_profile`) does
the messy reconstruction **once with full trace context**, and a small new
`AccelerateCPUOffload` scheduler only *executes* the clean plan.

All numbers below are measured from the trace this session (commands reproduced
by four analysis agents). **Every assumption must stay grounded in the trace** —
see `[[feedback-no-ducttape-ground-in-evidence]]`.

---

## 0. Targets (reference: llama-3-3B accelerate cpu_offload, 15-token decode)

| Metric | Value | Source |
|---|---|---|
| **e2e target** | **inference = 9.098 s** (true HW) | `noprofile/terminal_output.log` |
| profiled span (NOT the target) | 11.84 s (probe-inflated) | GPU node span; = single stream-7 wall |
| **VRAM peak target** | **810.4 MB** (manifest) ; 793.5 MB reconstructed (−2.1%) | `manifest.vram_peak_allocated_bytes` ; storage-granularity recon |
| model footprint (one copy) | 6.425 GB (bf16) | 255 weights × per-copy bytes |
| total H2D streamed | **108.20 GB** over **3826** Memcpy events | sum of device Memcpy-HtoD bytes |
| measured H2D bandwidth | **13.95 GB/s** aggregate (median 14.3 GB/s, pageable) | 108.20 GB / 7.759 s of Memcpy |
| logical streamed weights | **255** (253 × 15 loads, 1 × 30 [tied embed], 1 × 1) | grouped by cpu-source storage_id |

**e2e target is noprofile (P10), not the 11.84 s profiled span.** The old
scheduler "matched" 11.84 s only by double-counting H2D (§P3). Probe overhead is
compensated per-op via `probe_effect_table.csv` (`cpu_node_probe_effect_compensate`).

---

## 1. The bundle (cg-sim input) — `pytorch_runtime_v3`

`examples/trace/accelerate-cpu-offload__llama-3-3B__RTX4090/llama_bundle/`:
`manifest.json`, `runtime_nodes.csv` (128 700), `runtime_edges.csv` (325 398),
`pytorch_runtime_tensors.csv` (55 303). The loader reads these (CSV path).
Parse the node CSV with a real CSV reader — kernel names contain unescaped commas
(`awk -F,` mis-splits columns).

**Edge taxonomy** (`edge_kind`, src/dst id-prefix `t`=tensor `k`=node):

| edge_kind | count | shape | meaning |
|---|---|---|---|
| `thread_order` | 106 088 | k→k same thread | CPU program order |
| `data_input` | 93 372 | t→k | tensor feeds node |
| `data_output` | 75 763 | k→t | node produces tensor |
| `submit` | 22 608 | k→k CPU→CUDA | host launch → device kernel (matched by `correlation_id`; `submit_exact=true`) |
| `stream_order` | 22 607 | k→k same stream | GPU execution order |
| `wait` | 4 959 | k→k CUDA→CPU | device kernel → host `cudaStreamSynchronize` barrier |

The loader maps `data_input/output` → `node.input/output_tensors`, the four
control kinds → `parent/children_nodes`, **except** `submit`(submit-role →
gpu_runtime) edges, which it carves out to `trace.args["start_gated_edges"]`
(CUDA async-launch: the kernel may dispatch once the launch *starts*).

**Sync model (decisive):** all 22 608 GPU nodes are on **`stream_id=7`**; **zero
overlapping GPU intervals** → execution is strictly **serial / synchronous** (no
copy∥compute overlap — unlike diffusers' separate copy stream). Each Memcpy HtoD
ends a median **331 µs** before its consuming compute; loads are **interleaved
with compute**, not front-loaded. 3826 device `Memcpy HtoD (Pageable -> Device)`
kernels are 1:1 with 3826 host `cudaMemcpyAsync` zero-byte submit nodes.

---

## 2. The central difficulty (what makes this trace hard)

An offloaded weight is **not** one persistent `WEIGHT` tensor. Each forward
creates a fresh `CONTEXT`/cuda **reincarnation** loaded by a Memcpy; 54 417
CONTEXT tensors (538 GB of rows) back a 6.4 GB model. Three compounding facts:

1. **No single weight identity.** Reincarnations differ in `tensor_id` *and*
   destination `storage_id` (the dest is a small reused VRAM pool of ~39/1410
   slots). The 886 `WEIGHT`-kind tensors are **8-byte scalars (7 KB total)**, not
   matrices — there are **zero cold large weights**; the whole model is streamed.
2. **The same-tid TransferJob invariant (engine, §3).** A `TransferJob` requires
   `src_region.tensor_id == dest_region.tensor_id`. So you *cannot* move bytes
   from a cpu-master tensor into a different-tid reincarnation. The only way to
   model the H2D is to **collapse all reincarnations of a weight onto ONE master
   tid** and transfer master-RAM → master-VRAM (same tid, two regions). This is
   the root cause that forced (and repeatedly broke) the old runtime coalescing.
3. **Load→use is not a clean data edge.** A Memcpy's dest cuda tensor is consumed
   by a CPU `detach` (3825×), which spawns yet another reincarnation that the
   GEMM reads. The load→matmul order is carried by `stream_order` + the host
   stream-sync, **not** a data edge. Naively "redirect consumer to master" misses
   this detach hop — which is exactly why prior attempts deadlocked.

**Per-weight grouping key = the Memcpy's cpu-SOURCE `storage_id`** (verified
255 groups, no shape collisions, unifies the tied embed/lm_head). Rejected
alternatives: dest storage_id (only 39 — reused pool), cpu source *name* (256 —
splits the tied weight), `(module_path, shape)` (module_path blank on all Memcpy
nodes; shape collides up to 56-way).

**Exact loader recipe (becomes loader code):**
1. Memcpy nodes = `op_name == "Memcpy HtoD (Pageable -> Device)"` (3826).
2. Per Memcpy: its `data_input` edge from a **cpu** tensor = the CPU source; its
   `data_output` cuda tensor = the reincarnation; its `submit`-edge parent =
   the `cudaMemcpyAsync` launcher (3826/3826 pair).
3. **Group by cpu-source `storage_id`** → 255 masters. All cuda reincarnations
   (and their detach-chain descendants) of a group collapse to the master tid.
4. Master = one tensor, device `cpu`, an initial type (so it lays out in RAM),
   `size_bytes` = per-copy bytes (bf16). Redirect every consumer's
   `input_tensors`: reincarnation → master.

---

## 3. Engine mechanics that constrain the design (read before coding)

`sim/core/engine/engine.py`, `sim/core/job/*`, `sim/core/system.py`:

- **Lifecycle:** `sched.compile(trace)` (one-shot graph rewrite) → `sched.layout(init_storage)` (called until it returns `True`; may only emit *immediately-runnable* TransferJobs, else "Deadlock") → `sched.runtime(retired_jobs)` (per engine tick).
- **`job_waiting` is a FIFO deque with head-of-line blocking:** if `job_waiting[0]` isn't runnable **and nothing is running**, the engine ABORTS "Deadlock detected." So never queue a not-yet-runnable ComputeJob at the head with no in-flight job. `job_running` is an ETA-ordered heap.
- **Scheduler API (`System`):** `compute(hw,node)` and `transfer(batch)` enqueue jobs; **`claim(hw,tensor,page_idx)` and `release(region)` are IMMEDIATE/inline** (run assertion+mutation synchronously). `claim` needs an explicit *free* `page_idx_start` (compute first-fit yourself; `-1` fails).
- **ComputeJob runnable** (`compute_assertion`) iff: hw free & type matches; **if `node.custom_deps` non-empty → ONLY those are checked** (built-in control+data checks are bypassed); else all `parent_nodes` DONE + every input tensor has a `is_ready & is_latest & IDLE/BEING_READ` region in the compute hw's local memory + every output tensor has an `IDLE` region (must be pre-claimed).
- **TransferJob runnable** (`transfer_assertion`): src ready & not BEING_WRITTEN; dest IDLE; **`src.tensor_id == dest.tensor_id`**; `src_pages ≤ dest_pages`.
- **Coherence:** transfer end → dest `is_ready=True`, `is_latest = src.is_latest`. Compute writing an output `invalidate()`s every other region of that tid (`is_latest=False`) across all memories. A read-only weight master kept in RAM + copied to VRAM leaves **both** regions `is_latest` — the GEMM reads the VRAM copy fine, and the RAM master survives for the next reload.
- **`TensorAtHWDep(tid, custom_dep_tag)`** (`custom_dep.py`) blocks a node until `tid` has a ready+latest region on the hw tagged `custom_dep_tag` (config tags VRAM `"TARGET_VRAM"`). An engine-native, dependency-based residency gate — but custom_deps are all-or-nothing (see above), so using it means reconstructing the node's *other* deps too.
- **Timing:** ComputeJob work = `compute_time_micros` (probe-compensated), GPU `max_concurrent_jobs=1` (serial). TransferJob bandwidth = water-filled `memory_bandwidth_KBps` across concurrent transfers on each memory; one transfer gets full BW. **Serial load→compute→load chain emerges automatically** from the `stream_order` control edges (op_{i+1}'s parent is op_i) IF the scheduler issues a weight's transfer only when its consumer becomes ready.
- **VRAM peak = `MemorySpace.peak_num_used_pages`** (updated on every claim). Matching 810 MB requires the claim/release schedule to reproduce real concurrent residency (per-forward evict; peak ≈ the single 788 MB embedding).

---

## 4. The clean design (loader reconstructs, scheduler executes)

**LOADER (`pytorch_loader2`, behind an `offload_reconstruct` flag — keep
vanilla/lazy/eager untouched, §P12):** after the existing parse, run the §2
recipe: build 255 masters keyed by cpu-source storage_id; collapse
reincarnations (incl. detach chain) → master tid; redirect consumers; **neutralize
the recorded Memcpy + cudaMemcpyAsync `compute_time_micros` → 0** (§P3, avoid
double-count); emit a per-weight **load/evict schedule** (the ordered recorded
load events + their consuming nodes) into `trace.args` so volume = 108 GB by
construction and eviction is per-forward. Cold scalars (~7 KB) stay resident.

**SCHEDULER (`AccelerateCPUOffload`, slim, DAV-informed but not a fork):**
FAITHFUL TRANSLATION — the loader+scheduler mirror the trace AS-IS for accelerate
**and** diffusers: compute/CPU nodes → Nodes (durations kept, except those that
purely block on a transfer — see P3); each recorded **tensor transfer → an
explicit `TransferJob` at its recorded graph position**, NOT a residency/
stream-on-demand heuristic (which is accelerate-only and breaks diffusers
prefetch/D2H). Concretely: the loader tags each `Memcpy HtoD` node
`args["offload_transfer"]={master, dir:"h2d"}` (3826 triggers). `layout` places
masters in RAM (SSD→RAM). `runtime` treats a tagged node as a TRANSFER trigger —
when it's ready, claim the master's VRAM region + `sys.transfer` one same-tid
RAM→VRAM job, and mark the node DONE on that job's retire; normal nodes →
`sys.compute`. The GEMM (stream_order child of the trigger) then passes the
built-in input-residency check (master in VRAM) = load-before-use. Evict per
`evict_after_node` (accelerate has no recorded free event; diffusers will instead
carry `dir:"d2h"` triggers → VRAM→RAM transfer + release). **No on-demand staging
(§P7); no coalescing (loader owns identity).**

**CONFIG:** `trace.type: PytorchLoader2`; RAM/VRAM `memory_bandwidth_KBps`
**≈ 14 GB/s** (13.95, the measured pageable rate — replaces the old 8.32 fudge);
`scheduler.type: AccelerateCPUOffload`.

---

## 5. Problems (refined, with this session's verdicts)

### Loader-side
- **P1 — Reincarnation soup.** CONFIRMED. 54 417 CONTEXT/cuda reincarnations; emit one RAM master per logical weight + per-forward loads as transfer events.
- **P2 — Identity = Memcpy cpu-source `storage_id`.** CONFIRMED (255 groups). NOT dest storage_id (39, reused pool), NOT name (256, splits tied weight), NOT `(owner,shape)` (module_path blank on Memcpy nodes; shape 56-way collision).
- **P3 — H2D double-count.** CONFIRMED. Each load = device `Memcpy HtoD` kernel (bytes+duration here) **and** host `cudaMemcpyAsync` (zero-byte submit). Model H2D once: neutralize both nodes' compute, move bytes via the transfer. MEASURED: host `cudaMemcpyAsync` sum (7851 ms) ≈ device `Memcpy HtoD` sum (7758 ms), **ratio 1.012** — the host call BLOCKS for the whole H2D (synchronous copy), so its duration *is* the H2D time; neutralizing it is the faithful choice (else +7.85 s). `cudaStreamSynchronize` is already zeroed by `zero_wait_nodes`; genuine `cpu_leaf` dispatch (~1.18 s) is kept.
- **P8 — Cold weights.** REVISED → essentially nonexistent. Only 885 cpu 8-byte scalars + 1 cuda `[64]` tensor (~7 KB total) are never streamed. No large cold weight; don't expect weight matrices resident at layout.
- **P12 — Loader is COMMON.** Gate offload reconstruction behind a flag; re-verify vanilla/lazy/eager unchanged after every loader change.
- **P13 (NEW) — Detach-chain breaks load→use data edge.** A Memcpy's cuda dest is read by a CPU `detach`, not the GEMM. Collapse the whole detach chain onto the master so the GEMM directly depends on the master tid; otherwise ordering is lost and the sim deadlocks.

### Scheduler-side
- **P5 — Reproduce 108 GB.** CONFIRMED 108.20 GB / 3826 events / ~15× per weight. Volume must equal the recorded events — driven by per-forward evict + reload, not residency heuristics that under-stream.
- **P6 — Order by DEPENDENCY, not `start_ns`.** CONFIRMED serial. Load is ordered *before* its use as a real data dep (built-in input-residency gate); the serial chain comes from `stream_order` edges, never wall-clock matching.
- **P7 — No on-demand safety net.** Keep. Transfers are explicit (loader plan); a genuinely-missing input ABORTS. See `[[feedback-explicit-transfers-no-safetynet]]`.
- **P9 — VRAM peak needs the policy's free schedule.** CONFIRMED. No `cudaFree` ops (VMM unmap/release instead); impose per-forward eviction. Peak ≈ 793–810 MB, set by the single 788 MB embedding (~5 storages live at peak), NOT accumulating weights.

### Cross-cutting
- **P10 — e2e target = noprofile (9.098 s)**, consistently, for ALL configs.
- **P11 — Diffusers async vs accelerate synchronous.** Accelerate = single stream, serial (do this first). Diffusers = `use_stream` async, separate copy stream, concurrent H2D∥D2H, prefetch window — harder; after accelerate.
- **P14 (NEW) — Same-tid TransferJob invariant is the architectural pivot.** The whole "one master tid" design exists to satisfy `src.tensor_id == dest.tensor_id`. The loader MUST emit the master as the same tid the consumer reads.

---

## 6. Co-design order (one problem at a time)
1. **Loader: clean representation** (P1+P2+P3+P13+P14) behind the flag; re-verify vanilla/lazy/eager (P12).
2. **Config: bandwidth** = measured 14 GB/s (P4).
3. **Scheduler: execute explicit transfers** (P5+P6+P7) — load-before-use as real dep; abort on missing.
4. **Scheduler: per-forward eviction for peak** (P9).
5. **Validate vs noprofile across ALL configs** (P10, P12) via the MCP debugger.
6. **Diffusers async** (P11) — after accelerate is faithful.

## 7. Why the old approach was abandoned (do-not-repeat)
Runtime coalescing in the scheduler spawned an endless chain: `(owner,shape)`
group_key collided same-shaped weights → use-before-load; `_ensure_inputs_resident`
on-demand staging hid mis-modeling and under-streamed; redirecting consumers to
masters broke the detach-chain ordering → "weight not loaded for its matmul"
deadlock every forward; proactive load-before-use advanced to t≈69.7 s but timing
degraded (≫9.1 s) and still deadlocked. Each fix revealed the next. Lesson: do the
reconstruction **once in the loader** with full trace context; keep the scheduler
a thin executor. (DAV machinery worth reusing: multi-phase layout, first-fit page
claim, deferred-release retry, the `committed_per_compute` HOL-avoidance cap.)
