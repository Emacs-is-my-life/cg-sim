# TODO — Diffusers `group_offload` VRAM fidelity (continuation)

**Owner context:** Diffusers `group_offload` (leaf_level, non_blocking=True, use_stream=True,
record_stream=False, low_cpu_mem_usage=False) is IMPLEMENTED and **e2e-validated**. The one
OPEN problem is **VRAM peak fidelity**. This doc is self-contained so a fresh session can resume
without prior context. Companion refs: `docs/known_problems.md` (accelerate design + engine
mechanics), `docs/accelerate_cpu_offload.md`, memory `project-diffusers-group-offload`,
playbook `reference-offload-trace-quirks`.

Principles (do not violate): ground every constant in the trace; no duct-tape / no fudging to
hit a target; model recorded events explicitly, let mis-modeling ABORT rather than hide; announce
tie-break/threshold decisions in user-facing text. Test via the **cg-sim-mcp** debugger, NOT
`python main.py`. `examples/trace/` is read-only LFS — never edit/regenerate.

---

## 0. STATUS — what is DONE (do not redo)

- **Loader renamed** `PytorchLoader2` → **`PytorchOffloadLoader`** (`sim/load/pytorch_offload_loader/`).
  ONE loader for both offload modes; `offload_variant` arg auto-detects from the
  `cudaMemcpyAsync`/`Memcpy HtoD` duration ratio (≥0.5 → "accelerate", zero cudaMemcpyAsync;
  <0.5 → "diffusers", keep it). Memcpy matcher = prefix `"Memcpy HtoD"` (Pinned for diffusers,
  Pageable for accelerate). **Accelerate re-verified byte-identical** after the rename + the two
  shared-code additions below (3B e2e 8.6493 s / VRAM 777.5 MiB; 3B+8B loader invariants exact).
- **`DiffusersGroupOffload` scheduler** (`sim/sched/diffusers_group_offload/`): thin subclass of
  `AccelerateCPUOffload` that only **skips pre-claiming orphan dispatcher outputs**. The H2D∥compute
  overlap and free-on-last-use eviction fall out of the faithfully-translated trace (side-stream
  `stream_order` + the loader's evict schedule).
- **Two diffusers quirks fixed in SHARED code** (verified no-ops for accelerate):
  - *Scalar kernel-args:* tiny (≤ `_SCALAR_ARG_MAX_BYTES=4096`) non-master cpu tensors read by gpu
    kernels are by-value args (e.g. `AUnaryFunctor<MulFunctor<float>>` captures a scalar; diffusion
    sigmas/scales from `aten::sub`/`aten::empty`). Loader `_reconstruct_offload` step 5c drops them
    from gpu `input_tensors` (+ a control edge to the producer); a larger one is left in place and
    WARNs (possible missing transfer).
  - *HOL residency gate:* `AccelerateCPUOffload._submit_ready` now defers a gpu compute whose cuda
    inputs aren't VRAM-resident yet (`_inputs_resident_vram`), skipping `custom_deps` nodes. Without
    it, a side-stream-prefetched compute becomes ready before its master's cross-stream transfer
    lands → parked at the FIFO head → engine "Deadlock detected". No-op for accelerate (its gpu
    nodes are stream_order children of their own Memcpy, so resident once ready).
- **Configs:** `examples/run/diffusers-group-offload__{sdxl-turbo,sd3}__diffusers.yaml`
  (`type: PytorchOffloadLoader`, `offload_reconstruct: true`, `offload_variant: auto`,
  bandwidth 26.37 / 26.57 GB/s pinned — MEASURED, see §6).
- **Behavior facts established:** diffusers offload is a **pointer-swap, NO D2H** (≤6 `Memcpy DtoH`
  events total). H2D on side streams (13/17/21/25), compute on stream 7. So the overlap is free and
  the feared "concurrent H2D∥D2H bandwidth" is a non-issue.

### e2e RESULTS — PASS (this is the primary fidelity metric; leave as-is)
| trace | sim e2e | target (noprofile inference) | Δ |
|---|---|---|---|
| sdxl-turbo | 1.225 s | 1.151 s | **+6.5%** |
| sd3 | 5.318 s | 5.389 s | **−1.3%** (prior approach FAILED at −54.5%) |

---

## 1. THE OPEN PROBLEM — VRAM peak

### 1a. The target was MIS-SCOPED (corrected — use these numbers)
The 1513 MiB figure in old notes/manifest is `torch.cuda.max_memory_allocated()` over the
**whole process** (load + warmup + inference + **VAE decode**). The trace bundle is
`steady_state_inference_only` (decode/warmup are OUT of scope). The kineto trace's own running
`"Total Allocated"` counter peaks at the **inference-scope** allocated VRAM — the correct target:

| trace | **VRAM target (kineto traced peak)** | whole-process (wrong) |
|---|---|---|
| sdxl-turbo | **129.2 MiB** (135.5 MB) | 1513 MiB |
| sd3 | **464.4 MiB** (487 MB) | 1513 MiB |

Reproduce: `grep -oE '"Total Allocated": *[0-9]+' <trace>.json | grep -oE '[0-9]+' | sort -n | tail -1`
on `examples/trace/diffusers-group-offload__*/<...>_trace.json` (41726 memory events each).
(Tell that both report ~1513: it's the shared VAE decode.)

### 1b. Current sim VRAM (history; §4c birth-fix is the big mover)
| trace | sim VRAM | target | Δ | note |
|---|---|---|---|---|
| sdxl-turbo (orig) | 175.04 | 129.24 | +35% | pre-fixes |
| **sdxl-turbo (post-§4c)** | **125.17** | 129.24 | **−3.2% ✓** | within ±10% — SOLVED |
| sd3 (pre-§4a) | 1756 | 464.45 | +278% | |
| sd3 (post-§4a) | 1460.66 | 464.45 | +215% | double-claim fix |
| **sd3 (post-§4c, default)** | **719.25** | 464.45 | **+55%** | birth-fix; −741 MiB |

§4c (the producer-less `birth=0` fix, §4c below) is default-on and is the dominant mover:
SDXL is now SOLVED (125.17 ≈ 129). SD3 dropped 1460→719. Residual SD3 ~255 MiB = WEIGHT +55
(masters concurrent vs the lone 432 master) + INTERMEDIATE +183 (sim 215 vs real ~32 —
remaining *death-side* over-hold = the original §3 free-schedule; deferred).

cg-sim KB are binary (peak = `peak_num_used_pages * 4 / 1024` MiB; result.json key
`peak_memory_usage_KB`). 4 KB pages.

### 1c. Diagnosis (measured at the SD3 peak via break_lambda; see §5)
SD3 peak 1500 MiB decomposed:
- **432 MiB MASTER `[1,147456,1536]`** — streamed 20× (once/step), 0 evict points. **LEGIT** — it's
  the bulk of the *real* 464 MiB too (resident during its use). NOT the excess. (It is a real
  per-step-streamed tensor; its evict schedule being empty is benign since it's re-streamed each step,
  but verify it doesn't accumulate across steps.)
- **920 MiB "dispatcher(consumed)"** = many `[2048,6144]` / `[2,1024,6144]` MLP activation buffers
  held CONCURRENTLY. The real allocator keeps far fewer live. **THIS is the dominant excess** = the
  trace's last-consumer lifetimes run longer than the real allocator frees (activation
  over-concurrency / lifetime ceiling).
- **~137 MiB double-claim bug:** 11 tids have **2 VRAM regions** (one `is_latest=True`, one stale
  `is_latest=False`, both IDLE). These dispatcher outputs are produced by BOTH a dispatcher node
  (pre-claimed) AND a normal producer (`_ensure_outputs_claimed`) → two regions; the stale one is
  never released. CONCRETE bug.
- **125 MiB genuine activations** (gpu-produced, refcount-managed) — fine.

SDXL +35% (175 vs 129): mostly **one in-use weight master (56 MiB conv `[1280,2560,3,3]`) overlapping
the activation peak** — a timing artifact (reality has the weight resident at a different instant
than the activation peak). Hard to remove without finer timing; small absolute (~46 MiB).

Offline trace-faithful concurrency (no scheduler; `tmp/vram_timeline.py`): SDXL genuine-activation
peak 125 MiB (≈ kineto 129 ✓); SD3 926 MiB (≈ 2× kineto 464 — the activation over-concurrency).

---

## 2. WHAT NOT TO DO (dead-ends already tried)

- **storage-reuse-on-claim is UNSAFE.** Tried: on a VRAM claim, release prior IDLE regions of other
  tids sharing the same `storage_id` (assuming same storage = sequential reincarnation = prior dead).
  RESULT: SD3 aborts immediately — a real `MulFunctor` gpu compute (node 10706) needed tid 5622,
  which got freed because a **concurrently-live aliased view** shares its `storage_id`. In these
  diffusers traces `storage_id` is shared by concurrently-live views (the loader's lifetime-overlap
  aliasing did NOT merge them). A safe (no-pending-consumer) gate just duplicates refcount → no gain.
  Reverted. Do NOT retry storage-reuse without first fixing the aliasing merge (§4c).

---

## 3. PRIMARY PLAN — drive activation lifetimes from the kineto allocator timeline

> **STATUS (2026-06-02): MOSTLY RESOLVED by §4c, not by this plan.** The dominant over-concurrency
> was NOT a free-*timing* problem — it was the loader OVER-MERGING (`birth=0`) sequential
> reincarnations into union-lifetime mega-tids (see §4c). Fixing that birth bug dropped SD3
> 1460→719 and SOLVED SDXL (125.17). The kineto timeline (step 1, `tmp/kineto_vram_timeline.py`)
> was extracted and CONFIRMED the targets (max `Total Allocated` = 129.24 / 464.45 MiB) and was the
> ground truth that REVEALED the over-merge (real buffer lifetimes ~0.1 ms, not 6 s) — but the FIX
> needed no kineto matching (it uses the cg-sim graph's own first-consumer time). The matching this
> plan assumed is HARDER than written (sizes recur 1000s of times, not distinctive; clock rate IS
> 1:1 so alignable — see `tmp/align_probe.py`). Only the SD3 residual remains (death-side, below).
>
> **Residual SD3 (719→464, ~255 MiB)** = WEIGHT +55 (master eviction timing) + INTERMEDIATE +183
> (activations freed at cg-sim last-consumer, still later than the real free). THIS last piece is
> the genuine "free-schedule from kineto" work below — now scoped to the death side only.

Goal (residual): make the sim's VRAM peak track the real allocator (target 129/464). The trace's data-edge
last-consumer lifetimes over-hold activations; the kineto memory events ARE the ground-truth
alloc/free timeline.

**Blocker (known):** the bundle re-indexes `storage_id` to small ints and DROPS the device `Addr`
that the kineto memory events key on, so there's no clean tid↔allocation bridge. Workaround =
**size + time greedy match** (clock rate IS 1:1, confirmed — `tmp/align_probe.py`) within each size
class (sizes are NOT distinctive — they recur 1000s of times; confirmed).

**Steps:**
1. **Extract the kineto memory timeline** from `<trace>.json` (sdxl 0.5 GB, sd3 1.3 GB — stream/grep,
   don't `json.load` whole). Each memory event has `ts`, `Bytes` (signed: + alloc / − free), `Addr`,
   `Total Allocated`, `Device Id/Type`. Build the list of allocations: (size, alloc_ts, free_ts) by
   pairing +/− events on the same `Addr`. Sanity: the running sum peaks at 129/464.
2. **Map allocations → cg-sim tids by (size, time-order).** For each big cg-sim cuda INTERMEDIATE
   (the dispatcher/activation buffers — `tmp/vram_timeline.py` lists them), find the kineto allocation
   of matching `Bytes` whose `alloc_ts` is closest to the tid's birth (node start_ns; note the cg-sim
   trace_start_ns offset in the manifest to align clocks). Greedy match in time order; big sizes
   (452/48/24 MB) are nearly unique → reliable. Small/ambiguous ones: leave to refcount (they're a
   small fraction of VRAM).
3. **Emit a free schedule** in the loader's offload contract: `trace.args["offload"]["free_after_node"]`
   = {node_id: [tid,...]} where node_id is the graph node nearest each matched `free_ts`. (Mirror the
   existing `evict_after_node` plumbing.)
4. **Scheduler applies it** in `runtime()`: on that node's retire, release the tid's VRAM region
   (like `_evict_masters` but for activations). Keep refcount as the fallback for unmatched tids.
5. **Validate** SDXL → ~129 (±10%), SD3 → ~464 (±10%); e2e must stay within ±20% (it should — frees
   only change residency, not the critical path). Watch for the double-counted big master (don't
   double-free).

Where: extraction in a `tmp/` script first (prove the timeline + matching), then fold into
`PytorchOffloadLoader._reconstruct_offload` (gated, diffusers-variant only — re-verify accelerate).

---

## 4. SECONDARY / SUPPORTING FIXES

### 4a. Double-claim bug — ✅ DONE (2026-06-02), measured −295.6 MiB on SD3
Dispatcher outputs produced by both a dispatcher node and a normal producer got 2 VRAM regions when the
first was BEING_READ at the 2nd claim; the 2nd write `invalidate()`s the first → IDLE + not-latest. The
existing refcount path frees it only at the tid's LAST consumer (late), so dead dups pile up and inflate
the *peak* (transient: 0 remain at end-of-run — that's why they only show at the `>1500` break).
FIX (shared `AccelerateCPUOffload`): a **suspect set** registered at the 2nd-claim sites
(`_ensure_outputs_claimed` + base/subclass `_preclaim_dispatcher_outputs`) + a per-tick
`_release_stale_duplicates()` that releases not-latest IDLE regions (provably dead: no ComputeJob reads
a non-latest region, and `_dispatch_transfer` only sources is_latest). Gated by scheduler arg
`release_stale_duplicates` (default True; kill-switch + clean A/B). Counter `stale_dups_freed` in
`log_states`.
**A/B (identical code, flag off/on):** SD3 VRAM **1756.28 → 1460.66 MiB (−295.6, −16.8%)**, e2e
**5.3181 s UNCHANGED**, 570 regions reclaimed, all 378922 nodes done, 121.10 GB exact, 0 leftover dead.
NO-OP for SDXL (175.04) + accelerate 3B (8.6493 s / 777.51 MiB — exact regression) — their serial /
single-claim paths never register a suspect. (−295.6 > the old 137 estimate: 137 was an 11-tid snapshot
at the lower `>1500` break; over 20 steps the leak recurs and more dups coincide at the true peak.)
**PERF caveat (RESOLVED by §4b):** the §4a sweep transiently made SD3 slower (~13 vs ~8 min) via its
`get_by_tensor_id` calls. §4b's `MemorySpace` tid-index made all such lookups O(1), so the sweep is
now cheap and SD3 runs in ~1–2 min with §4a on. The VRAM-only-scan idea is moot; sweep left general.

### 4b. SD3 speed — ✅ DONE (2026-06-02). Premise was WRONG; real cause + fix below.
**Measured, NOT rescan churn.** Instrumented `_submit_ready`: SDXL `pops/tick=1.0` (max ready 8),
SD3 `pops/tick=1.01` (max ready 18), requeues <1% (mostly compute-cap). The `ready`-rescan the
old §4b targeted is a non-issue — the suggested residency-deferred-set re-trigger would have done
almost nothing.
**Real cause:** `MemorySpace.get_by_tensor_id` was an **O(all-regions) linear scan**, called ~10x
per node (input residency, output claim, eviction, refcount release, §4a sweep). SD3 holds ~2k RAM
+ ~6k VRAM regions → 600k nodes × ~10 scans × O(thousands) = the whole ~8-min wall. SDXL is fast
only because it has few regions. (`MemorySpace.release` had its own O(N) id-scan too.)
**FIX (core, `sim/hw/memory/common/memory_region.py`):** added a `tid -> list[region]` secondary
index maintained in `claim`/`release` (a region's tensor_id is immutable, so the only mutators are
those two). `get_by_tensor_id` now returns the per-tid list **sorted by page_idx_start** — byte-
identical to the old page-ordered scan, so the region a caller picks (begin_mutation,
`_dispatch_transfer`) is unchanged → placement/peak deterministic. `release` now looks up by the
region's own page_idx_start key (O(log N)) instead of scanning. Behavior-preserving by construction.
**Validated byte-identical** (+ a brute-vs-index consistency check, 0 mismatches, at layout AND
mid-run incl. releases): SDXL 175.04/1.2254, accelerate 3B 777.51/8.6493, **pytorch-lazy 3B vanilla
6238.7/0.1299** (non-offload, DeviceAwareVanillaAsync — core change is safe beyond offload), SD3
1460.66/5.3181 (570 dups, 121.10 GB, all nodes done).
**Speedup: SD3 ~8 min (baseline) / ~13 min (with §4a) → ≈1–2 min** (finished between the 50 s
start-timeout and the 140 s poll; coarse). The index also made §4a's vram+ram sweep cheap, so the
VRAM-only optimization I'd considered is moot — left the sweep general. The §3 test loop is now fast.

### 4c. Loader aliasing — ✅ DONE (2026-06-02). Bug was OVER-merge, not under-merge.
The aliasing WAS the dominant culprit (instinct right), but the bug is the REVERSE of what was
written here. NOT "fails to merge concurrent views" (that ceiling is only ~24 MiB — measured: at
the SD3 peak only 8 storage_ids had duplicate concurrent regions; the loader already merges nearly
all views). The real bug: in `_apply_storage_aliasing`, **producer-less tensors defaulted to
`birth=0`** (`birth.setdefault(tid, 0)`). Cuda activations whose producer was a cross-device
dispatcher / machinery node the offload reconstruction STRIPPED have no node in `output_tensors`,
so birth=0. With every reincarnation of a reused storage slot "starting" at 0, the lifetime-overlap
clustering collapsed thousands of 0.1 ms buffers into ONE union-lifetime mega-tid held for ~6 s
(e.g. one 24 MiB tid absorbed 1143 tensors, 1621 consumers across the whole run). That over-HOLD
(not view duplication) was the excess.
**FIX:** a producer-less tensor is born when **first read** (min consumer `start_ns`), not at 0.
Track `prod_birth` (from producers) and `first_use` (from consumers) separately; `birth = prod_birth
if produced else first_use`. ~15 lines in `_apply_storage_aliasing`. No kineto matching needed —
the kineto timeline only CONFIRMED the targets + revealed the bug (real lifetimes ~0.1 ms).
**Effect:** clusters ≥50 members 84→1; biggest cluster 5666→61. **SD3 1460→719.25, SDXL 175→125.17
(SOLVED, −3.2%), accel 777.5→768.55** (more accurate vs real 772.8: −0.55% vs old +0.6% — NEW
locked baseline, user-approved). All e2e unchanged & exact; all complete cleanly; loader invariants
(masters/volume/0-orphans) unchanged. Shared offload-loader code → affects accel+diffusers only
(eager/lazy use the separate PytorchProfile loader). The §2 storage-reuse dead-end is now also
safer (far fewer concurrent same-storage tids), but untested. Residual SD3 = death-side (§3).

---

## 5. HOW TO RUN / MEASURE / DEBUG

**Env:** `export LD_LIBRARY_PATH=/gnu/store/2jzflb96n50lknzn7znfq9ri4qlbifjc-zlib-1.3.1/lib PYTHONPATH=/home/kwpark/cg-sim`
(numpy/pandas need the zlib path; pure-csv analysis scripts don't).

**MCP debugger workflow** (server tool `cg-sim-mcp`):
1. `restart_simulation(input_path='examples/run/diffusers-group-offload__sdxl-turbo__diffusers.yaml')`
2. `start_simulation` → runs to finish or a breakpoint. SD3 is slow (~7 min) → `start` returns
   `timed_out=true`; re-poll with `current_state` (use a background `sleep` timer to wait).
3. VRAM-peak composition dump: `toggle_breakpoint('BREAK_AFTER_LAYOUT_STAGE')`, `start`, then
   `execute("debug.break_lambda = lambda eng,sysm: eng.sched.vram.space.num_used_pages*4/1024 > 1500")`,
   `continue_simulation`, then `execute(<dump>)`. Classify resident tids: master (`tid in sch.master_tids`)
   / initial / dispatcher(`dispatcher_outputs`) / activation(has producer) / orphan; sum bytes ×
   `len(get_by_tensor_id(tid))`. `sch = engine.sched`.
4. `BREAK_ON_ABORT` (default on) catches deadlocks: `abort_args`, `job_waiting[0]` = stuck node,
   `job_running` empty.

**Result e2e/VRAM** from `output/diffusers-group-offload__<t>__diffusers/sim_results/result.json`:
e2e = max `timestamp_end` (µs) /1e6; VRAM = `peak_memory_usage_KB` /1024 MiB.

**Offline analysis scripts (tmp/, pure-csv, fast):**
- `tmp/measure_diffusers.py {sdxl|sd3}` — Memcpy inventory, bandwidth, masters, stream structure, overlap.
- `tmp/vram_timeline.py {sdxl|sd3}` — trace-faithful concurrent VRAM peak (offline) + composition.
- `tmp/verify_diffusers.py {sdxl|sd3}` — loader reconstruction invariants (masters, volume, 0 orphans, variant).
- `tmp/verify_loader.py {3B|8B}` — ACCELERATE regression (run after ANY shared loader/scheduler change).

**Regression discipline (P12):** any change to `PytorchOffloadLoader` or `AccelerateCPUOffload`
(shared) → re-run `tmp/verify_loader.py 3B`+`8B` AND MCP accelerate 3B (must stay **8.6493 s /
768.55 MiB** — NEW baseline as of the §4c birth-fix, 2026-06-02; was 777.5 before, the change is the
fix correcting accelerate's minor over-merge, −0.55% vs real 772.8).

---

## 6. KEY MEASURED NUMBERS (don't re-measure)
| | sdxl-turbo | sd3 |
|---|---|---|
| H2D events (`Pinned→Device`) | 7435 | 20050 |
| D2H events (eviction = pointer-swap) | 3 | 2 |
| total H2D volume | 22.18 GB | 121.10 GB |
| **pinned bandwidth (aggregate)** | **26.37 GB/s** | **26.57 GB/s** |
| masters (cpu-source storage_id) | 2400 | 1851 |
| compute union vs H2D union | 86 vs 841 ms | 1100 vs 4558 ms |
| cudaMemcpyAsync host/device ratio | 0.052 (async) | 0.023 (async) |
| nodes (gpu/cpu) | 122333 (17426/104907) | ~600k |
| **e2e target / VRAM target** | 1.151 s / 129 MiB | 5.389 s / 464 MiB |

Bandwidth = total H2D bytes / total device `Memcpy HtoD` duration (the value that makes the modeled
transfers reproduce the recorded H2D wall). KBps in config = bytes/s ÷ 1000 (decimal): 26.37 GB/s →
26370000.

---

## 7. ENGINE/SCHEDULER MECHANICS RELEVANT TO VRAM (read before coding)
- VRAM peak = `MemorySpace.peak_num_used_pages` (updated on every `claim`). Match needs the right
  claim/release schedule.
- `claim(hw, tensor, page_idx)` / `release(region)` are immediate/inline. Scheduler computes first-fit
  page (`_find_free_page`). A tid can have multiple regions (the double-claim bug).
- Region flags: `is_ready`, `is_latest`, `access_status` (IDLE / BEING_READ / BEING_WRITTEN). A compute
  writing an output `invalidate()`s other regions of that tid (is_latest=False) → stale regions if not released.
- Intermediates freed by refcount (`_remaining_consumers`, `_release_intermediate` on last consumer).
  Masters schedule-evicted (`evict_after_node`). Add `free_after_node` for activations (§3.4).
- Dispatcher outputs (cpu node producing cuda) pre-claimed via `_preclaim_dispatcher_outputs`; orphans
  (no consumer) are skipped in the diffusers subclass.
- `job_waiting` is FIFO with head-of-line blocking → only submit runnable gpu computes (the §0 HOL gate).
