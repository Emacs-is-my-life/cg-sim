# Accelerate CPU-Offload in cg-sim

How cg-sim reproduces a HuggingFace `accelerate` cpu_offload run, and — just as
importantly — **where the cg-sim model deliberately differs from the real run.**
We tried to keep the simulated trace faithful to the recorded one, but it is
**not a byte-for-byte clone**; §4 documents every place we model something
differently and why that is acceptable.

- Loader: `sim/load/pytorch_offload_loader/` (`PytorchOffloadLoader`, `_reconstruct_offload`,
  gated by the `offload_reconstruct` arg).
- Scheduler: `sim/sched/accelerate_cpu_offload/` (`AccelerateCPUOffload`).
- Configs: `examples/run/accelerate-cpu-offload__llama-3-{3B,8B}__accelerate.yaml`.
- Problem analysis + evidence: `docs/known_problems.md` (P1–P14).

---

## 1. What the real policy does

`accelerate.cpu_offload(model, execution_device)` keeps **one** copy of the model
state dict in CPU RAM. Per submodule an `AlignDevicesHook` copies that submodule's
weights **CPU→GPU just before its forward** (`pre_forward`) and **frees the GPU
copy right after** (`post_forward`, `offload=True`). Inference weights are
read-only, so the "offload" is a **free** of the GPU allocation, not a copy-back.
The copies run **synchronously on the default stream** — no prefetch overlap —
which is why the policy trades ~8–15× less VRAM for a large slowdown.

cg-sim's goal is to reproduce the **VRAM working set** (one submodule's weights
resident at a time), the **H2D streaming volume** (each weight re-streamed once
per forward), and the **end-to-end time** on the true (non-profiled) hardware.

## 2. Why the recorded trace cannot be replayed as-is

The bundle (`llama_bundle/*.csv`, schema `pytorch_runtime_v3`) is a **profiled**
run seen through the CUDA caching allocator, which is not directly simulatable:

- **No clean `WEIGHT` tensors.** Each forward streams a weight into a fresh GPU
  buffer, so the loader records each use as a distinct `tensor_kind=="CONTEXT"`
  *reincarnation* (3B: 54 417 of them; the real model is ~6 GB). The `WEIGHT`-kind
  tensors that exist are 8-byte scalars.
- **Aggregate reincarnations dwarf VRAM.** A vanilla layout would place every
  cuda-homed `INPUT`/`CONTEXT` reincarnation in VRAM and overflow 24 GB.
- **The weight→consumer link is severed in the trace** (see §4.2): the GEMM does
  not read the Memcpy's output through a data edge.

So the loader must **reconstruct** a clean representation once, with full trace
context, before the engine runs. That reconstruction is where the differences in
§4 originate.

## 3. How cg-sim models it (current architecture)

**Principle: faithful 1:1 translation.** Compute and CPU nodes become `Node`s
that replay their (probe-compensated) durations; each recorded **tensor transfer
becomes one explicit `TransferJob`** placed at its position in the graph. No
runtime reverse-engineering, no residency/stream-on-demand heuristics.

**Loader (`_reconstruct_offload`, runs before storage-aliasing):**
1. Group the recorded `Memcpy HtoD` events by their **cpu-source `storage_id`** →
   one **master** tensor per logical weight (3B: 255 masters / 6.43 GB; 8B: 292 /
   16.06 GB), kept in RAM.
2. Collapse all per-forward cuda reincarnations/views/buffers of a weight and
   **redirect every GPU consumer to read the master tid** (§4.2 explains the
   storage-based linkage).
3. **Neutralise** the recorded device `Memcpy HtoD` and host `cudaMemcpyAsync`
   (compute_time→0) and **tag each Memcpy node** `args["offload_transfer"]=
   {master, dir:"h2d"}` so the scheduler fires one transfer there.
4. Retarget **cold scalars** (cpu-labelled, gpu-only buffers) → cuda (§4.5).
5. Emit `trace.args["offload"] = {master_tids, evict_after_node}` — the per-weight
   free schedule (last GPU use in each load epoch).

**Scheduler (`AccelerateCPUOffload`):** multi-phase layout (masters→RAM,
cold/cuda-initials→VRAM); at runtime a transfer-trigger node fires a same-tid
RAM→VRAM `TransferJob` (master in VRAM ⇒ the GEMM's built-in input check passes =
load-before-use), masters are freed per `evict_after_node`, and intermediate
activations are freed by last-consumer refcount.

---

## 4. Where the cg-sim model differs from the real run (honest divergences)

The simulated trace is faithful in **behaviour** (working set, H2D volume, e2e
within bounds) but is **not** a byte-for-byte clone. Each difference below lists
what the real run does, what cg-sim does instead, and why it is acceptable.

### 4.1 Per-forward reincarnations → one logical RAM master
- **Real:** every forward allocates a fresh GPU buffer for a weight; the caching
  allocator recycles a small pool (~39 physical storage slots) across thousands
  of reincarnations.
- **cg-sim:** one master tensor per weight lives in RAM; each forward claims a
  **fresh VRAM region** (first-fit) for it via a transfer, then frees it.
- **Why acceptable / what's approximated:** inference weights are read-only, so
  every reincarnation holds byte-identical data — collapsing them changes no value
  any consumer reads. The RAM master count + footprint match accelerate's single
  state-dict copy. We do **not** reproduce the allocator's exact slot-reuse
  pattern; cg-sim's first-fit placement is functionally equivalent for peak VRAM
  and volume but is not the same allocator state.

### 4.2 Weight→GEMM linkage reconstructed from storage_id, not dataflow
- **Real trace:** the Memcpy's cuda dest is consumed by a `detach` that records
  **no output** — the data edge is severed even in the raw trace. The GEMM reads
  a separate `as_strided`/`empty_strided` *view* that merely shares the dest's
  cuda `storage_id`.
- **cg-sim:** we infer the weight↔GEMM mapping from **(cuda storage_id + lifetime
  epoch + bytes==master)** and redirect the GEMM straight to the master tid; the
  `detach`/`as_strided`/`empty_strided` machinery tensors are dropped.
- **Why acceptable / what's approximated:** the GEMM reads the same *value* (the
  weight) and runs in the same graph order. But the dependency it runs under is
  **reconstructed**, not the one the profiler recorded — we replace the recorded
  pointer-machinery dataflow with a direct master dependency. (Validated: 0
  use-before-load conflicts on both 3B and 8B.)

### 4.3 H2D modeled as a TransferJob, recorded Memcpy duration discarded
- **Real:** a device `Memcpy HtoD` kernel of a recorded duration, plus a host
  `cudaMemcpyAsync` that **blocks** for the whole copy (host/device duration ratio
  ≈ 1.006–1.012 — a synchronous copy).
- **cg-sim:** both are neutralised (compute_time→0); the bytes move via a
  RAM→VRAM `TransferJob` whose time = bytes / configured bandwidth. Neutralising
  the blocking host call is *required* — otherwise its duration would double-count
  the H2D (the recorded `cudaStreamSynchronize` is already zeroed by
  `zero_wait_nodes`).
- **Why acceptable / what's approximated:** modelling the H2D **once** is correct
  and lets timing respond to hardware/bandwidth changes (vs replaying a fixed
  recorded duration). We do **not** reproduce each Memcpy's exact recorded
  duration; we reproduce the **total** H2D wall via a calibrated rate (§4.4).

### 4.4 One linear bandwidth per memory, measured per-trace
- **Real:** PCIe H2D throughput varies with transfer size (small copies pay fixed
  overhead; large copies approach the link ceiling).
- **cg-sim:** `SimpleRAM`/`SimpleVRAM` use **one linear rate** (no per-transfer
  fixed cost), water-filled across concurrent transfers. We set it to each
  trace's **measured aggregate** rate (recorded H2D bytes ÷ recorded Memcpy time):
  **3B = 13.95 GB/s, 8B = 14.88 GB/s** on the *same* RTX-4090.
- **Why acceptable / what's approximated:** the aggregate rate reproduces the
  total H2D wall. But per-transfer times diverge from reality (small transfers
  modelled slightly too fast, large ones slightly off), and the **two configs
  need different rates** purely because their transfer-size mix differs — a direct
  symptom of the single-rate model, not a physical bandwidth difference. The rate
  is measured, never back-fit to the e2e target.

### 4.5 Cold scalars retargeted cpu→cuda
- **Real:** ~800–1000 tiny `WEIGHT` `shape=[]` 8-byte buffers (rotary `inv_freq`,
  attention scales, …) are read only by GPU kernels and are never streamed (no
  Memcpy). The profiler labels them `device=cpu`, but the kernels read them
  on-device.
- **cg-sim:** we retarget them to cuda so they are VRAM-resident from layout
  (otherwise the GPU op's input-residency check can never be met → deadlock).
- **Why acceptable / what's approximated:** ~6.6 KB total; functionally these
  *are* on-device constants in the real run, so the cpu label is the artifact and
  cuda is the faithful home. It is nonetheless an explicit modelling decision the
  trace did not state.

### 4.6 Eviction (free) schedule is imposed, not recorded
- **Real:** the trace records allocations and H2D but **not** frees; the allocator
  frees a weight's GPU buffer after the submodule forward.
- **cg-sim:** we impose per-forward eviction — free a master's VRAM after its last
  GPU use in each load epoch (epochs derived from the recorded re-load points) —
  and free activations by last-consumer refcount.
- **Why acceptable / what's approximated:** this reproduces the working set (VRAM
  peak) and the re-stream volume (one load per epoch = the recorded Memcpy count).
  The exact free *instants* are reconstructed, not replayed.

### 4.7 Transfer∥compute may overlap, vs the real synchronous single stream
- **Real:** copies and compute share one GPU stream and the host blocks on each
  copy, so H2D and compute **serialize** (no overlap).
- **cg-sim:** transfers run on the RAM/VRAM memories and compute on the CPU/GPU
  compute units — **different engine resources** — so they *can* overlap. The
  `stream_order` control chain serializes most of it, but slight overlap remains.
- **Why acceptable / what's approximated:** this is the main reason simulated e2e
  lands **~5–6 % under** the real (noprofile) time (§5). cg-sim does not model the
  single-GPU-stream serialization exactly; it relies on the dependency graph,
  which permits a little overlap the real run does not have.

### 4.8 Synthetic ordering edges for orphaned in-place alias nodes
- **Real:** an in-place `as_strided` on a cuda activation is ordered by the CUDA
  stream and its producer.
- **cg-sim:** the temporal-edge pass excludes in-place nodes, leaving some with no
  parents; we **inject a control edge** producer→alias plus a `NodeDoneDep` so the
  scheduler waits for the data and the engine bypasses the (inapplicable) CPU
  residency check for this cross-device pointer op.
- **Why acceptable / what's approximated:** it restores the real ordering and
  treats the op as the pointer/metadata op it is, but the ordering edge is
  synthesized by us, not taken from the trace.

### 4.9 Probe-effect compensation on CPU durations
- **Real (target):** the *no-profiler* run's CPU op durations.
- **cg-sim:** the trace is profiled (kineto-inflated); the loader subtracts per-op
  `probe_effect_ns` (from a sibling `probe_effect_table.csv`) to approximate true
  durations.
- **Why acceptable / what's approximated:** standard PyTorch-trace calibration,
  but the table covers a handful of aten ops, not every offload-specific op — so
  the compensation is partial.

---

## 5. Results (vs the true-hardware no-profiler run)

Target: **noprofile `inference_seconds`** (true hardware), bounds **±20 % e2e**;
and `manifest.vram_peak_allocated_bytes`, bounds **±10 % VRAM**. (We target
noprofile, not the probe-inflated profiled span — see `docs/known_problems.md`
P10.) Both configs pass on both metrics:

| Workload   | e2e sim | noprofile | Δ e2e | VRAM sim | manifest peak | Δ VRAM | H2D volume |
|------------|--------:|----------:|------:|---------:|--------------:|-------:|-----------:|
| llama-3-3B | 8.649 s | 9.098 s   | −4.9 % ✅ | 768.55 MiB | 772.8 MiB | −0.55 % ✅ | 108.20 GB (3826 xfers) exact |
| llama-3-8B | 18.345 s| 19.592 s  | −6.4 % ✅ | 1023.9 MiB | 1024.0 MiB | −0.01 % ✅ | 240.91 GB (4366 xfers) exact |

> **VRAM updated 2026-06-02 (diffusers §4c birth-fix).** Was 777.5 (+0.6%) / 1029.95 (+0.6%). The
> shared offload-loader fix (producer-less tensors born at first-use, not time 0; corrects an
> over-merge of sequential storage-slot reincarnations) also removed a *minor* over-merge in the
> accelerate traces, making both sizes MORE accurate vs the manifest peak (3B −0.55%, 8B −0.01%).
> e2e and H2D volume unchanged & exact. These are the new locked regression baselines.

- **VRAM peak (+0.6 %)** confirms the residency model: the working set is one
  weight (the embedding dominates) + a small activation set, exactly as the real
  hook leaves it — not the accumulated reincarnations.
- **H2D volume is exact** (the number of transfers equals the recorded Memcpy
  count by construction).
- **e2e is consistently ~5–6 % under** the real time. This is the honest residual
  of the divergences above — chiefly §4.7 (overlap the real synchronous stream
  doesn't have), plus uncaptured host sync/scheduling overhead and the single-rate
  bandwidth model (§4.4). It is **not** fudged to match.

Validated end-to-end through the cg-sim-mcp debugger; four modelling bugs were
found and fixed during bring-up (cold scalars, an embedding-buffer leak, an
orphan-view leak, and the no-parent alias of §4.8) — see
`docs/known_problems.md` and the project memory for the debugging trail.

---

## 6. What is deliberately NOT changed
- **Op compute durations** are the trace's (after §4.9 loader-level probe
  compensation) — the simulated *workload* is the recorded one.
- **The control/data graph** for non-weight tensors, `output_tensors`, and
  activations is untouched; only weight `input_tensors` are redirected and the
  Memcpy nodes retagged/neutralised.
- **The hardware model** is the stock RTX-4090 config (only the bandwidth knob is
  set to the measured per-trace rate, §4.4).

This keeps the modelling surface auditable: cg-sim runs the recorded workload; we
rewrite only the **weight residency + transfer representation** from the
allocator's per-forward view to the logical one-copy-in-RAM-streamed-per-forward
view, and we are explicit (above) about every place that rewrite is an
approximation rather than a clone.
