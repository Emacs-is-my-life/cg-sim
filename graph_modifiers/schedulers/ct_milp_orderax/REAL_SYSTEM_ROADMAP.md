# ct_milp_orderax → real system: remaining research steps

_Drafted 2026-06-18. Target real system: `/data/pytorch-source` weight-streaming
runtime (torch.compile/Inductor + `torch/csrc/inductor/weight_streaming_ops.cpp`)._

## Bottom line

This is **not a greenfield integration** — it is a **faithfulness / validation
research problem**. The plumbing is ~90% built:

- `solve_neutral(...)` (orderax) → `NeutralSchedule` already exists.
- `streaming_e2e.py` already registers `cgsim_milp_orderax` (line 110), resolves
  `schedulable_tids` via `map_trace_tids_to_sidecar`, sets `MILP_COLD_NO_RELEASE=1`,
  and converts to `jit_sim_prune_schedule.json` via `neutral_to_pytorch_anchored`.
- The real runtime already implements the orderax model's two load-bearing
  assumptions, with **matching default constants**:
  - in-flight paced pool: `inflight_cap_bytes = 130MB` (`TORCH_WS_INFLIGHT_MB`) ==
    sim `DAV_PACED_PREFETCH_MB=130`.
  - evict leash ("claims wait for planned evicts"): `evict_leash_bytes = 256MB`
    (`TORCH_WS_EVICT_LEASH_MB`) == the `DAV_PF_WAIT_ON_FULL` analogue.
  - `wait_miss_synced` counter = the cleanliness / faithfulness oracle.

So the contribution to defend is empirical: **a faithful order-axis plan
transfers to real hardware (realized peak ≤ cap, `miss_synced ≈ 0`), and the
prior SOTA (SwapAdvisorRuntime) — which beat the mathematical volume floor in
sim and is therefore provably unfaithful — must show up as OOM / high
`miss_synced` on real HW.** Real hardware is the final arbiter of the sim
faithfulness debate this project has been running.

## The remaining steps (ordered)

### Step 0 (engineering, blocks everything for the headline model)
**Register orderax/solve_neutral in the LLaMA harness.** `streaming_e2e_llama.py`
only lists the `cgsim_*` (old `solve()`) variants; it has **no
`NEUTRAL_VARIANT_MODULES` / `_stage_schedule_neutral` path**. orderax's headline
is the cyclic LLaMA decode, so without this the contribution model cannot run.
Port the `_stage_schedule_neutral` block from `streaming_e2e.py` (lines ~287–359)
into the llama harness (or unify the two harnesses). Cheap, mechanical.

### Step 1 (the pivot experiment — do this first, before any theory)
**Run `streaming_e2e_llama.py --variant cgsim_milp_orderax` on LLaMA-8B at the
tight cap (e.g. 6 GiB) and read three numbers:** realized peak vs cap,
`wait_miss_synced`, and end-to-end extension/makespan.
- If `miss_synced ≈ 0` and realized peak ≤ cap → the plan transfers; the paper is
  about transfer fidelity and the B/leash params are confirmed consistent. Most of
  the "risk" below evaporates.
- If not → you have localized the exact divergence empirically (which tids
  sync-restore, at which launch), which drives the next step.
This run is instrumented already; it settles the framing.

### Step 2 (threat-to-validity, now a measurement not a defect)
**Confirm model↔runtime parameter consistency** — read `weight_streaming_ops.cpp`
in full and verify:
- the sim's `constant_floor = extra_static + B(130) + max_streamed` is consistent
  with the runtime's `inflight_cap(130) + evict_leash(256)`. Direction matters:
  real in-flight < B is *conservative* (real peak below modeled, safe); only real
  in-flight **>** B is a cap violation. The harness already reports realized peak,
  so this is a measured quantity.
- whether the runtime's claim-miss path is a *wait* (matches `PF_WAIT_ON_FULL`) or a
  *synchronous safety-net H2D* (diverges from the model). `fire_skip_inflight` and
  `wait_miss_synced` tell you which fires in practice.
- pick/justify `TORCH_WS_INFLIGHT_MB` / `TORCH_WS_EVICT_LEASH_MB` so they equal the
  values the MILP was solved under. (Today they default-match at 130; document it.)

### Step 3 (headline fidelity table)
**Realized vs modeled, across models × caps.** Columns: model peak, modeled
extension, realized extension, overhead (ms), `miss_synced`, realized peak. The
SDXL version of this table already exists in `WEIGHT_STREAMING_OFFLOADING.md`
(4–24 ms overhead, miss_synced≈0); reproduce it for LLaMA (the harder, cyclic
case orderax was built for) and SD3.

### Step 4 (real-HW baselines — currently sim-only)
All orderax head-to-heads in `exp_results/` are **simulator** runs. Get real
numbers for the baselines on the same harness:
- `cgsim_belady` / `ct_maxbytes` / `ct_largest` (already wired in both harnesses).
- swapadvisor (offline planner) and the SwapAdvisorRuntime baseline. **This is the
  empirically settleable claim**: RT's 118 GB in sim is below the 148.6 GB policy-free
  volume lower bound (provably unfaithful, 1278 ungated uses) — on real HW it should
  OOM or spike `miss_synced`. Measuring that closes the faithfulness argument.

### Step 5 (generalization gaps — real engineering, not validation)
- **Multi-graph** (TE1/TE2/UNet/VAE for diffusion; cross-graph `issue_compiled_graph_id`):
  orderax solves a single unified pool; verify the cross-graph anchors emit correctly
  and the cyclic `iter_mask` / `cross_iter` ops are produced for the decode loop.
- **Cyclic / cross-iteration**: orderax's order axis is stretch-invariant within one
  invocation; confirm the per-iteration mask machinery in `neutral_to_pytorch_anchored`
  covers the repeated-wrapper decode pattern.
- **SSD→VRAM path**: the sim has `DAV_SSD_VRAM_ONLY` / direct SSD streaming; the real
  runtime's SSD path is **explicitly incomplete** (pinned-DRAM-first, not validated).
  If the contribution claims SSD-tier streaming, this is real implementation work, not
  validation. Otherwise scope it out and say so.

### Step 6 (known faithfulness loose ends carried from sim)
- The `HARNESS_GIB_TARGET=1` tail-end claim abort (reaches 96% of the run at the true
  GiB cap before a tail abort) — check whether the real runtime's leash/safety-net
  absorbs this or whether it surfaces as a late `miss_synced`/OOM. The validated
  abort-free sim result is at `margin=0.005`; decide which the real runs use.
- Seed fallback (`MILP_SEED_FALLBACK`) ships the Belady incumbent when the LP plan is
  exact-infeasible — confirm which plan actually shipped for each reported cell
  (diagnostics carry `seed_fallback`).

## What is explicitly already done (don't redo)
Identity (`compiled_launch_id`/`compiled_tensor_id`), trace→compile-space resolver
(`tid_resolve.py`), schedule JSON schema + injector, device-side H2D order pacing,
deferred-evict-on-compute-event, safety-net sync, in-flight cap + evict leash,
SDXL end-to-end measured results. The `convert_schedule` CLI and
`neutral_to_pytorch_anchored` bridge are scheduler-agnostic and already consume
orderax output.
