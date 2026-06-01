# TODO — cg-sim offload-scheduler verification

**Reading order for catching up**:
1. `README.md` — what cg-sim is and how to run it
2. `CLAUDE.md` — durable session-carried context (faithfulness principle,
   bandwidth calibration, container quirks, commit hygiene)
3. `docs/cg-sim_divergence_sources.md` — D1-D8 divergence sources between
   simulator and real run; 4-step correction hierarchy; current state;
   profile-setup recommendation
4. **This file** — current state snapshot, active work, next concrete action

Once those four are read, you should be on the same page as the prior session.
NOTE: the "DIRECTION CHANGE" section immediately below supersedes the older
"Active work / next action" further down (which described approach B).

---

## One-sentence snapshot

cg-sim's offload-scheduler verification has accelerate cpu_offload landing
within ±1% e2e for both Llama sizes, diffusers SDXL passing the ±20% bound
by accidental error-cancellation, and diffusers SD3 missing the bound
at −54.5%. We've decided the next step is to re-profile diffusers with
`leaf_level + use_stream=True + with_stack=True` to fix the root cause
upstream (at the profile-setup level) rather than continuing to
compensate downstream.

## DIRECTION CHANGE (current) — approach A, trace-driven

**Decision (supersedes approach B below).** We are NOT continuing to make the
scheduler synthesize offload behavior from the eager trace (approach B). That
path proved unviable for matching e2e time, for two evidence-backed reasons
discovered this session:

1. **The accelerate/diffusers documentation is not accurate** — `record_stream`
   semantics, matched-block eviction (docs imply pointer-swap; real traces show
   real D2H bytes, see D6), and the matched/unmatched stream split all differ
   from what the docs describe. Synthesizing a policy from docs alone produces
   the wrong behavior.
2. **e2e time is dominated by microscopic events** — per-leaf CPU hook overhead
   (~1 ms/leaf), `cudaStreamSynchronize`, `cuMem*` driver calls — far more than
   transfer bytes/bandwidth. These are invisible in the eager trace (D1, D2), so
   no scheduler synthesis on the eager trace can recover them.

**Conclusion (a real finding, not a retreat):** without the *same* trace — the
offload run's own trace, which is the most accurate description of the
scheduler's actual behavior — the simulator cannot match the real run's e2e
time. So we feed the simulator that trace.

**Approach A: trace-driven.** Feed the *offload* trace (not eager). Modify the
loader (`sim/load/pytorch_profile/`) to recognize the real transfer events and
mark them so the engine/DAV reproduces the run; let cg-sim's hardware model
compute timing.

Why A dissolves most of D1-D8 (the policy becomes ground truth, not a guess):
- D1 hook CPU chain — present in the trace as real `cpu_leaf` nodes. Solved.
- D2 Python frames — named CPU ops present; residual slivers small (with_stack
  bottom-edge reconstruction can close the rest later).
- D5 batching — each Memcpy is its own event → per-tensor transfers, ~1:1.
- D6 matched-block D2H — the D2H events are in the trace; no guessing.
- D8 tied weights — both transfers recorded separately; no tid-collapse guess.
- D3/D4 (single bandwidth knob vs concurrent H2D∥D2H streams) — the only
  residual; access pattern is exact, timing model still imperfect if recomputing.

What A trades away: it does NOT predict offload behavior for a model you only
have an *eager* trace of. You must already have the offload trace. That's the
accepted cost (per the decision above).

What A means for the offload schedulers: `AccelerateCpuOffload` and
`DiffusersGroupOffload` (synthesis logic) become largely unnecessary — the
transfers are explicit in the trace. The phantom-node compensation can be
removed once A lands (the real CPU overhead is in the trace).

### Concrete A plan (grounded in the current loader)

The building blocks already exist; this is extension, not from-scratch:
- The loader already handles HtoD-memcpy `gpu_runtime` nodes, device-crossing,
  storage aliasing, and "transfer-on-input-mismatch" firing
  (`pytorch_profile.py` ~L530, 830-880, 1054).
- There is already an `inject_schedule_path` hook (L1054+) that makes DAV
  replay a weight-streaming schedule via transfer-on-input-mismatch, **without**
  a bespoke replay scheduler. This is the most promising integration point.

Two viable designs (decide after reading the loader's node/tensor construction
path in full — NOT yet verified at line level):

  D-1. **Explicit transfer nodes.** Loader re-types `Memcpy HtoD`/`DtoH`
       `gpu_runtime` nodes into transfer operations (src/dst tensors resolved
       from data edges + device fields — done manually this session in Test E/F),
       preserving submit/wait/data deps. Engine runs them as `TransferJob`s.
       Most faithful to the recorded schedule.

  D-2. **Residency-driven (reuse `inject_schedule_path`/transfer-on-mismatch).**
       Loader marks offloaded weights RAM-resident initially (signal: any tensor
       that is the *source* of an H2D Memcpy) + extracts the eviction schedule
       (when each weight leaves VRAM, from DtoH event timing). DAV's existing
       transfer-on-input-mismatch recreates the H2D; eviction hints drive the
       D2H. Less new code; reuses DAV machinery; slight risk of re-introducing
       policy-guessing for eviction timing.

Recommended: prototype D-2 first (least new code, reuses `inject_schedule_path`),
fall back to D-1 if eviction fidelity is insufficient.

### First steps for the next session (approach A)

1. Read `sim/load/pytorch_profile/pytorch_profile.py` node/tensor construction
   end-to-end; confirm the line-level integration point for marking transfers.
   (I only spot-read it; the "X lines" estimate is NOT yet grounded.)
2. Pick a small offload trace to start: `examples/trace/llama3b_offload_model/`
   (accelerate, single pageable path — simplest; no stream concurrency, so D3/D4
   don't bite). Get A working there first before diffusers.
3. Verify: trace-driven sim e2e + peak VRAM vs the same trace's recorded span.
   Target tighter than B (±10%?) since the policy is now ground truth.
4. Then diffusers (where D3/D4 concurrent-stream timing is the open question).

### Methodological bonus to keep in mind

Once A works, it can validate B retroactively: A-sim vs real validates the
*timing model*; B-sim vs A-sim validates *policy synthesis*. Splitting the two
halves tells us exactly where residual error lives (today they're conflated,
which is why SDXL "passes by accident").

---

## (SUPERSEDED — approach B) Current verification state (as of commit `7ed0fa3` on `sim-test`)

| Workload | Sim e2e | Real e2e | Δ e2e | Δ VRAM | Bound (±20% e2e, ±10% VRAM) |
|---|---|---|---|---|---|
| Llama 3B accelerate | 11.87 s | 11.84 s | +0.3% | −8.6% | ✅ pass |
| Llama 8B accelerate | 21.85 s | 21.85 s | +0.02% | −8.4% | ✅ pass |
| SDXL diffusers (`block_level + use_stream`) | 2.11 s | 2.60 s | −19.0% | +1.8% | ⚠️ pass by accident (D3⇄D6 cancellation) |
| SD3 diffusers (`block_level + use_stream`) | 11.51 s | 25.33 s | −54.5% | −6.0% | ❌ fail |

Compensations applied (all in `sim/sched/`, no `sim/core/` changes):
- Per-leaf phantom CPU hook nodes (calibrated 1090 µs pageable / 70 µs pinned)
- Tied-weight fix in `_resolve_paged_leaves` (Llama embed/lm_head)
- PCIe bandwidth compromise (13 GB/s flat for SDXL/SD3, documented in `docs/cg-sim_bandwidth_calibration.md`)

## Active work / next action  (SUPERSEDED by "DIRECTION CHANGE" at top — kept for context)

> The approach-B plan below is no longer the active direction. We chose
> approach A (trace-driven). This section remains only to document what
> approach B would have entailed and why we know its specific flag choices.
> If approach A stalls and B is revisited, the leaf_level re-profile recipe
> here is still the right B recipe.

**Re-profile diffusers SDXL and SD3** with the configuration spelled out
in `docs/cg-sim_divergence_sources.md` ("Profiling-setup recommendation
in detail"). Specifically:

```
--group-offload-type leaf_level
--group-offload-use-stream
--with-stack
--profile-memory --record-shapes --with-modules
```

(Leave `record_stream` and `non_blocking` at defaults.)

Optional same-time re-profile of Llama accelerate runs with
`--offload-buffers --with-stack` for completeness.

This is the maximum **Step-1 leverage** (per the 4-step correction
hierarchy). The goal is to close D1, D2, D3, D4, D6, D7 simultaneously
by changing what we model rather than how we model it.

### After the new traces arrive

1. **Ingest the new trace bundles** under `examples/trace/`. Naming
   convention: `diffusers-group-offload-leaf__sdxl-turbo__RTX4090/` and
   `diffusers-group-offload-leaf__sd3__RTX4090/` would slot cleanly next
   to the existing dirs.

2. **Add new YAML configs** under `examples/run/` pointing at the new
   trace bundles. The bandwidth knob can return to honest per-Memcpy
   measurement (no more flat 13 GB/s compromise) since the leaf_level
   trace should show a single transfer path.

3. **Run verification**. Hypothesis: existing `AccelerateCpuOffload`
   scheduler may apply directly to leaf-level diffusers traces (one
   path per leaf, no matched/unmatched split). If not, a small
   `DiffusersGroupOffload` variant (or a new
   `DiffusersLeafLevelOffload` subclass) will be needed.

4. **Adjust the per-leaf phantom calibration** if `with_stack=True` now
   puts the Python interpreter time into the trace at the loader level.
   The phantom may be reducible from 1090 µs to just the
   non-Python CUDA-driver overhead (~300-400 µs based on the named-op
   accounting in `docs/cg-sim_divergence_sources.md` Path β analysis).

5. **Inspect for D6 / D7 residuals**: are matched-block D2H bytes still
   present at leaf level? Is SD3 D2H bandwidth still asymmetric? If
   either is yes, decide between scheduler-level emission, `record_stream=True`
   re-profile, or document-and-accept.

## Decision log (so the new session doesn't re-litigate)

Decisions already made and committed:

- **Faithfulness principle** added to `CLAUDE.md`: scheduler/loader
  should faithfully model the real run; phantom nodes are allowed
  when they represent real CPU work the trace doesn't capture, but
  arbitrary tuning knobs are not. The current per-leaf phantom is
  inside this principle (it models accelerate's / diffusers' real
  hook code, calibrated from inter-Memcpy gaps in reference traces).
- **No `sim/core/` changes** during the deadline-driven phase. Per-stream
  bandwidth modeling (D3/D4 fix) is deferred — would require ~200-300
  LoC in engine/hardware, too risky/expensive for this milestone.
- **Per-transfer phantom node mechanism** (Option α from the original
  discussion): scheduler injects CPU Node objects into `trace.node_map`
  during `compile()`. One phantom per TransferJob, gating the next
  prefetch. Calibration: pageable 1090 µs (Llama 3B/8B agreement),
  pinned 70 µs (SDXL).
- **Tied-weight fix** in `AccelerateCpuOffload._resolve_paged_leaves`:
  removed the `break`, accumulate all leaf modules that consume a
  WEIGHT tid.
- **Diffusers calibration acknowledged as "passing by accident"** for
  SDXL. Documented at `docs/cg-sim_bandwidth_calibration.md` and
  reiterated in `docs/cg-sim_divergence_sources.md`.

Approaches tried and rolled back:

- **`fixed_latency_micros` on memory hardware** — rejected. Per-Memcpy
  regression has intercept ≈0; the overhead isn't transfer setup, it's
  between transfers.
- **Single global per-TransferJob `cpu_overhead_us` knob** — rejected
  in favor of the two-value table (pageable + pinned) for DMA-path
  semantics.
- **Emit matched-block D2H in `DiffusersGroupOffload.compile()`** —
  tested, broke SDXL (+44%, out of bound). Reverted. Without per-stream
  bandwidth in `sim/core/`, the new D2H competes with concurrent H2D
  in cg-sim's water-filling but doesn't in reality. Documented in
  `docs/cg-sim_divergence_sources.md` D6.

## Open / unknown (to investigate when new traces arrive)

- Does `leaf_level + record_stream=False` actually pointer-swap, or
  also emit real D2H bytes per leaf? (Block-level did, despite docs.)
- Is D7 (SD3 D2H per-event bandwidth asymmetry) block_level-specific?
- How much of the per-leaf phantom can be loaded from `with_stack=True`
  trace data instead of scheduler-emitted?
- For accelerate: does `--offload-buffers` change verification numbers
  meaningfully, or only affect the RoPE buffer edge case?

## Files that matter

```
README.md                                # what cg-sim is
CLAUDE.md                                # durable session context
AGENTS.md                                # MCP debugging surface, agent runner
docs/cg-sim_divergence_sources.md        # ← D1-D8, hierarchy, profile recipe
docs/cg-sim_bandwidth_calibration.md     # SDXL/SD3 bandwidth compromise rationale
TODO.md                                  # ← you are here

sim/sched/accelerate_cpu_offload/        # accelerate scheduler + phantom emission
sim/sched/diffusers_group_offload/       # diffusers scheduler + phantom emission
sim/sched/device_aware_vanilla_async/    # base DAV scheduler with calibration table
sim/load/pytorch_profile/                # trace loader (D8 fix lives here-adjacent)

examples/run/pytorch-eager__*__accelerate_cpu_offload.yaml          # accelerate verification configs
examples/run/pytorch-eager__*__diffusers_group_offload.yaml         # diffusers verification configs

examples/trace/llama3b_offload_model/                               # accelerate Llama 3B reference
examples/trace/llama8b_offload_model/                               # accelerate Llama 8B reference
examples/trace/diffusers-group-offload__sdxl-turbo__RTX4090/        # diffusers SDXL block_level reference
examples/trace/sd3_offload_group/                                   # diffusers SD3 block_level reference (LFS)
```

## Verification commands

```bash
# Accelerate (working)
python3 main.py -i examples/run/pytorch-eager__llama-3-3B__accelerate_cpu_offload.yaml
python3 main.py -i examples/run/pytorch-eager__llama-3-8B__accelerate_cpu_offload.yaml

# Diffusers (SDXL passes-by-accident, SD3 fails)
python3 main.py -i examples/run/pytorch-eager__sdxl-turbo__diffusers_group_offload.yaml
python3 main.py -i examples/run/pytorch-eager__sd3__diffusers_group_offload.yaml

# Metrics extraction
python3 scripts/analysis/extract_sim_metrics.py output/<config>/sim_results/result.json
```

Reference numbers (e2e wall span, peak VRAM) for verification targets
are in `docs/cg-sim_divergence_sources.md` "Current state" section.

## Commit / push reminders

- **Never commit anything under `examples/trace/`** (LFS-managed, false-
  positive "modifications" appear after `git lfs pull` — see CLAUDE.md
  commit-hygiene section).
- Always stage explicit code paths; never `git add .` or `git add -A`.
- Container has no GitHub credentials by default. PAT-in-chat is the
  current workaround (revoke immediately after use); session-level
  GitHub binding is the clean fix.
