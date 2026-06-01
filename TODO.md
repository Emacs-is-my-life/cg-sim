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

---

## One-sentence snapshot

cg-sim's offload-scheduler verification has accelerate cpu_offload landing
within ±1% e2e for both Llama sizes, diffusers SDXL passing the ±20% bound
by accidental error-cancellation, and diffusers SD3 missing the bound
at −54.5%. We've decided the next step is to re-profile diffusers with
`leaf_level + use_stream=True + with_stack=True` to fix the root cause
upstream (at the profile-setup level) rather than continuing to
compensate downstream.

## Current verification state (as of commit `7ed0fa3` on `sim-test`)

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

## Active work / next action

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
