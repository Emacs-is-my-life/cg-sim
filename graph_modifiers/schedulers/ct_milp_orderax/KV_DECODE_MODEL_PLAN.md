# Concrete model: joint weight + KV co-scheduling for KV-cached decode

_Drafted 2026-06-20. Responds to the Codex adversarial review (collapse point:
"offloaded KV cannot be a schedulable class whose PCIe transfers are hidden
without invalidating the already-tight weight stream"). The model below is JOINT
and roofline-gated by construction._

## 0. Roofline gate (analytical — build FIRST, before any MILP)

Decode is PCIe-bound under weight streaming. Numbers (llama3-8B fp16, GQA 8 KV
heads, PCIe 25 GB/s, weights ≈16 GB):

| regime | KV size | KV/weights | implication |
|--------|---------|-----------|-------------|
| ctx 4k, B=1 | 0.54 GB | 0.03× | KV trivial — **co-scheduling pointless** |
| ctx 32k, B=1 | 4.3 GB | 0.27× | KV stays resident; weights dominate |
| ctx 4k, B=32 | 17 GB | 1.07× | KV ≈ weights — **regime of interest** |
| ctx 32k, B=32 | 137 GB | 8.6× | KV dominates — **regime of interest** |
| ctx 128k, B=1 | 17 GB | 1.07× | long-context single req — interesting |
| MHA ctx 32k, B=1 | 17 GB | 1.07× | non-GQA models — interesting |

All-weight-stream ≈ **640 ms/token**; KV restore @4k/B=1 ≈ 21 ms. So per-token
PCIe is dominated by weights until KV ≳ weights.

**Gate output = a regime map**: the (B, ctx, GQA ratio, Cap, PCIe BW) box where
`KV_resident + weight_working_set > Cap` AND there is channel/compute slack to
hide KV transfer. Outside that box the answer is "keep KV resident, stream
weights only" — i.e. the *existing* problem. This directly bounds the
contribution and answers the collapse point empirically.

## Regime decision (load-bearing — pick before formulating)
The roofline forces a scope choice; each yields a *different* schedulable KV:
- **(R1) Large-batch continuous serving** — offload *inactive/low-priority
  requests'* whole KV; active request KV stays resident. KV is schedulable at
  request granularity (not per-token reread). Closest to real serving; competes
  with vLLM CPU-swap.
- **(R2) Long-context single request** — KV grows to ≳ weights; must offload
  *old* KV blocks. But exact attention rereads them every token ⇒ only viable
  with sliding-window / sparse / quantized attention, OR if PCIe slack ≥ offloaded
  bytes/token. Hardest; most novel; highest risk.
- **(R3) MHA / large-KV models** — same as R2 but KV is 4–8× larger so the
  crossover arrives at shorter context.

> Recommendation: **R1 as the defensible core** (request-granular KV is genuinely
> schedulable and doesn't fight per-token attention reads), with R2/R3 as a
> stretch only if the roofline shows real slack. The rest of this doc is written
> to cover R1 and degrade gracefully to R2.

## 1. The joint per-phase model (static, one context-length regime)

Three pools share one cap; the weight/KV split is endogenous (NOT `Cap−KV(t)`).

**Sets**
- weights `w` (per layer + embed/norm/lm_head) — as today
- KV units `k`: R1 → per *request* r; R2/R3 → per (layer ℓ, age-band a) to avoid
  per-token blowup (bucket tokens into O(10) age bands)
- channel time buckets (reuse orderax EDF)

**Decision vars (per phase p)**
- `x_w ∈ {0,1}` weight w resident(pinned) vs streamed-per-token
- `z_k ∈ {0,1}` KV unit k resident(HBM) vs offloaded(host)
- continuous channel-load / lateness vars (reuse `_solve_orderax`)

**Phase constants** — representative context `Lp` (R2) or active-set occupancy
(R1); `compute_per_token(p)` (grows with `Lp` via attention) → sets channel
release-time slack.

**Constraints**
1. Peak VRAM (JOINT): `Σ x_w·sz_w + Σ z_k·kvsz(k,p) + B_pool + act(p) ≤ Cap`.
2. Per-token PCIe load feeding the EDF channel model:
   `traffic(p) = Σ(1−x_w)·sz_w  +  Σ(1−z_k)·restore_freq(k,p)·kvsz(k,p)  +  writeback(p)`
   - R1: `restore_freq`=0 for inactive offloaded requests (not read this phase);
     restore is a one-shot at request *reactivation* (own phase).
   - R2/R3: `restore_freq`=1 (reread every token) ⇒ this term is what makes R2
     brutal; the EDF lateness objective will refuse offload unless slack exists.
3. Attention-read deadline (R2/R3): each offloaded (ℓ,a) restored before layer ℓ
   each token ⇒ on the critical channel path that phase.
4. Existence/causality: KV unit k present only when `Lp ≥ age(k)`.

**Objective** — per-token latency proxy `TPOT(p)` = per-token channel makespan
(weight stream + KV restore/writeback overlapped with compute), = orderax's
`L_max` channel model with the two-class traffic of constraint 2.

## 2. Non-stationarity = a small phase sequence, solved parametrically

KV grows monotonically; the optimal partition shifts at a few breakpoints.
- Partition decode into `P` phases by context-length bands (R2) or occupancy
  bands (R1); `P` = number of partition changes, small and **independent of token
  count** (kills the 100k-step blowup).
- Solve the §1 joint MILP per phase; **warm-start phase p+1 from p** (budgets
  change slightly). Endogenous KV means each phase *re-decides* the weight↔KV
  split — directly fixes the "exogenous Cap−KV(t)" critique.
- Total decode latency `= Σ_p tokens_in_phase(p)·TPOT(p)`.

## 3. Offline→online reconciliation (answers "offline MILP can't serve")

The MILP output is **not a timeline** — it is a **policy table** indexed by
(phase = context band / occupancy), i.e. a precomputed partition curve. At serve
time the runtime looks up the phase for the current `L`/occupancy. Solve the
parametric family offline over a representative (B, ctx) grid; validate the policy
is stable across batch composition by checking neighboring grid points agree.
This is "offline-solved policy, online-applied," not a fixed schedule.

## 4. Build & validation order

1. **Roofline notebook** (§0) → regime map + scope decision. No sim changes.
2. **Representation (sim):** capture/synthesize ONE decode-step graph
   (`use_cache=True`, seq_len=1) + a KV size formula `kvsz(L)`; add a growing/
   age-banded KV tensor to the trace model. (Closes gap-analysis layers 1–3 at
   model fidelity; full real capture deferred.)
3. **Joint MILP (sim):** extend the orderax pool to two classes + the §1 joint
   budget + two-class channel traffic. Validate: realized sim peak & TPOT match
   modeled across phases (the orderax faithfulness bar: realized ≤ modeled ≤ cap).
4. **Parametric/phase solve + policy table** (§2–3); validate policy stability on
   the (B,ctx) grid.
5. **Runtime KV offload primitive** — the missing real-system op (gap-analysis
   layer 4) — built ONLY if §0 shows viable slack.

## 5. Baselines & prior-art differentiation (answers novelty critique)
- Baselines on **TPOT vs context/batch**: KV-resident-only (weights streamed),
  vLLM-style CPU-swap (heuristic LRU), FlexGen-style throughput schedule,
  weight-stream-only.
- Differentiation to state explicitly: vLLM PagedAttention = heuristic intra-GPU/
  CPU-swap KV paging, no joint weight+KV optimization, no channel-contention/
  lateness model; FlexGen = offline *throughput*, large-batch, greedy block
  schedule, no faithful peak guarantee. **This work = joint weight+KV co-schedule
  with a faithful, contention-aware optimal model and provable peak, targeting
  latency (TPOT) under a hard VRAM cap.** Novelty lives in *joint + faithful +
  latency*, not in "offload KV" per se.

## 6. The assumption that, if false, kills it (track explicitly)
Codex's collapse point, restated as a falsifiable test in step 3:
**"there exists channel/compute slack to move KV over PCIe without inflating
TPOT beyond KV-resident-only."** If the roofline (§0) and the sim TPOT (§3) show
no slack in any practical regime, the honest result is a *negative*: weight
streaming saturates PCIe, KV must stay resident, and the contribution reduces to
the weight↔KV *resident partition* (a static knapsack) — still publishable but
much smaller. Decide go/no-go on R2 after §0+§3, not before.
