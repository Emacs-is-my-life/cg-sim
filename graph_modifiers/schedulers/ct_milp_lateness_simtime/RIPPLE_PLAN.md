# ct_milp_lateness_simtime — modeling the ripple effect

**Question (user):** the relaxed lateness formulation is too simple. The
biggest gap is the **ripple effect**: if a prefetch lands late, its consumer
stalls, so *every downstream node shifts later* — and that shift is not in the
model. Can it even be modeled? Go back to the optimal problem and decide what
we can actually do.

This document (a) writes down the true optimization problem, (b) shows exactly
which piece of "ripple" each candidate mechanism does and does **not** capture,
and (c) gives an order of attack whose **first step is a measurement that
selects the path** — because the central claim ("deadline shift is second
order") is precisely what we should not assume.

---

## 0. Ground truth established before planning (verified in code)

These pin down which abstraction is even valid. All checked against the
`h2d_streams=1` config that every validated run uses
(`examples/run/*.yaml`, `scripts/sim_sweep_script.py:129`,
`scripts/sliding_window_sweep.py:100`).

1. **Compute runs in fixed program order.** `DeviceAwareVanillaAsync` dispatches
   from a node-order `ready_node_ids` deque, FIFO, dependency-gated
   (`_submit_ready_nodes_core`). A late input *stalls* a node; it does **not**
   reorder the compute stream. ⇒ a late prefetch produces a *downstream shift*,
   not a reschedule. This is what makes any fixed-order relaxation defensible.

2. **The PCIe channel is a single serial server.** With `h2d_streams=1`, one
   slot counter + one FIFO `_prefetch_queue` gate every transfer
   (`_drain_prefetch_queue`, lines 1554-1567). So H2D transfers serialize —
   the premise the whole lateness model rests on.

3. **Eviction is a free VRAM drop, not a transfer.** The scheduler emits
   `evict_after_node`, which the sim services via `_release_vram_only`
   (lines 1339-1345) — frees the VRAM region, keeps the RAM mirror, **no D2H
   copy, no PCIe**. The shared-queue D2H path (`_d2h_arrivals_by_issuer`) is
   only for hf_accelerate `module.to("cpu")` offload, and those tids are
   explicitly excluded from `evict_after_node` (line 290).
   ⇒ **The channel carries H2D only** (initial prefetch + per-gap refetch).
   The LP's "evictions don't enter the H2D budget" assumption is correct.

**Consequence:** the model we need is a *single serial server (PCIe, H2D-only)
feeding a fixed-order compute stream, subject to a VRAM cap*. That is a
single-machine-with-release-times scheduling problem coupled to a knapsack —
not a general RCPSP. This is the key simplification the sim hands us.

---

## 1. The true (optimal) problem — a max-plus recurrence

Decisions (unchanged from today): for each pool tid `t`, `c_t ∈ {0,1}`
(cold/resident vs streamed) and `e_{t,k} ∈ {0,1}` (evict+refetch across gap
`k`). These induce a **set of H2D transfer jobs** `J`: one per streamed initial
prefetch (`c_t=0`) and one per chosen refetch (`e_{t,k}=1`), each with byte size
`size_t` and channel time `δ_t = h2d_latency + size_t/bw`.

The simulator then evaluates, for compute nodes `j` in fixed program order, the
recurrence (this is the thing we're approximating):

```
  S_j  = max(  S_{prev(j)} + dur_{prev(j)},          # program-order compute
               max_{t ∈ streamed_inputs(j)} A_t  )    # wait for its prefetches
  A_t  = C_channel(t)                                 # transfer t completes
  C_channel : the finish time of job t when J is run FIFO on ONE server,
              each job released no earlier than R_t = S_{issuer(t)} (the
              issuer node must fire before its prefetch can be enqueued),
              and no earlier than VRAM has a free slot for size_t.
  VRAM(τ)  = Σ_t alive(t, τ)·size_t + intermediates(τ) ≤ cap   ∀τ
  makespan = max_j (S_j + dur_j)
```

Minimize `makespan` over `{c, e}`. Two couplings make this hard and are the
heart of "ripple":

* **`A_t` depends on `S_{issuer(t)}` which depends on upstream `A`** — the
  feedback the user is pointing at. A late arrival raises `S_j`, which raises
  the release time of every later prefetch issued off a downstream node, which
  raises their `A`, … The timeline is a **fixed point of the recurrence**, not
  a fixed input.
* **The channel finish `C_channel` is order-dependent** (FIFO over a single
  server) and **gated by the VRAM cap** (can't land a tid until its slot frees,
  i.e. until the evicting consumer has retired).

Written as a MILP this is a resource-constrained schedule with start-time
variables `S_j`, disjunctive channel-ordering binaries, and time-indexed VRAM
rows. It is **exactly formulable and completely intractable** at this scale
(8k–22k compute nodes, 70k events; today's far smaller LP already times out).
So the real work is choosing *which terms of the recurrence to linearize and
which to solve by iteration*.

---

## 2. Decomposition of "ripple" — what each mechanism actually buys

There are **two distinct phenomena** people lump into "ripple." Keep them apart;
the current model misses both, and they have different fixes.

| Phenomenon | Plain statement | Captured by |
|---|---|---|
| **(A) Backlog carryover** | The serial channel can't reset between iterations: H2D work that overflows one window is still queued in the next; early idle PCIe can pre-serve later demand. | A **cumulative-flow (conveyor) constraint** on the channel. Linear, cheap. Already prototyped behind `MILP_CUMULATIVE_LATENESS`. **Fixed deadlines.** |
| **(B) Deadline shift** | A late prefetch stalls its consumer, so *that consumer and everything after it move later in wall-clock* — relaxing later deadlines and re-timing the VRAM peak. | **No fixed-timeline LP.** Requires re-deriving the timeline from the chosen plan ⇒ a **fixed-point / re-simulation loop** around the LP. |
| **(C) Order/resource re-solve** | Full RCPSP: variable start times + channel-order + cap interacting. | Exact MILP, **intractable**; unnecessary because compute order is fixed (§0.1). |

**The user's question is specifically (B).** Be honest about this: the
cumulative form (A) is a real improvement and is the right channel model, but it
**does not shift deadlines** — it carries backlog forward on a *fixed* timeline.
Claiming "the cumulative form already models the ripple" answers the wrong half.

So the answer to *"can it even be modeled?"* is:

* **(A) yes, exactly and cheaply, inside the LP.**
* **(B) not inside a single fixed-timeline LP — but yes, by alternating
  between the LP and a timeline evaluator** (the LP picks `{c,e}` for a given
  timeline; the evaluator recomputes the timeline the recurrence produces for
  that `{c,e}`; iterate to a fixed point). This is the deliverable.
* **(C) yes in principle, no in practice — and we don't need it.**

---

## ROOT CAUSE & FIX (2026-06-05, supersedes everything below) — the model is blind to overlap

The investigation ended somewhere more fundamental than ripple. Findings, in order:

1. **Ripple (deadline-shift) does not bind** — the fixed-deadline model predicts
   actual makespan to ≤11%, optimistic, non-compounding (STEP 1 RESULTS below).
   Do not build the §5 fixed-point loop.
2. **It is not the solver** — the MILP incumbent ≈ its LP-relaxation bound. The
   MILP finds its model's optimum.
3. **The model's optimum is worse than Belady even at the full cap** (sd3@11,
   margin 0: model-best lateness 235 ms vs Belady's actual 216 ms). So the model
   is an *unfaithful relaxation* — it cannot even represent Belady's quality.
4. **The baseline that beats us does no planned prefetch.** `SwapAdvisorRuntime`
   runs with proactive prefetch OFF by default (`SWAPRT_PREFETCH`, swapadvisor_
   runtime.py:72) — it is Belady eviction + *synchronous on-demand* reloads. Yet
   it beats/ties our async-prefetch MILP. In the loose regime it stalls *less*
   than us (sd3@14: RT 93 ms vs ct 162 ms) **without prefetching at all.**

### The root cause

Prefetching has exactly one value: **overlap** — running a transfer during
compute so it leaves the makespan-critical path. **The formulation has no
representation of overlap**, so the optimizer cannot see, reward, or pursue it.

The lateness rows are `Σ δ_t (due in window) ≤ wall-clock length + L`. This:

- **Has no release / earliest-issue time** → implicitly assumes any prefetch can
  be issued arbitrarily early → **unlimited free overlap whenever the window has
  wall-clock room** → prices streaming as nearly free → **over-streams.**
- **Prices a prefetched read and a synchronous read identically** (both just
  `δ_t` in a window) → "prefetch" is invisible to the objective.

Then the emit issues each prefetch **JIT** (`_pick_issuer_node`: *latest* node
`≤ consumer − τ`, scheduler.py:2344) onto a **single serial channel**
(`h2d_streams=1`). The free overlap the model assumed is never realized — under
contention the JIT prefetch lands late and the consumer stalls. Net: **the MILP
optimizes the best *synchronous* plan under an over-optimistic free-overlap
fiction**, and an honest sync optimizer (Belady) beats that. That is why "even
without prefetching it beats us."

### The fix: model the H2D channel as a single machine with release times

Replace the wall-clock lateness windows with the **single serial channel
schedule** — the structure that makes overlap explicit and honest. Keep the peak
constraint and its sampling (overlap modeling lives only on the lateness side,
which becomes **event-based, no per-ns sampling**).

**Channel job set induced by `(c, e)`** (channel carries H2D only — evictions
are free drops, §0.3):
- one *initial-prefetch* job per streamed tid (`x = 1 − c_t`), deadline =
  `first_consumer.start`;
- one *refetch* job per chosen gap (`x = e_{t,k}`), deadline =
  `consumer_{k+1}.start`.

Each job `j` carries:
- **processing** `p_j = size_j / bw_h2d + λ` — `λ` is the non-pipelined
  per-transfer setup latency (0 if sim pipelines setup under the prior
  transfer's bandwidth phase; **calibrate against sim**, IMPROVEMENTS §3c). This
  replaces the per-tid `δ_t = h2d_latency + size/bw` whose summed `h2d_latency`
  was the optimistic-channel bias.
- **release** `r_j` = earliest the transfer can be on the channel:
  - initial prefetch: `r_j = max(d_j − W, layout_end)` (data is RAM-resident
    from the start; bounded below only by the lookahead horizon `W`, see below);
  - refetch across gap `k`: `r_j = max(consumer_k.end, d_j − W)` (the prior use
    must finish before the slot can be freed and refilled).
- **deadline** `d_j` as above (baseline timeline — STEP 1 showed this is safe).

**Variables (added):** one continuous `C_j ≥ 0` (channel completion) per job, and
one continuous `L ≥ 0` (max channel lateness = makespan extension). The `x_j` are
the existing `c`/`e` binaries — **no new binaries.**

**Rows** — sort jobs by deadline (EDF order; deadlines are fixed, so the order is
fixed). For `j = 1..M`:

```
  C_j ≥ C_{j-1} + p_j · x_j        (FIFO accumulation in EDF order — backlog/ripple carry)
  C_j ≥ (r_j + p_j) · x_j          (release respect: a selected late-released job finishes late)
  L   ≥ C_j − d_j                  (backlog at each deadline point)
  C_0 = layout_end
```

Minimizing `L` drives each `C_j` to `max(C_{j-1} + p_j x_j, (r_j+p_j) x_j)`, which
is **exactly** the EDF completion `max(C_{j-1}, r_j) + p_j` when `x_j=1` and
`C_{j-1}` when `x_j=0` (proof by the two cases `C_{j-1} ≷ r_j`). So this is not a
loose bound — it is the exact channel completion *given EDF order*. `L` is the
max positive lateness, which under fixed compute order equals the makespan
extension (the worst backlog point delays compute by `L`; downstream shifts by
≤ `L`). The recurrence carries backlog forward — this is the "ripple," modeled
correctly as a single scalar `max`, not a sum of independent windows.

**Objective:**
```
  minimize  L  +  ε · Σ_j size_j · x_j        (ε keeps the min-traffic tiebreaker)
```
Delete the 20 windows, the per-window slacks, and the lateness→peak coupling.
This **values overlap correctly**: a job with wide slack `[r_j, d_j]` sequences
before its deadline and adds *zero* `L` (genuinely hidden); a tight-slack job
costs makespan. Streaming a byte is cheap *only* if the channel can actually
sequence it in time given everything else — which **caps over-streaming by
construction.** What can't be hidden, the optimizer keeps resident (≥ Belady);
what can, it streams and overlaps (> Belady).

### The lookahead horizon `W` — ties overlap benefit to its peak cost

Early issue is what creates overlap, but a tensor prefetched early occupies its
dst VRAM from arrival to consumption — **overlap costs peak.** Rather than couple
the (continuous, variable) arrival `C_j` into the peak rows (bilinear,
intractable), bound early issue by a **lookahead horizon `W`**: a prefetch issues
at most `W` ahead of its deadline. Then *both* sides use one fixed coefficient:

- channel release `r_j = max(prior_use_end, d_j − W)` (above);
- peak residency arc width = `W` (the streamed tid is alive over `[d_j − W,
  consumer_end]` in the peak rows, replacing today's `[consumer − τ, …]`).

`W` is the single overlap↔peak tradeoff knob (this is exactly what
`SwapAdvisorRuntime`'s `prefetch_lookahead_nodes` controls). First cut: sweep a
global `W` (or a few size-buckets). The peak model stays sampled and fixed-coef;
no bilinearity.

### Emit must match the model

Change `_pick_issuer_node` from **latest** (`≤ consumer − τ`) to **earliest
allowed** (`≥ consumer − W`, i.e. issue `W` ahead), in EDF order. Otherwise the
model assumes an overlap the emit doesn't deliver — the exact model↔realization
mismatch that makes today's prefetches behave synchronously.

### Approximations (documented, deferred — none re-introduce overlap-blindness)

- **EDF channel order** (vs sim's emit order): make them agree by emitting in EDF
  order. Optimal for `1|r_j|Lmax` preemptively; a good fixed heuristic here.
- **VRAM-slot release coupling ignored:** `r_j` uses prior-use-end, not "when an
  eviction actually frees the slot under cap pressure." Exact handling is the
  full RCPSP coupling (intractable); the peak rows still enforce the cap
  independently, so this is optimism about *how early a refetch starts*, not a
  cap violation. Second-order vs the fix.
- **`λ` (per-transfer latency):** calibrate once against sim (§3c).
- **Honest peak / margin→0** is a *complementary* fix (the over-count finding):
  without it, under-fill re-creates over-streaming even with perfect overlap
  modeling. Pursue alongside.

### Why this makes MILP ≥ Belady

- *Unsaturated regime:* the channel has slack, so the model selects streaming
  that sequences before deadlines → `L → 0` → beats Belady's opportunistic sync
  stalls (the sd3@14 case where we currently *lose* to a non-prefetcher).
- *Saturated regime:* the channel schedule won't admit transfers it can't fit →
  the optimizer keeps high-reuse weights resident → streamed volume ≤ Belady →
  ties or beats on the bandwidth bound (the llama case where we currently stream
  1.8×).

Either way the MILP now optimizes **the one thing prefetching is for**, which a
sync baseline structurally cannot exploit.

### Tractability (addresses the "sampling is too slow" constraint)

- Lateness side: `+M` continuous vars, `~3M` rows, **0 new binaries**, **no
  per-ns sampling** (`M` ≈ today's initial-prefetch + feasible-gap count).
- Peak side: unchanged (keep the event-aligned sampling you rely on).
So solve time is comparable to today; the change is structural, not a size blowup.

---

## STEP 1 RESULTS (2026-06-05) — ripple does NOT bind; the loop is not justified

Ran the §3 discriminator on a full saturation sweep via
`scripts/milp_redesign_harness.py` (default config: relax_cinfeasible,
best_fit, arc=1, margin=0.05, h2d_streams=1). For each cell: LP-predicted stall
`lat_ms` (sum-window form) + per-model no-stall `floor` vs **actual sim
makespan**. (sd3 baseline was stale → regenerated; its no-stall makespan 0.988 s
matches the harness floor exactly.)

| cell | floor | pred lat | predicted mk | **actual mk** | gap | gap% | RT (Belady) | ct H2D vs RT |
|---|---|---|---|---|---|---|---|---|
| sd3@14 (loose) | 0.988 | 0.047 | 1.035 | **1.150** | +0.115 | +11% | 1.081 | 17.3 vs 16.7 GB |
| sd3@11 | 0.988 | 0.281 | 1.269 | **1.362** | +0.093 | +7% | 1.204 | 20.2 vs 16.7 GB |
| sd3@8 (tight) | 0.988 | 0.515 | 1.503 | **1.573** | +0.070 | +4.7% | 1.369 | 22.5 vs 18.1 GB |
| llama@14 | 0.266 | 1.380 | 1.646 | **1.770** | +0.124 | +7% | 0.736 | 51.4 vs 25.9 GB |
| llama@10 (sat) | 0.266 | 4.679 | 4.945 | **4.976** | +0.031 | +0.6% | 2.729 | 124 vs 70 GB |

**Three findings, all pointing away from the ripple hypothesis:**

1. **Prediction is accurate (0.6–11%) and always *optimistic* (model
   under-predicts stall).** Deadline-shift — the user's hypothesis — would make
   the fixed-deadline model *pessimistic* (it charges against early deadlines
   that have actually relaxed). The sign is the opposite of (B). So (B) is not
   the binding error.

2. **The gap does not compound.** It is a bounded ~70–125 ms offset that
   *shrinks* toward saturation and is *smallest at the tightest cap* (sd3@8,
   4.7%). A real cascade/ripple would *grow* with the number of late prefetches
   (worst at tight caps). It does the reverse → it's a roughly-constant per-run
   offset (most likely δ_t head-of-queue latency not pipelining, §3c, or a fixed
   startup serialization), not a timeline cascade.

3. **The lateness *formulation* is not the lever.** The cumulative/backlog form
   (A, `MILP_CUMULATIVE_LATENESS=1`) on sd3@11 produced an **identical plan**
   (makespan 1.361 vs 1.362 s, H2D 20330 vs 20218 MB). Sum vs cumulative does
   not change the decision.

**What the data says the real problem is.** ct_milp **loses to RT (the Belady
runtime) in every cell**, always by **streaming more H2D bytes** (e.g. sd3@11:
20.2 vs 16.7 GB; llama@10: 124 vs 70 GB). And the model **predicts its own
makespan accurately** — so this is not a *prediction* error, it is an
*optimization* error: the LP is choosing high-streaming-volume plans. At sd3@11,
11 GiB cap, it keeps only cold=8.6 GB resident and streams 20 GB; RT keeps more
resident and streams 16.7. This is the documented
[[swapadvisor_runtime_beats_ctmilp_rootcause]] regime: the peak model + ε
tiebreaker under-fill VRAM, so the plan over-evicts/over-refetches relative to
Belady.

**Revised recommendation (supersedes §5 below).**

- **Do NOT build the §5 fixed-point loop.** It would chase an effect (deadline
  shift) the data shows is negligible and wrong-signed. The §3 branch was
  "match within margin ⇒ loop not justified"; we got match.
- **The lever is plan quality: make the LP stream less / fill VRAM like Belady.**
  That is the cold-vs-stream + per-gap evict/refetch *decision* model (peak rows
  + ε objective), not the lateness rows. Concretely: the ε-per-byte streaming
  tiebreaker is too weak relative to how much the peak model under-fills; the
  objective should more aggressively maximize resident reuse (minimize total H2D
  volume) subject to cap — i.e. drive it toward the Belady frontier RT already
  achieves. This is a separate work item from anything in this ripple doc.
- **Optionally** close the ~100 ms residual: check whether queued H2D pipelines
  setup latency under the prior transfer's bandwidth phase (§3c). If sim
  pipelines it, the per-window δ_t latency sum is a small constant over-/under-
  count; cheap to fix, ~1 sim micro-check. Low priority (≤11%, doesn't change
  rankings).

The cumulative form (§4) is still a *cleaner* model than 20 arbitrary windows
and worth adopting for hygiene, but the data says it won't move makespan — so
it's not urgent.

---

## 3. STEP 1 (original plan): measure whether (B) actually matters

The plan's structure depends on one number we don't yet have. **Do not assume
deadline-shift is second-order — measure it**, because that assumption is
exactly what's in doubt and the measurement is cheap with tooling we already
have (`cgsim_mcp_oracle_recipe` memory note: read true makespan off the live
engine without the multi-GB dump).

**Discriminator.** Take 2–3 already-validated plans (e.g. sd3-med@11g,
llama8b@10g, sdxl@4g from the README table). For each:

1. From the solved plan's `{c,e}`, compute the **fixed-deadline cumulative
   Lmax** = max over prefixes of (cumulative H2D channel work released-and-due
   by `t`) − (channel time available by `t`). (This is the (A) model's predicted
   makespan extension.)
2. Read the **actual sim makespan** via the MCP oracle; subtract the
   no-stall compute floor to get the actual extension.
3. Compare `baseline_floor + Lmax_cumulative` vs `actual_makespan`.

**Branch on the result:**

* **Match within margin (say ≤5–10%)** ⇒ deadline-shift is genuinely
  second-order at these operating points. (B) is not worth a loop. **Promote
  the cumulative form (A) to default** (§4) and stop. The user's worry, while
  theoretically real, doesn't bind here — report that with the numbers.
* **Diverge** ⇒ the divergence *is* the deadline-shift. (B) is the deliverable.
  Build the fixed-point loop (§5). The size and sign of the gap also tells us
  whether the fixed-deadline model is optimistic (under-counts stall) or
  pessimistic (over-counts because it doesn't credit relaxed downstream
  deadlines) — which shapes the loop's damping.

This measurement turns "redesign the formulation" from speculation into a
decision with evidence. It is ~an afternoon of oracle calls.

---

## 4. (A) The cumulative-flow channel model — make it the lateness model

Independent of the §3 outcome, the current **default** (20 independent
equal-width wall-clock windows, slacks *summed*) is the wrong objective: it
neither carries backlog forward nor lets early idle PCIe serve later demand, and
the sum double-counts. Replace it with the single-server release-time form.

**Formulation (one slack, prefix rows):**

* Channel jobs `J` from `{c,e}` as in §1; job `t` has work `δ_t` and a
  **release time** `R_t = S_{issuer(t)}` and a **due time** `d_t =
  S_{consumer(t)}` (both from the *current* timeline — baseline on first pass).
* For every distinct due time `T` (event-aligned, **not** 20 arbitrary
  windows — this fixes IMPROVEMENTS §3b):

  ```
    Σ_{t : d_t ≤ T} δ_t · [t is streamed/refetched]   ≤   (T − t0)  +  L
  ```

  i.e. cumulative channel demand due by `T` ≤ channel time elapsed by `T`, plus
  one global backlog slack `L`. Objective minimizes `L` (= makespan extension
  under fixed order, single server). **Use the max-backlog `L`, not Σ
  per-window** — the README's "stalls cascade so sum them" is the wrong scalar;
  on a single server the makespan extension is the *max cumulative backlog*, and
  summing over-penalizes demand-late/idle-early patterns prefetch can absorb.
* Optionally add **release times** to the prefix sum (a job can't be served
  before `R_t`), making it a true 1|r_j|Lmax conveyor rather than just
  due-bucketed. This captures "a prefetch can't start before its issuer fires."

This is the model already gestured at in `MILP_CUMULATIVE_LATENESS` and
IMPROVEMENTS §3a. Two honesty notes for the code comments and README:

* Call it **"exact for the fixed-timeline relaxation,"** not "exact." The
  relaxation is the very thing (B) questions; the residual gap *is* the ripple.
* It subsumes and lets us **delete** the lateness→peak coupling (already off)
  and the 20-window heuristic.

Cross-check: re-confirm `δ_t` (does queued H2D pipeline setup latency under the
prior transfer's bandwidth phase, or serialize it? IMPROVEMENTS §3c) — the
conveyor's per-job work should be `size/bw` with latency added only if it
serializes. One sim micro-check.

---

## 5. (B) The fixed-point timeline loop — the actual ripple model

Only build this if §3 shows divergence (or to prove the loop converges to the
oracle even when the gap is small). This is how deadline-shift enters without a
monolithic MILP: **alternate** between choosing `{c,e}` and recomputing the
timeline the recurrence (§1) produces for that choice.

```
  timeline T0 ← baseline (trace or cold-all sim_times)      # current behavior
  repeat:
    {c,e} ← solve LP/MILP with deadlines/releases/peak-samples derived from T_i
    T_{i+1} ← EVALUATE(timeline produced by {c,e})          # the ripple step
  until ||T_{i+1} − T_i|| < tol   or   max_iters (2–4)
```

`EVALUATE` is the §1 recurrence run forward once for a *fixed* `{c,e}` —
O(events), deterministic, no branching. It propagates each prefetch's realized
lateness into `S_j`, shifting all downstream `S`, due times, and the
event-aligned peak-sample times. That shifted timeline is fed back as the next
LP's deadlines. **This is exactly "subsequent node timings become late and
shift," modeled by alternating projection instead of one intractable program.**

Design decisions and risks (call them out, don't paper over):

* **Which evaluator?** Two options:
  * *Internal max-plus pass* — fast, but converges to the **wrong fixed point**
    if it omits sim effects (allocator fragmentation, `coverage_repair`
    demotions, claim-at-issue VRAM timing). **Must be validated against the MCP
    oracle on a handful of plans before trusting it in the loop.**
  * *Real sim in the loop* — correct by construction, slower (one sim per
    iteration). Given 2–4 iterations and the oracle's cheap makespan read, this
    is likely the safer first cut. Start here; swap in the internal evaluator
    only once it's oracle-verified to match.
* **Convergence / oscillation.** The map `{c,e} → timeline → {c,e}` can cycle
  (a plan that's good for timeline A is bad for the timeline it induces). Damp
  it: bound iterations (2–4), and/or accept a new plan only if its
  oracle-measured makespan improves (best-so-far). Monotone acceptance turns the
  loop into a descent rather than a risk of divergence.
* **Warm-start across iterations.** Feed iteration `i`'s integer solution as the
  `setSolution()` warm-start for `i+1` (deadlines move only slightly) — keeps
  each re-solve cheap (ties into IMPROVEMENTS §1d).
* **This is the principled version of the "closed-loop sim-feedback repair"**
  already floated in IMPROVEMENTS §2 Class-A — generalized from "feed back the
  peak overrun" to "feed back the whole timeline."

---

## 6. Why not the full RCPSP (C)

For completeness, so the plan shows what we deliberately drop: the exact MILP
adds `S_j` start-time vars (~10⁴), disjunctive channel-order binaries
(O(|J|²) or a time-index), and time-indexed VRAM rows whose membership now
depends on variable `S_j` (bilinear `alive(t, S)·size`). At this scale it will
not solve, and §0.1 (fixed compute order) means its extra freedom — reordering
compute — isn't even physical in this sim. The §4+§5 split recovers everything
(C) would, minus the unphysical reorder, at tractable cost.

---

## 7. Order of attack

1. **§3 discriminator (oracle measurement).** One afternoon. *Selects the rest
   of the plan.* Without it we're guessing whether (B) binds.
2. **§4 cumulative-flow lateness as default**, event-aligned prefix rows, single
   max-backlog slack, release times. Delete the 20-window heuristic and the
   dormant lateness→peak coupling. Re-validate the README grid + oracle peaks.
   (Prerequisite hygiene from IMPROVEMENTS §1a/1bis — grid cap + MB/ms
   rescale — already landed, so the LP solves fast enough to iterate.)
3. **§5 fixed-point loop** *iff* §3 diverges (or to certify convergence):
   real-sim-in-loop first, bounded iterations + monotone acceptance + warm
   starts; then an oracle-validated internal evaluator if loop cost matters.
4. Fold in the cheap correctness items already triaged in IMPROVEMENTS that
   touch the same code: honest overrun-repair (done), `δ_t` latency-pipelining
   check (§4 cross-check), intermediate-residency on the *shifted* timeline
   (IMPROVEMENTS §3e — falls out for free once §5 supplies the timeline).

## 8. One-line summary for the user

The ripple is two things: **backlog carryover** (cheap, exact-in-relaxation, a
cumulative single-server constraint — adopt as default) and **deadline shift**
(your real concern — *not* expressible in any fixed-timeline LP, but recoverable
by a 2–4 step LP↔simulator fixed-point loop). The full variable-start-time MILP
is exact but intractable and, because compute order is fixed in this sim,
unnecessary. **First measure** (fixed-deadline cumulative Lmax vs true sim
makespan, via the oracle) whether deadline-shift actually binds at our operating
points — that single number decides whether step 5's loop is the deliverable or
an unnecessary complication.
