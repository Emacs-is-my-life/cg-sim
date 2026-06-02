# ct_milp_lateness

Pool-first MILP weight-streaming scheduler. Given a peak-VRAM cap,
chooses for each runtime tensor whether to keep it resident from
layout (cold) or stream it JIT (and per cross-iter gap, whether to
evict + refetch), minimizing the time compute waits for prefetches.

## Problem

For each cuda-resident weight tensor in the trace, decide:

- **`c_t ∈ {0, 1}`** — cold-start indicator. `c_t = 1` ⇒ tensor `t`
  is allocated at layout and stays resident the whole run. `c_t = 0`
  ⇒ JIT prefetched before its first consumer.
- **`e_{t,k} ∈ {0, 1}`** — per-gap evict-refetch. `e_{t,k} = 1` ⇒
  after consumer `k` retires, evict; before consumer `k+1` fires,
  refetch from RAM. One `e` variable per pair of consecutive
  consumers of `t` (one per cross-iter / cross-launch boundary).

Subject to **peak VRAM ≤ cap** at every moment, minimize total
**compute lateness** (the time compute jobs spend waiting for their
input prefetches to land — a direct proxy for end-to-end stall).

## Optimization domain: a flat pool

The pool is built directly from `trace.tensor_map`:

```
pool = {cgsim_tid : Tensor
        for cgsim_tid, t in trace.tensor_map.items()
        if t.tensor_type in {WEIGHT, LEAF, INPUT}
        and t.device.startswith("cuda")
        and t.size_bytes > 0
        and has at least one gpu consumer}
```

The trace loader merges aliasing cgsim_tids by `(device, storage_id)`
at load time, so each pool entry is a distinct *physical storage*.
There is no compile-side `(graph_id, compiled_tensor_id)` layer —
multi-graph and multi-iter consumers are just additional entries in
each tid's consumer list, sorted by `trace_start_ns`. Cross-graph
weight sharing, if it ever happened, would be handled implicitly
because both graphs' consumers reference the same `cgsim_tid`.

## Variables and coupling

For each pool tid `t` with `n_t` consumers:

| Variable      | Domain | Count |
|---------------|--------|-------|
| `c_t`         | {0, 1} | one per feasible pool tid (binary) |
| `e_{t,k}`     | {0, 1} | one per *feasible* cross-iter gap (binary) |
| `P`           | ℝ ≥ 0  | one global (modeled peak VRAM, bytes) |
| `L_window_i`  | ℝ ≥ 0  | one per timeline window (default 20 windows; ns of stall) |
| `s_P`         | ℝ ≥ 0  | one global (peak overrun slack, bytes) |

**Coupling: `c_t + e_{t,k} ≤ 1`** on every feasible gap whose tid is
*c-feasible* (i.e., the c bound is the free `[0, 1]` interval). This
forbids the unrealizable `c=1, e=1` pattern for tids the LP could
choose to stream — the injector would demote it back to resident.

For **c-infeasible tids** (`c_feasibility = False`, c pinned to
`(1, 1)` via bounds), the coupling row is **suppressed**. These tids
are going to be cold-loaded at layout no matter what, so allowing
`e_{t,k} = 1` on a feasible gap unlocks the *hybrid* pattern: cold
at layout, evict mid-run between widely-spaced consumers, refetch
from RAM before the next use. This is exactly the right knob for
tight caps with multi-iter weights (sd3med UNet across N steps,
llama decoder weights across N tokens) — without it those tids
would occupy peak across every dead zone.

Per-tid patterns now possible:

| `c` | `e` | meaning | available to |
|---|---|---|---|
| 1 | 0 | cold, locked in VRAM the whole run | all tids |
| 0 | 1 | per-iter JIT prefetch+evict cycle | c-feasible tids |
| 0 | 0 | streamed but never evicted across this gap | c-feasible tids |
| 1 | 1 | *hybrid* — cold at layout, mid-run evict+refetch | **c-infeasible tids only** |

⚠ **LP–sim peak gap under hybrid.** The injector's
`coverage_repair` pass treats any tid that has a prefetch arrival
as "demote candidate": consumers BEFORE the first refetch are
un-gated (cold-start residency doesn't count as a gate), so the
tid gets silently patched back to fully resident. This means the
LP's dead-zone savings (the `size · (1 − e)` term in the peak row)
won't fully materialize in sim. Empirically this gap is small
(``coverage_repair`` silent-patch overhead measured at 0 MB on the
validated grid). The dominant LP↔sim peak driver is instead the
streamed-residency working set, which the per-sample peak rows
model directly. ``safety_margin_frac`` (default 0.05) pads the
residual. The honest overrun-repair (see *Overrun repair* below)
makes the reported peak and ``target_infeasible`` truthful, so a
plan that overruns is flagged rather than silently shipped.

Infeasible gaps have no `e` variable (implicit `e ≡ 0`).

### Why hybrid for c-infeasible tids only

The hybrid pattern lets a cold-started tid reclaim VRAM during long
inter-consumer gaps and refetch from RAM before the next use.
Conceptually it's exactly the right knob for tight caps with
spread-out consumers (sd3med UNet weights, llama8b decoder weights
used across many tokens).

A first attempt enabled hybrid for *every* tid. The LP picked
clean integer hybrid plans (highspy two-phase warm-start converges
fine), but **sim peak exceeded LP prediction by ~1 GB on sd3med
8g** because of the injector's `coverage_repair` pass:

1. Adds the tid to `prefetch_covered_cgsim` once any prefetch
   arrival exists (the refetch is a prefetch).
2. Iterates all gpu consumers of the tid and demands each be gated
   by an async arrival in the schedule.
3. Cold-start residency *doesn't count as a gate*. So consumers
   BEFORE the mid-run evict are un-gated and the tid gets demoted
   back to fully resident — adding silent-patch VRAM overhead the
   LP didn't see.

We now restrict hybrid to the **c-infeasible** set (tids pinned to
`c=1` because no GPU node fires early enough for an async initial
prefetch). The reasoning:

- These tids are going to be cold-resident through `[layout,
  first_consumer]` no matter what — coverage_repair's demotion
  during that window doesn't change residency, because they were
  already resident there.
- Allowing `e_{t,k} = 1` on their feasible cross-iter gaps lets
  the LP free the VRAM the original `c+e=1` formulation forced
  into peak. This is the dominant lever for tight-cap feasibility
  on multi-iter workloads.
- For c-feasible tids the coupling row stays in place — the LP
  can still pick `c=0, e=1` (full per-iter streaming) or `c=1,
  e=0` (resident throughout); it just can't try to be cold AND
  refetch later, which the injector would demote.

The LP–sim peak gap doesn't fully vanish: even for c-pinned tids,
coverage_repair will demote across mid-run dead zones, so the
`size · (1 − e)` savings in the peak rows are partially aspirational.
Two future paths if the residual gap matters:

- Fix the injector so cold-start counts as a gate for consumers
  in `[layout, first_evict_node]` (cleanest);
- Model silent-patch overhead in the LP's peak constraint when
  it picks hybrid (would need bilinear `size · c · e` terms —
  breaks linearity).

In the meantime, bump `safety_margin_frac` on workloads where
sim peak overshoots LP peak.

## Feasibility filters

These are checks on *what's structurally expressible* by each
variable, not bans on residency strategies. They keep the LP's plan
honest about what the emitter + injector can actually realize.

**Per-gap (`gap_feasibility[k]`)** — gates the existence of
`e_{t,k}`:

- A gap is **infeasible** if `consumer_{k+1}.start − consumer_k.end <
  τ_h2d_t`, where `τ_h2d_t = h2d_latency_ns + size_t / h2d_bw`. The
  evict + refetch round trip can't physically fit in that window, so
  the tid stays resident across that gap regardless of any decision.
  No `e_{t,k}` variable is created (implicit `e ≡ 0`); the peak
  constraint records the tid as alive across the gap. D2H runs
  concurrent with H2D under duplex, so only one side of the round
  trip enters this test.

**Per-tid (`c_feasibility`)** — pins `c_t = 1` when the initial JIT
prefetch can't be async:

- A tid is **c-infeasible** if `consumer[0].start − graph_first_gpu_ns
  < τ_h2d_t`. The emitter's issuer-picker searches *within the
  consumer's graph* for a predecessor gpu node ≤ `consumer.start −
  τ_h2d`. If none exists, the emit falls back to a synchronous
  prefetch (issuer = consumer), which the injector's coverage-repair
  silently demotes back to cold. Pinning `c_t = 1` upfront matches
  what would happen anyway — but crucially the LP can still pick
  `e_{t,k} = 1` for individual gaps (the coupling is lifted for
  these tids), enabling the hybrid `(c=1, e=1)` pattern. Without
  this pin, the LP could pretend the tid is streamed and save
  cold_floor bytes the injector would silently take back.

**Both** (`!c_feasibility` AND no feasible gap): tid moves to a
`forced_cold` set and is removed from the LP entirely; its bytes
contribute as a constant floor on `P`. Usually this set is empty
(`--audit` reports `forced_cold=0` on every workload in the matrix).

## Objective

```
minimize  Σ_i L_window_i  +  1e6 · s_P
       −  ε · Σ_t size_t · c_t
       +  ε · Σ_{(t,k) feasible} size_t · e_{t,k}
```

Layers in order of magnitude:

1. **`s_P` at 1e6 ns/byte**: a hard penalty on overrunning the peak
   cap. 1 byte over cap ≈ 1 ms of equivalent lateness, dominating
   every other term. The LP fits cap whenever any feasible plan
   exists.
2. **`Σ L_window_i`**: total stall summed over all timeline windows.
   Each window's slack ≥ 0 absorbs that window's PCIe overshoot;
   summing matches the physical reality that per-window stalls
   cascade — total e2e wall-clock extension is the sum, not the max.
   This is the primary objective: minimize the actual stall time.
3. **`ε · streaming bytes` (ε = 1)**: a per-byte cold tiebreaker.
   When multiple plans tie at `Σ L_window_i = 0` (no stall), this
   pushes the LP to pick the plan with the *least* streaming —
   equivalent to "minimize PCIe traffic for free." Without this
   tiebreaker the LP picks an arbitrary feasible plan, often heavy
   streaming → worse PCIe contention → worse sim e2e.

The ε term covers both initial-prefetch bytes (`size · (1 − c_t)`,
expanded as `−ε·size·c`) and per-feasible-gap refetch bytes
(`size · e_{t,k}`). Multi-iter weights pay the size cost N times if
they evict per iteration, correctly accounting for the cycle
pattern's PCIe load.

## Peak constraint (per-moment alive-set sum)

For each of `K` sample points `T_i` along the trace timeline:

```
P  ≥  Σ_t  alive(t, T_i) · size_t  +  forced_cold_bytes  +  extras
```

Where `alive(t, T_i)` is a function of `c_t`, `e_{t,k}`, and where
`T_i` falls in `t`'s consumer pattern. Classification:

| Region                                           | Contribution    |
|--------------------------------------------------|-----------------|
| `T_i < first_consumer.start − τ` (pre-arc)       | `size · c_t`    |
| `[first_consumer.start − τ, first_consumer)`     | `size` (always) |
| at consumer (`T_i` is the consumer node)         | `size` (always) |
| dead-zone, gap `k` feasible                      | `size · c_t`    |
| dead-zone, gap `k` infeasible (no evict can fit) | `size` (always) |
| `[consumer_{k+1}.start − τ, consumer_{k+1})`     | `size` (always) |
| `T_i > last_consumer.end` (post)                 | `size · c_t`    |

The "always alive" cases get added to a constant addon at `T_i`; the
`size · c_t` cases get added to a variable-term coefficient for
`c_t`. The peak row is `P ≥ const_i + Σ size_t · c_t`. (An optional
lateness→peak coupling term was historically added to each row; it is
**disabled by default** because it over-models peak — see
*Lateness→peak coupling* below.)

Under the symmetric `c + e = 1` coupling, `(1 − e) = c`, so the
dead-zone contribution `size · (1 − e_{t,k})` reduces to
`size · c_t` — we encode the latter for a uniform row shape across
all regions. If the coupling were lifted (hybrid mode enabled),
the dead-zone term would need the `size · (1 − e)` encoding to
correctly observe the VRAM freed by mid-run eviction. See the
"Why no hybrid mode" subsection above for why we don't currently
do this.

The peak-cap row that ties `P` to the user's target lives next to
the lateness rows (see below).

## Lateness constraint (per-window PCIe budget)

The timeline `[trace_start, trace_end]` is split into `N = 20` equal
windows. For each window `i` with bounds `[s_i, e_i]`:

```
Σ_{t : first_consumer(t).start ∈ [s_i, e_i]}        δ_t · (1 − c_t)
+ Σ_{(t,k) feasible : consumer_{k+1}.start ∈ [s_i, e_i]}  δ_t · e_{t,k}
≤  (e_i − s_i)  +  L_window_i
```

where `δ_t = h2d_latency + size_t / h2d_bw` is the per-tid H2D time.

**What this models**: with `h2d_streams = 1`, the PCIe queue is
serial. In each window the queue has `(e_i − s_i)` ns of throughput
available. If the H2D work whose *deadline falls in this window*
exceeds the window length, the schedule stalls compute in that
window — `L_window_i` absorbs the excess.

**Why per-window, not a single `L_max`**: a single global slack
lets the LP "average" PCIe load across windows — it can plan a
plan where iter 0 has 200 ms of slack and iter 1 has 200 ms of
overshoot, and the max would only be 200 ms. But sim's actual
behavior is *cascading*: iter 0's slack doesn't carry forward, and
iter 1's overshoot stalls iter 1's compute regardless. Per-window
slacks force each iteration / phase to fit independently, and
summing them in the objective matches the physical wall-clock
extension (stalls add, they don't max).

D2H evictions run concurrent with H2D under duplex, so eviction
transfers don't enter the H2D budget.

A separate **soft cap row** ties `P` to the user's target:

```
P − s_P  ≤  cap · (1 − margin)
```

`margin` (default 0.07) is a safety pad that absorbs the gap between
"what the LP modeled at sample points" and "what sim actually does
at unsampled moments" — see *Sampling* below.

## Sampling (event-aligned, capped)

The LP can't write peak + lateness rows at every nanosecond. The grid
is **event-aligned**: the peak alive-set is piecewise-constant and only
transitions at (a) gpu consumer starts, (b) arc-start events
`consumer.start − τ·arc_queue_factor` (when a streamed tid's prefetch
window opens), and (c) producer events of large transient intermediates
(≥ 64 MB). Sampling at the union of those makes the modeled peak its
true peak under the residency model.

**Capping (critical).** On multi-iter workloads the arc-start events
dominate — one per consumer of every streamed tid (e.g. llama8b ≈ 22 k
arc vs 9 k consumer events, sd3-med ≈ 69 k arc). A naive grid that kept
them all made `--max-peak-samples` a no-op and the LP relaxation itself
time out. So **arc samples are thinned too**: keep the arcs of the
largest-`size` tids (they dominate peak) plus a uniform time-spread of
the rest, capped at `2 × max_peak_samples`; consumer/intermediate
samples are uniformly thinned to `max_peak_samples`. Typical post-cap
grid ≈ 768 rows regardless of trace size. `--max-peak-samples 256` is
the sweet spot.

The peak alive-set is exact at sampled moments; the `safety_margin_frac`
pad (default 0.05) absorbs unsampled transients.

## Numerical scaling (MB / ms)

The model is **solved in MB and ms** — every byte quantity (sizes, peak,
RHS, cap) and every ns quantity (δ, window, lateness) is divided by
`MODEL_SCALE = 1e6` at build time. In raw bytes/ns the problem had a
~1e16 objective dynamic range (sizes ~7e8, RHS ~2e10, slack penalty
1e6), which made HiGHS' simplex fail with `model_status = Unknown` /
"Solve error" on some models/caps even though the LP is provably
feasible. Because bytes and ns scale by the *same* factor, every ratio
coefficient (e.g. the bytes/ns PCIe rate in the lateness→peak coupling)
and all objective weights are invariant — the optimum is unchanged,
only conditioning improves (Matrix/Cost ~[1e-2, 1e6], RHS ~[1, 1e4]).
`P`, `s_P` and the per-window slacks are converted back to bytes/ns at
decode, so the emit path and diagnostics are untouched. (This replaced
an earlier brittle approach using HiGHS `user_bound_scale` /
`user_objective_scale` exponents, which weren't robust across caps.)

## `arc_queue_factor`

Widens each streamed tid's residency arc to `arc_queue_factor × τ_h2d`
in the peak rows. The intent was to model queued prefetches all holding
dst VRAM, but with `h2d_streams = 1` the simulator claims dst **at
issue, one at a time** (`_issue_prefetch`), so there is no
claim-at-enqueue pile-up. `arc_queue_factor = 1` (the default) matches
sim; values > 1 over-model peak and push the LP toward needless
stream-everything plans. Leave it at 1.

## Lateness→peak coupling (disabled by default)

An optional term — call it *option-b* — added `bw · L_window_i` bytes to
**every peak-sample row** of window `i`, the idea being "if the LP accepts
`L_window_i` ns of late streaming, those bytes are effectively cold-resident,
so charge them to peak." It was meant to stop the LP escaping the peak cap by
accepting lateness.

**It is off by default — it was mis-calibrated.** It conflates stall *time*
with resident *bytes*: at 25 MB per 1 ms of window-lateness, a window with
~100 ms of modeled stall injects ~2.5 GiB of **phantom peak** that the
simulator never actually holds. The LP, seeing a fake-high peak, under-fills
VRAM and over-streams weights — the opposite of what you want at a tight cap.
The lateness objective (`Σ L_window_i`) already discourages lateness directly,
so the coupling is redundant as well as wrong.

Measured impact (`exp_results/0601_calibrate`, validated against the cg-sim
MCP oracle — true sim peak / makespan read off the live engine):

- On sd3-med@11gib the coupling over-predicted peak by **+2.2 GiB** (modeled
  10121 vs true sim 7934 MiB). Disabling it collapses the gap to **−117 MiB**
  (modeled now ≈ true within ~1 %; the small residual is the
  intermediates/fragmentation that `safety_margin_frac` is there to absorb).
- Coupling-off + feeding the **real cap** (no inflated target) beats both the
  coupling-on plan and swapadvisor on every model at its tightest budget:

  | model | budget | MILP off | MILP on | swapadvisor |
  |---|---|---:|---:|---:|
  | sdxl-turbo | 4 GiB | **0.631 s** | 0.848 | 0.826 |
  | sd3-med | 8 GiB | **1.591 s** | 2.254 | 3.068 |
  | llama3b | 4 GiB | **1.974 s** | 2.261 | 3.258 |
  | llama8b | 10 GiB | **5.381 s** | 6.176 | 7.760 |

  (All feasible, no OOM. swapadvisor consistently leaves VRAM unused; the
  budget-filling MILP wins by streaming less.)

**Re-enable only for A/B comparison** via the `lateness_peak_coupling=True`
arg to `solve_neutral`, the `--lateness-peak-coupling` CLI flag, or
`MILP_ENABLE_LP_COUPLING=1`. `--audit` prints `coupling: ON` / `OFF (default)`.

## Tight-budget knobs (feasibility at low VRAM)

Three changes let the scheduler run at the tight budgets where it used to
return `target_infeasible` or hit a spurious sim OOM. Two are on by default
(a bugfix and a realism choice); one is opt-in.

### `intermediate_axis_fix` (param, **default True** — bugfix)

The activation-residency floor (`_build_intermediate_residencies`) was
over-counted ~10× on diffusion: modeled max **1941 MB** on sdxl vs **~192 MB**
actually resident in sim. Cause: when `baseline_sim_result` supplies sim-times,
the old code fell back to a node's **trace** ns whenever it lacked a sim-time.
Sim-axis and trace-axis are offset per node (one node: sim 67 ms, trace 469 ms),
so a sim-time producer paired with a trace-fallback consumer (e.g. a cpu_thread
consumer with no sim-time) produced a ~400 ms **phantom** residency window;
~160/219 sdxl intermediates were inflated this way. That phantom floor blocked
diffusion from targeting below ~4 GiB. The fix uses **sim-time endpoints only**
(skip no-sim-time nodes; no axis mixing), recovering ~200 MB ≈ the sim truth.
Set `--no-intermediate-axis-fix` for the ablation that exhibits the bug.

LLMs are unaffected (their intermediate residency ≈ 0). Effect on diffusion at
a binding budget: **sdxl@4gib 0.63 s → 0.45 s** (the LP keeps ~2× more weight
resident once the fake floor is gone); and sdxl now runs at **1.6 GiB / 0.82 s**
and sd3 at **6.5 GiB / 1.68 s**, vs sliding_window's 1.15 GiB/1.61 s and
6.35 GiB/7.98 s — i.e. far faster at comparable budgets.

Caveat: the fix lands *≈* actual, not provably above (sd3: 167.8 modeled vs
192 actual). The residual is covered by `safety_margin_frac`, which is therefore
load-bearing at the floor — push targeting very tight only with margin in mind.

### `relax_cinfeasible` (param, **default False** — opt-in)

A *c-infeasible* tid (load time > time before its first consumer ⇒ no async
prefetch possible) is pinned `c=1` (resident), and a tid that is also gap-less
goes to `forced_cold` (removed from the LP). When the sum of these pins exceeds
the cap, the LP is `target_infeasible` even though sliding_window runs (e.g.
llama8b@6gib: 5.05 GB pinned vs 5.8 GB cap). With `--relax-cinfeasible` these
tids become streamable LP vars (`c∈{0,1}`, plus the gap-less `forced_cold` set):
the LP may stream them, eating a startup stall (priced by the window-0 lateness
row) to free VRAM. The emit issues a sync prefetch; the injector's
`sync_fallback` path keeps it in RAM and stalls the consumer (it does **not**
demote to resident — confirmed: `demoted=0, sync_fallback=28 (2952 MB)` on
llama8b@6gib). Result: llama8b@6gib goes from infeasible → runs (8.19 s), and
loose budgets don't regress (12gib 3.44 s). Opt-in because it changes the
feasible set / decisions; validated on llama so far.

Note: for continuously-reused weights (LLM decoder weights used every token)
the residency floor is the working set, not the pins, so relaxing helps tids
with **short** consumer spans (embedding/lm_head) more than reused weights.

### `alloc_policy` (sim scheduler arg on DeviceAwareVanillaAsync — **default `first_fit`**)

Not a scheduler param — a **sim allocator** choice (`scheduler.args.alloc_policy:
first_fit | best_fit`). First-fit-from-page-0 drifts allocations low and shreds
free space under heavy evict/refetch churn, so a tightly-packed plan can fail to
find a *contiguous* slot even with ample total free (llama3b@4gib: 366 MiB free,
largest gap 43.9 MiB, a 48 MiB weight aborts). `best_fit` reuses an evicted
same-size weight's slot for the next same-size load — the anti-fragmentation the
real CUDA caching allocator gets from size-segregated free lists. Placement is
invariant for peak (byte-sum) and makespan (address-independent), so this only
changes **which runs can place**, never the resulting numbers (verified: sd3-med
@8gib byte-identical under both). Recommended `best_fit`; left default
`first_fit` pending a project-level call since it re-baselines every scheduler's
feasibility frontier in the comparison.

## Overrun repair

When the solved plan's true peak (recomputed over the sample grid with
`_alive_peak`) exceeds `cap·(1 − margin)`, the injector-bound emit would
overflow sim. The repair flips cold tids → streamed in priority order
(fewest consumers, farthest first use), **recomputing the true peak
after each batch** and stopping once it fits — converging toward the
stream-everything floor. The reported `peak_bytes` is this recomputed
value and `target_infeasible` is set truthfully when even
stream-everything overruns (cap below the streaming floor). It does
*not* under-credit by subtracting streamed bytes (the earlier bug that
silently shipped aborting plans).

## Emit: schedule entries with cgsim_tid pre-resolved

After the LP solves, each pool tid emits one of:

- **Cold (c ≥ 0.5)**: one `NeutralColdStart` anchored at the tid's
  first consumer's launch, with `cgsim_tids = [t]`.
- **Streamed (c < 0.5)**: one initial `NeutralPrefetch` before the
  first consumer (issuer picked as the latest gpu node in the
  consumer's graph with `start_ns ≤ consumer.start − τ_h2d`;
  fallback to a sync prefetch if none exists), plus per-feasible-gap
  `NeutralEvict + NeutralPrefetch` pairs targeting the specific iter
  consumer node ids (not iter-0's, since each iter's consumer is a
  distinct trace node).

Every entry carries `cgsim_tid` (or `cgsim_tids` for cold-starts)
directly. The injector reads these from `NeutralPrefetch.cgsim_tid`
etc. and skips the shape-disambiguation resolver entirely — no
`synth_gates`, no `coverage_repair` for these tids.

## Solver

**Two-phase MILP via [highspy](https://pypi.org/project/highspy/)**:

1. **Phase 1**: solve LP relaxation (continuous `c`, `e` in `[0, 1]`,
   no integrality). Fast (~1 s on sd3med scale) — gives a near-integer
   fractional point (99 %+ of binaries at boundaries under the
   symmetric coupling).
2. **Round** Phase-1 solution at 0.5: each binary set to its nearest
   integer. The symmetric coupling guarantees that 0.5-thresholding
   keeps `c + e = 1` intact (proof by case: `c ≥ 0.5` ⇒ `c = 1` and
   `e < 0.5` ⇒ `e = 0` since LP has `e = 1 − c`; similarly for the
   other case).
3. **Phase 2**: flip binaries to integer, pass the rounded values
   as warm-start via `Highs.setSolution()`, run MILP. A good
   warm-start sets a tight initial incumbent → aggressive
   branch-pruning. On the validated matrix the MILP either converges
   to integer optimality (`fell_back=False`, status `Optimal`) or
   returns a proven-integer incumbent at time limit
   (`fell_back=False`, status `Time limit reached`); either way the
   plan is a real integer solution, not a rounded relaxation.
4. **Feasibility-aware warm-start**: if Phase 1 does **not** reach
   Optimal (e.g. a numerical stall), Phase 2 is seeded with the
   *stream-everything* assignment (`c=0` for c-feasible tids, `c=1`
   for c-infeasible, `e=1` on feasible gaps), with its continuous
   slacks set to safe over-estimates so it is a valid feasible point.
   This guarantees the MILP returns a feasible incumbent rather than a
   garbage high-peak one, instead of running cold. With the MB/ms
   rescale (above) Phase 1 now reaches Optimal on every validated
   config, so this path is rarely taken — but it bounds the worst case.
5. **Fallback**: if highspy isn't installed, the implementation falls
   back to `scipy.optimize.linprog` with `method='highs'` — same
   warm-start logic isn't exposed there, so it just runs MILP cold
   and may time out / fall back to LP relaxation rounded at 0.5.

Default time limit: 240 s (`solve_neutral`); the CLI sets its own.
Use `--phase1-time-limit-s` to bound the LP relaxation separately. The
`--audit` flag prints the sample-grid breakdown, c-value distribution,
MILP success status, and per-window stall breakdown.
