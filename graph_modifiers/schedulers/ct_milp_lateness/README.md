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

| Variable | Domain | Count |
|----------|--------|-------|
| `c_t`           | {0, 1}         | one per tid (binary) |
| `e_{t,k}`       | {0, 1}         | one per *feasible* cross-iter gap (binary) |
| `P`             | ℝ ≥ 0          | one global (modeled peak VRAM, bytes) |
| `L_max`         | ℝ ≥ 0          | one global (max consumer lateness, ns) |
| `s_P`           | ℝ ≥ 0          | one global (peak overrun slack, bytes) |

**Conditional coupling.** For tids where the initial JIT prefetch
fits (`c_feasibility = True`), the LP imposes `e_{t,k} = 1 − c_t`
on every feasible gap. This locks the tid into one of two natural
patterns:

- `c=1, e=0 ∀k` — pinned cold, resident the whole run (no PCIe).
- `c=0, e=1 ∀k` — JIT prefetched, per-iter evict+refetch cycle.

The cycle pattern matches `jit_sim_prune` and empirically gives the
right peak/PCIe tradeoff. Without coupling, the LP gravitates toward
`c=0, e=0` (streamed but never evicted, alive the whole run), which
collapses peak to near-cold-floor and fails cap on every workload.

For **forced-cold tids** (`c_feasibility = False`, meaning the
initial prefetch can't fit — `c` is pinned to 1 via variable bounds),
the coupling is **lifted**. The LP could in principle pick:

- **`c=1, e=1` for some `k`** — *hybrid*: loaded at layout for early
  consumers, evicted mid-run when its VRAM is needed elsewhere, then
  refetched from RAM before later consumers.

The motivation: a forced-cold tensor *can* be evicted later in the
run even though its initial JIT prefetch isn't viable; the LP should
be free to choose that pattern when peak pressure makes it worth one
round trip.

In the current implementation, the *coupling* is correctly lifted
(forced-cold tids' `e` variables are free), but the dead-zone peak
classification still encodes `size · c` rather than `size · (1 − e)`,
so the LP doesn't yet observe the VRAM savings from `e=1` and picks
`e=0` by default (the per-byte refetch penalty in the objective
also pushes that way). The straightforward `size · (1 − e)` rewrite
that would let hybrid mode actually fire causes HiGHS numerical
issues on tight-cap workloads — fixing this is a future LP
refinement (better matrix scaling, MIP cuts, or warm-starting). For
now the structural coupling is removed but the hybrid pattern is
not yet incentivized.

Infeasible gaps have no `e` variable (implicit `e ≡ 0`).

## Feasibility filters

Two structural reasons a gap is **infeasible** for evict-refetch:

- **Gap too narrow**: `consumer_{k+1}.start_ns − consumer_k.end_ns < τ_h2d_t`,
  where `τ_h2d_t = h2d_latency_ns + size_t / h2d_bw` is the per-tid
  H2D transfer time. The eviction + refetch round trip can't fit, so
  the tid stays resident across the gap. (D2H runs concurrent with
  H2D under duplex, so only one side of the round trip enters this
  test.)

And **c-infeasibility** — the initial prefetch can't fit:

- **No async issuer fits**: the tid's first consumer is the earliest
  gpu node in its graph. The emitter's issuer-picker searches
  *within the consumer's graph* for a predecessor gpu node ≤
  `consumer.start_ns − τ_h2d`. If none exists, the emit falls back
  to a synchronous prefetch (issuer = consumer), which the injector's
  coverage-repair typically demotes back to cold. So we pin `c_t = 1`
  upfront — the LP plans the residency explicitly and the peak
  constraint sees it.

If neither path works (`!c_feasible` AND no feasible gap), the tid
is moved into a `forced_cold` set and removed from the LP entirely;
its bytes contribute to a constant floor.

## Objective

```
minimize  L_max  +  1e6 · s_P  −  ε · Σ_t size_t · c_t
                              +  ε · Σ_{(t,k) feasible} size_t · e_{t,k}
```

Three layers, in order of magnitude:

1. **`s_P` at 1e6 ns/byte**: a hard penalty on overrunning the peak
   cap. 1 byte over cap ≈ 1 ms of equivalent lateness, dominating
   every other term. The LP fits cap whenever any feasible plan
   exists.
2. **`L_max`**: the max consumer lateness across the run (ns), the
   primary objective. The LP genuinely minimizes time compute spends
   waiting for prefetches.
3. **`ε · streaming bytes` (ε = 1)**: a per-byte cold tiebreaker.
   When multiple plans are tied at `L_max = 0` (no stall), this
   pushes the LP to pick the plan with the *least* streaming —
   equivalent to "minimize PCIe traffic for free" since cold tensors
   load once at startup. Without this tiebreaker the LP picks an
   arbitrary feasible plan, often heavy streaming → worse PCIe
   contention → worse sim e2e.

The ε term also includes refetch bytes: a refetched tid contributes
`size · e_{t,k}` per gap. This correctly accounts for multi-iter
weights where streaming pays the size cost N times (one per
refetch).

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
`c_t`. The peak row is then `P ≥ const_i + Σ size_t · c_t`.

Under the conditional coupling (c-feasible tids), the dead-zone
contribution `size · c_t` is mathematically equivalent to
`size · (1 − e_{t,k})` because `e = 1 − c` is enforced. We encode
the former because it preserves a uniform row shape across all tids
and keeps HiGHS's matrix scaling well-conditioned. The
`size · (1 − e)` rewrite that would let hybrid mode actually fire
for forced-cold tids is a future refinement (see "Conditional
coupling" above).

A separate **soft cap row** ties `P` to the user's target:

```
P − s_P  ≤  cap · (1 − margin)
```

`margin` (default 0.07) is a safety pad that absorbs the gap between
"what the LP modeled at sample points" and "what sim actually does
at unsampled moments" — see *Sampling* below.

## Lateness constraint (cumulative PCIe model)

For each sample point `T_i`:

```
Σ_{t : first_consumer(t).start ≤ T_i}       δ_t · (1 − c_t)
+ Σ_{(t,k) feasible : consumer_{k+1}.start ≤ T_i}  δ_t · e_{t,k}
≤  T_i  +  L_max
```

where `δ_t = h2d_latency + size_t / h2d_bw` is the per-tid H2D time.

**What this models**: with `h2d_streams = 1`, the PCIe queue is
serial. At any time `T`, every transfer whose consumer's deadline has
passed must have completed. The sum on the left is the total H2D
work the queue must have processed by `T_i`. If that total exceeds
`T_i`, the schedule has stalled compute — `L_max` absorbs the excess.

The maximum-lateness formulation (single `L_max` shared across all
sample-point constraints) is the right one for end-to-end time:
under cascading stalls in a serial queue, total runtime overshoots
the ideal by exactly the max cumulative excess, not the sum.

D2H evictions run concurrent with H2D under duplex, so eviction
transfers don't enter this cumulative budget.

## Sampling

The LP can't write peak + lateness rows at every nanosecond. Instead
it picks `max_peak_samples = 256` evenly-spaced gpu compute events
from the trace and writes constraints at each:

```
gpu_events:  ──┬──┬──┬──┬──┬─...─┬──┬──┬──>     ~10,000 events total
sample at:    ↑       ↑       ↑       ↑         every ~40th
              T₀      T₁     ...     T₂₅₅
```

The peak alive-set evolves between samples; moments not sampled are
not directly constrained. The 7% safety margin on the soft cap row
absorbs this sampling-induced slack. (The margin is empirical — at
256 samples, every workload in the validated matrix passes; at 128
samples some workloads fail, at 512 samples the LP becomes slower
and tighter without a meaningful safety gain.)

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

HiGHS via `scipy.optimize.linprog`. The MILP (integer `c` and `e`)
times out on workloads with >2000 binaries within the 120 s budget,
so the implementation falls back to the LP relaxation (continuous
`c`, `e` in [0, 1]) and rounds at 0.5 at emit time. Empirically the
relaxation lands near-integer (e.g., sd3-med 14g: 266 ≈ 0, 2175 ≈ 1,
only 11 fractional out of 2452) so the rounding is a small
perturbation of the relaxation's optimum. The `--audit` flag prints
the c-value distribution for inspection.
