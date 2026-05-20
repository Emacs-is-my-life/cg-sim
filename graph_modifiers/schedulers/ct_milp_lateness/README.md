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

**Symmetric coupling: `c_t + e_{t,k} = 1`** on every feasible gap.
Locks each tid into one of two patterns:

| `c` | `e` | meaning |
|---|---|---|
| 1 | 0 | cold, locked in VRAM the whole run |
| 0 | 1 | per-iter JIT prefetch+evict cycle |

The two patterns the symmetric coupling rules out:

- `c=0, e=0` — streamed but never evicted (tensor alive from initial
  prefetch to last consumer with no PCIe round trip; peak inflates
  with no benefit). Forbidden by `c + e ≥ 1`.
- `c=1, e=1` — *hybrid*: cold at layout, evict mid-run, refetch
  before later consumers. Mathematically sound, but **not currently
  supported by the injector**. See note below.

For **c-infeasible tids** (`c_feasibility = False`, c pinned to 1 via
bounds), coupling forces `e = 0` on every feasible gap — they sit
cold the whole run.

Infeasible gaps have no `e` variable (implicit `e ≡ 0`).

### Why no hybrid mode (`c=1, e=1`)?

The hybrid pattern would let a cold-started tid reclaim VRAM during
a long inter-consumer gap and refetch from RAM before the next use.
Conceptually it's exactly the right knob for tight caps with
spread-out consumers (sd3med UNet weights, llama8b decoder weights
used across many tokens).

We tested it. The LP picks hybrid plans cleanly (integer MILP via
highspy two-phase warm-start converges), but **sim peak exceeds LP
prediction by ~1 GB on sd3med 8g** because the injector's
`coverage_repair` pass:

1. Adds the tid to `prefetch_covered_cgsim` once any prefetch arrival
   exists (the refetch is a prefetch).
2. Iterates all gpu consumers of the tid and demands each be gated by
   an async arrival in the schedule.
3. Cold-start residency *doesn't count as a gate*. So consumers
   BEFORE the mid-run evict are un-gated and the tid gets demoted
   back to fully resident — adding silent-patch VRAM overhead the LP
   didn't see.

Fixing this would need either:

- An injector change so cold-start counts as a gate for consumers in
  `[layout, first_evict_node]` (out of scope here, breaks injector
  compatibility);
- Or modeling silent-patch overhead in the LP's peak constraint when
  it picks hybrid (LP becomes harder to converge — adding ~size·c·e
  bilinear terms breaks linearity).

The symmetric coupling forbids `c=1, e=1` structurally so the LP
stays in sync with what the injector + sim will actually realize.

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
`c_t`. The peak row is `P ≥ const_i + Σ size_t · c_t`.

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
4. **Fallback**: if highspy isn't installed, the implementation falls
   back to `scipy.optimize.linprog` with `method='highs'` — same
   warm-start logic isn't exposed there, so it just runs MILP cold
   and may time out / fall back to LP relaxation rounded at 0.5.

Default time limit: 240 s. The `--audit` flag prints the c-value
distribution, MILP success status, and per-window stall breakdown.
