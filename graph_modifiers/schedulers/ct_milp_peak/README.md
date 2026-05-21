# ct_milp_peak

Pool-first MILP weight-streaming scheduler. Minimizes **modeled peak
VRAM** subject to a **hard zero-stall constraint** (PCIe load per
timeline window must fit inside the window's wall-clock duration; no
slack allowed).

Inverse of [`ct_milp_lateness`](../ct_milp_lateness/README.md), which
minimizes stall under a peak cap.

## When to use

- You want the LP to find the smallest peak VRAM it can achieve while
  guaranteeing no PCIe-induced stalls.
- Useful as an *upper bound on streamability*: "what's the lowest cap
  this workload can hit without slowing down?"

If you have a fixed cap and want to know the stall cost, use
`ct_milp_lateness`. If you want the lowest cap at any stall cost,
use `ct_milp_lateness` with a tight target.

## Problem

For each cuda-resident weight tensor in the trace, decide:

- **`c_t ∈ {0, 1}`** — cold-start indicator (1 ⇒ resident from layout).
- **`e_{t,k} ∈ {0, 1}`** — per-gap evict-refetch (1 ⇒ evict after
  consumer `k`, refetch before consumer `k+1`).

Subject to:

- **`Σ load(t, window_i) ≤ window_length`** for every timeline window
  (hard; no stall slack).
- **`P ≥ Σ alive(t, T_i) · size_t + forced_const`** at every sample
  point.

Minimize **`P + ε · (streamed + refetched bytes)`**, with `ε = 1e-6`
bytes/byte so the tiebreaker never overrides peak resolution.

## Variables and coupling

| Variable      | Domain | Count |
|---------------|--------|-------|
| `c_t`         | {0, 1} | one per feasible pool tid (binary) |
| `e_{t,k}`     | {0, 1} | one per *feasible* cross-iter gap (binary) |
| `P`           | ℝ ≥ 0  | one global (modeled peak VRAM, bytes) |

No `L_window_i` slacks (zero stall is hard). No `s_P` (no soft cap
target — we minimize `P` directly).

**Symmetric coupling `c_t + e_{t,k} = 1`** on every feasible gap —
same as `ct_milp_lateness`. Forbids the hybrid `(c=1, e=1)` pattern
because the injector's `coverage_repair` can't realize it; see the
lateness README for the long form.

## Why cold-all is always feasible

The cold-all plan (`c_t = 1` everywhere, `e_{t,k} = 0` everywhere)
puts zero PCIe load in every window, so `Σ load ≤ window_length`
trivially holds. The LP can never become infeasible from the
zero-stall constraint alone — the worst case is just a high `P`
(everything resident).

If the solver returns no plan due to numerical quirks, the
implementation falls back to cold-all directly.

## Objective

```
minimize  P
       −  ε · Σ_t size_t · c_t
       +  ε · Σ_{(t,k) feasible} size_t · e_{t,k}
```

with `ε = 1e-6`. `P` is in bytes (~`1e10` on typical workloads), so
1 byte of peak improvement always beats any tiebreaker shuffle.
Without the tiebreaker the LP would pick an arbitrary feasible point
among equal-peak plans — often heavier streaming than necessary.

## Peak constraint (per-moment alive-set sum)

Identical to `ct_milp_lateness`. See that README for the region
classification table and the `(1 − e) = c` substitution under
symmetric coupling.

A **global cold-floor cut**

```
Σ_t size_t · c_t  +  forced_cold_bytes  +  extras  ≤  P
```

tightens the LP relaxation; cold tids are alive at every sample by
definition, so `P` must dominate their sum.

## PCIe constraint (per-window, hard)

The timeline is split into `N = 20` equal-duration windows. For each
window `i` with bounds `[s_i, e_i]`:

```
Σ_{t : first_consumer(t).start ∈ [s_i, e_i]}        δ_t · (1 − c_t)
+ Σ_{(t,k) feasible : consumer_{k+1}.start ∈ [s_i, e_i]}  δ_t · e_{t,k}
≤  (e_i − s_i)
```

where `δ_t = h2d_latency + size_t / h2d_bw`. **No `L_window_i`** —
the budget is wall-clock duration exactly.

D2H evictions run concurrent with H2D under duplex, so eviction
transfers don't enter the H2D budget.

## Sampling

Same as `ct_milp_lateness`: 256 evenly-spaced gpu compute events out
of typically ~10k. The peak alive-set evolves between samples; moments
not sampled are not directly constrained.

**⚠ Caveat — peak is a lower bound.** Because the LP samples sparsely,
`milp_peak_mb` reports the lowest peak the LP *could see at its sample
points*. Sim may observe a higher peak at unsampled moments
(intermediate activation spikes, page fragmentation, very brief
transitions between consumers). Treat the LP peak as a lower bound;
verify in sim.

In `ct_milp_lateness` the soft cap's `safety_margin_frac` absorbs this
gap. Here there's no cap target to pad — the modeled peak *is* the
objective, so you get exactly what the LP modeled.

## Solver

Same two-phase HiGHS approach as `ct_milp_lateness` (Phase 1 LP
relaxation → round → Phase 2 MILP with warm-start). The fallback to
scipy linprog is identical.

If the MILP returns no solution at all (rare; the LP is feasible by
construction), the implementation hard-falls back to cold-all
(`c_t = 1` for every feasible tid, `e_{t,k} = 0`) and reports the
worst-case peak.

## API

```python
from graph_modifiers.schedulers.ct_milp_peak import solve_neutral

neutral = solve_neutral(
    trace,
    hw=hw,
    max_peak_samples=256,       # peak/PCIe sample count
    time_limit_s=240.0,         # HiGHS time limit; None = no limit
    lp_relaxation=False,        # debug: skip integrality
    audit=False,                # print pool/LP/solver diagnostics
)
```

CLI:

```
python -m graph_modifiers.schedulers.ct_milp_peak.main BUNDLE --hw HW.yaml [--audit]
```

Output schema (`neutral.meta`):

| Key                  | Meaning |
|----------------------|---------|
| `milp_peak_mb`       | LP-modeled peak VRAM (lower bound on sim peak) |
| `pcie_used_mb`       | streamed + refetched bytes (total H2D) |
| `cold_bytes_mb`      | bytes resident from layout |
| `streamed_bytes_mb`  | bytes JIT-prefetched at first consumer |
| `n_cold_starts` / `n_prefetches` / `n_evicts` | schedule entry counts |
