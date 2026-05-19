# `ct_milp_oracle`

`ct_milp_oracle` is the optimization-oriented scheduler in this
directory. It treats weight streaming as a global peak-memory
minimization problem subject to a bandwidth budget: choose which
feasible tensors get evicted (and reloaded each use) versus kept
resident, so the worst-launch peak VRAM footprint is as small as
possible without committing more H2D bytes than each graph's compute
window can drain.

The high-level goal is to answer a question that greedy schedulers only
approximate:

```text
What is the smallest peak VRAM footprint that is compatible with the
compiled-graph timeline AND with the modeled PCIe bandwidth?
```

It is a *true integer MILP* (HiGHS branch-and-bound, via
`scipy.optimize.linprog(method="highs", integrality=...)`). The
previous implementation solved an LP relaxation and threshold-rounded
at 0.5; that is now opt-in (`--lp-relaxation`) for very large bundles
where branch-and-bound times out.

## Formulation

For each evictable tensor `i` (size `s_i`), one binary decision

```text
z_i = 1   keep tensor i resident throughout (cold-start, no streaming)
z_i = 0   evict tensor i and reload before each use (cyclic streaming)
P         continuous upper bound on peak VRAM
```

Objective: `minimize P`.

### Peak constraint (per-launch)

For each unique compiled launch `(g, ℓ)` the trace contains, every
tensor that launch consumes is necessarily resident at that moment,
regardless of `z`. Tensors not consumed by `(g, ℓ)` contribute their
size only when kept (`z = 1`). So:

```text
  P  ≥  forced_keep_bytes
        +  Σ_{i : ℓ ∈ used_by_launch_ids(i)}  s_i        (constant A_{g,ℓ})
        +  Σ_{i : ℓ ∉ used_by_launch_ids(i)}  z_i · s_i
```

Distinct `(g, ℓ)` pairs that consume the same set of feasible tensors
yield identical constraints and are deduped before the solve.

### Why the formulation changed

The previous formulation bucketed time per graph and treated a tensor
as "active" in every bucket touching any of its uses. For graphs that
fire many times per pipeline call (LLM decode loops, SDXL UNet steps),
every weight had a use somewhere in every bucket, so every tensor was
counted as active everywhere → the LP forced `peak = total_weights`
regardless of `z`. Per-launch constraints remove that overcounting:
each weight is now active only at the launches that actually consume
it.

### Bandwidth constraint (per-graph aggregate)

Each graph `g` gets one budget row that bounds how many H2D bytes the
LP may commit *within g's compute window*:

```text
  Σ_{i used in g} (1 − z_i) · s_i · uses_within_g(i)
        ≤  bw · graph_compute_ns(g) · α
```

`uses_within_g(i)` is the number of times `g` consumes tensor `i`
(once per fire under cyclic streaming), so a tensor that participates
in many UNet iterations contributes its full per-pipeline traffic.
`α = bw_overcommit_frac` (default 1.0) lets the user tighten or
loosen the bound.

Per-graph caps are tighter than a single pipeline-wide cap on
multi-graph workloads — they prevent the LP from packing all of UNet's
H2D demand into TE/VAE's slack, which would yield a static-residency-
correct schedule that the runtime can't actually drain in time.

If the hardware spec sets `max_pcie_bytes_per_iter`, that single value
overrides the per-graph caps (legacy behavior).

### H2D per-rank deadlines (opt-in)

Per-graph aggregate bandwidth is an average constraint. For an extra
upper bound on within-iter H2D pressure, opt into the FIFO drain
model:

```text
  Σ_{r' ≤ r in iter k}  (1 − z_{r'}) · h2d_dur(r')   ≤   T_r
```

This is **off by default** (`disable_per_rank_deadlines = True`,
`--enable-per-rank-deadlines` to opt in). The FIFO model is fictional
— the runtime resolves bandwidth via water-filling, not strict EDF —
so adding the constraints distorts the LP toward "keep more tensors"
without genuinely matching what the simulator will replay. Use it
only when you specifically want a per-iter bound tighter than the
per-graph average.

## Feasibility pre-filter

Before solving, tensors are split into:

- `forced_keep`: locked tensors (caller-supplied) plus tensors whose
  largest gap between consecutive uses (or wraparound) is too small to
  fit a round-trip H2D + D2H — i.e., the runtime physically cannot
  evict-and-reload them on schedule.
- `feasible`: everything else. These get a `z_i` variable.

## Emission

Integer `z` lands on `keep` (cold-start, with optional async preload)
or `full evict` (cyclic prefetch + evict around each use). Under
`--lp-relaxation`, fractional `z` is rounded at 0.5.

For each evicted tensor, emission performs a backward FIFO-aware ALAP
pass: walk evicted tensors in reverse first-use order, compute
`required_finish[r] = min(first_use[r], required_start[r+1])`, then
choose the latest valid issue point (same-graph same-iter > same-graph
cross-iter wrap > cross-graph). Falls back to a synchronous reload at
the consumer if no async point fits.

## Knobs

- `--lock`: comma-separated list of `graph_input_name`s to force
  resident (e.g. embedding tables, KV cache).
- `--duplex`: assume H2D and D2H lanes are independent. Affects the
  feasibility pre-filter only (D2H deadlines aren't otherwise modeled).
- `--time-limit-s`: HiGHS branch-and-bound time limit (default 120 s).
  HiGHS will return the best feasible MILP solution found within that
  window; `keep everything` is always feasible as a fallback.
- `--bw-overcommit-frac`: scales the per-graph H2D budget (default 0.9
  — leaves 10% margin for kernel-level queueing the LP doesn't model;
  at 1.0 the LP saturates the lane exactly). Raise toward 1.0 for
  maximum streaming aggressiveness; raise above 1.0 to deliberately
  overcommit.
- `--lp-relaxation`: drop integrality up front, solve as LP and round
  at 0.5 during emission. Mostly unnecessary — when the integer MILP
  fails or hits the time limit, the solver auto-retries as LP before
  falling back to keep-everything. Use this flag only to skip the
  integer attempt entirely on bundles you know are too large.
- `--enable-per-rank-deadlines`: opt into the fictional FIFO drain
  model. Default off.

## Strengths

- Optimizes peak VRAM globally instead of relying on local ordering.
- True integer decisions — no LP-relaxation rounding artifacts (in
  the default mode).
- Captures interactions between tensors competing for the same
  per-launch peak.
- Multi-iter graphs work naturally — each launch is a separate
  constraint, and per-graph bandwidth caps absorb the multiplicity.
- Multi-graph pipelines respect per-graph compute windows, so UNet's
  bandwidth demand can't borrow from TE/VAE slack.
- Diagnostics: feasible tensor count, forced keep count, MILP peak,
  per-graph stats, solver status, integer-vs-LP mode.

## Limitations

- Per-graph bandwidth is an average constraint within each graph;
  within-iter spikes can still happen if a graph runs many iters and
  the LP packs reloads unevenly. `--enable-per-rank-deadlines` is the
  refinement, with the caveat that the FIFO model is approximate.
- Per-launch peak is the *static* live set at each launch — it does
  not model the transient overlap of in-flight prefetches between
  launches. Real peak in simulation can be slightly higher (typically
  by `pending_evict_peak` and one in-flight prefetch worth of bytes).
- Emission anchors reloads to compiled launches — if no valid async
  issue point exists, the schedule falls back to synchronous reload at
  the consumer (sacrificing async hiding for that tensor).
- Integer MILP scales to thousands of variables but takes longer than
  the LP for large bundles. For SDXL Turbo (~2.5k feasible tensors),
  use `--lp-relaxation` for sub-minute solves.

## Empirical results

Validated against `peak_greedy` (sized-EDF baseline) using
`scripts/stall_check.py`'s analytic two-lane PCIe replay:

| Bundle | Scheduler | Peak VRAM | Stall | H2D busy | Notes |
|---|---|---|---|---|---|
| llama3b-instruct | peak_greedy bw=0.05 | 6356 MB | 920 ms | 93% | |
| llama3b-instruct | **ct_milp_oracle frac=0.9** | **6350 MB** | 915 ms | 89% | matched stall |
| sdxl-turbo | peak_greedy bw=0.5 | 6629 MB | 175 s | 272% | 2 missing tensors |
| sdxl-turbo | **ct_milp_oracle --lp-relaxation** | **6141 MB** | **8 s** | **64%** | bandwidth-feasible |

The MILP doesn't always dominate by a huge margin on peak — when most
weights touch every iter (LLM decode), the bandwidth budget binds
tightly and there's little room to evict. When the workload has
streamable structure (multi-graph SDXL), MILP wins on both peak *and*
stall by 22× while staying bandwidth-feasible.

## When to use

Use `ct_milp_oracle` when you want the global peak-memory floor under
a bandwidth-feasible cyclic schedule. It is the right scheduler for
asking "how low can peak VRAM go without overcommitting PCIe?" — and
the answer it gives is now meaningful. It is less ideal when you need
a quick policy sweep (greedy is faster) or when you specifically want
a non-cyclic schedule (the LP only models keep-vs-cycle).
