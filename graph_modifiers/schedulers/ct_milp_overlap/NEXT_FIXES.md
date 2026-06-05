# ct_milp_overlap — next fixes

Status: the overlap channel model is correct and solves to **proven optimality on
sdxl**, beating no-prefetch Belady at mid (W=5) and tight (W=2, margin=0) caps; ties
at loose. Two things block it from winning everywhere:

- **Fix 1 (tractability):** the per-job channel model makes ~36k jobs on sd3/llama →
  the C_j chain LP times out. Needed before sd3/llama solve at all. **Priority.**
- **Fix 2 (honest in-flight peak):** the per-tid arc=W over-counts peak → under-fill →
  the 5% margin is load-bearing; dropping it (which wins) isn't universally
  abort-safe. Needed to set margin→0 safely across all caps.

Plus a cheap calibration item (λ) that touches both.

---

## Fix 1 — tractable channel model (bucketed energetic form)

**Why the current form is intractable.** One completion var `C_j` + 3 chained rows per
channel job; jobs = (streamed inits) + (feasible refetch gaps) ≈ 36k on sd3 → ~108k
rows in a long dependency chain. Phase-1 LP times out. We need an O(B)-row model with
`B ≈ 256` breakpoints (the sampling scale that's known tractable), **no per-job vars**.

**The model.** Single serial channel, release times `r_j = max(prior_use_end, d_j−W)`,
processing `p_j = δ_j`, deadline `d_j`. The makespan extension is `L = max lateness`.
The exact single-machine release-time lower bound (energetic reasoning): for every time
interval `[a,b]`,

```
  Σ_{j : r_j ≥ a, d_j ≤ b}  p_j · x_j   ≤   (b − a)  +  L
```

"work that must run inside [a,b] can't exceed the channel time available there." `x_j`
is the existing binary (`1−c` for an init job, `e` for a refetch). One `L`, **zero new
binaries, no C_j**.

**Why it's O(B), not O(B²).** All `(a,b)` pairs would be O(B²) ≈ 65k rows. But
`r_j ≥ d_j − W`, so a job only enters rows whose window is **within W of its deadline**:
the only useful `a` for a given `b` lie in `[T_b − W − Δ, T_b]` (Δ = one bucket). With
W small (2–5 ms) and bucket width ≈ timeline/B, that's O(1) `a`-values per `b` → **O(B)
rows**. Add the single global prefix row per `b` (`a = T_0`) to carry long-range
backlog (the conveyor):

```
  Σ_{j : d_j ≤ T_b}  p_j · x_j   ≤   (T_b − T_0)  +  L        # backlog carry
  Σ_{j : r_j ≥ T_a, d_j ≤ T_b} p_j·x_j ≤ (T_b − T_a) + L      # release band, T_b−T_a ≲ W
```

The band rows are what make streaming *honestly priced* (a job can't be done before its
release ⇒ tight-slack streaming costs `L`); the prefix rows carry backlog. Together they
reproduce the per-job model's behavior at O(B) cost.

**Build (incremental, O(B + M)).**
1. Breakpoints `T_0..T_B`: reuse the event-aligned peak sample grid (already capped at
   ~256–768), or N iteration-aligned points. Sort.
2. Bucket each channel job by `d_j` (deadline bucket) and `r_j` (release bucket).
3. Prefix rows: accumulate `Σ p·x` by deadline bucket incrementally (one coef per
   binary, carried forward) → B rows.
4. Band rows: for each `b`, for each `a` with `T_a ∈ [T_b−W−Δ, T_b]`, the row sums jobs
   with `r_j ≥ T_a ∧ d_j ≤ T_b`. Maintain a small per-`a` running coefficient map within
   the band window; emit when `b` advances. → O(B · band_width) rows.
5. Objective unchanged: `min L + ε·Σ size·x`.

**Replaces** section 8 (`channel_jobs` C_j chain) entirely; keep `channel_jobs`
construction (it already has `(d, var, complement, p, r)`), drop the `C_IDX_BASE`
vars and the 3-row-per-job loop and the warm-start C_j computation (warm-start only
needs `L`).

**Validation.** (a) On sdxl, the bucketed form must reproduce the per-job result
(WIN at mid/tight) — same plan within rounding. (b) sd3@11 and llama8b@{12,14} must now
reach phase-1 Optimal and phase-2 within the time limit (no stream-everything fallback).
(c) Re-check the sdxl WIN survives.

---

## Fix 2 — honest in-flight peak (so margin→0 is safe everywhere)

**The bug.** The peak rows charge each streamed tid `W` of residency before its
consumer (`_pt_arc_tau = W`). But with `h2d_streams=1` the channel delivers at most
`bw·W` bytes early, so at most ~`bw·W` of "arrived-early, waiting" residency exists at
any instant — **not** `Σ over all streamed tids in their W-window`. Measured over-count:
sdxl@6 modeled 5836 vs true 5261 (+575 MB). This forces the 5% margin and blocks a clean
margin→0.

**The fix — cap the per-sample early-residency at `bw·W` (plan-dependent).** At each peak
sample `T_i`, split a streamed tid's contribution:
- **resident-for-use** window `[d, consumer_end]`: charge `size·x` as today (this is real
  residency, must be counted).
- **early/in-flight** window `[d−W, d)`: instead of charging `size·x` per tid, add a
  single auxiliary `inflight_i ≥ 0` with
  ```
    inflight_i ≥ Σ_{streamed tids in early-window at T_i} size·x   − (resident already counted)
    inflight_i ≤ bw · W                                            # channel can't pre-deliver more
  ```
  and `P ≥ const_i + Σ resident-for-use size·x + inflight_i`. The `min(Σ early-bytes,
  bw·W)` is realized by the two inequalities (LP picks the binding one). `bw·W` for
  W=5ms ≈ 125 MB (vs the +575 MB over-count today).

This is plan-dependent (cold plans → few early tids → `inflight_i ≈ 0`, no phantom
floor) and bounded (heavy streaming → capped at `bw·W`). One continuous var per sample
(~256), two rows each — cheap, no new binaries.

**Calibration (the accept test).** Build a plan, recompute modeled peak, read true sim
peak via the MCP oracle ([[cgsim_mcp_oracle_recipe]]); require `modeled ≈ true` (within
a few %) across sdxl/sd3 caps. Today modeled overshoots; target parity. Then:

**Set `safety_margin_frac` default → 0** (currently 0.05). With an honest peak, margin is
no longer load-bearing; the empirical win at sdxl@4 came from margin=0. Keep a tiny
(≤1%) pad only if calibration shows a residual under-count. Guard: the overrun-repair
(already honest) catches any real overrun and flags `target_infeasible` rather than
shipping an aborting plan.

---

## Fix 3 (cheap, do alongside) — λ calibration

The scheduler's `δ_t = h2d_latency + size/bw`. The sim's `TransferJob` charges
`fixed_latency_micros` only when an endpoint is a `BaseStorage` (SSD); for RAM→VRAM H2D
it's ~0 and the cost is pure bandwidth (`4·num_pages KB / rate`). So the per-tid
`h2d_latency` term in `δ` may be phantom for RAM-sourced streaming. Action: confirm the
RAM→VRAM fixed latency in the hw params / a 2-transfer sim micro-bench; if ~0, set
`p_j = size/bw` (drop the per-tid latency) in the channel `p` and in `tau_h2d_ns` used by
the lateness side. Small, removes a systematic bias in `L`.

---

## Sequencing

1. **Fix 1** (bucketed energetic) — unblocks sd3/llama; validate it reproduces the sdxl
   win and that sd3/llama solve to optimal. Highest priority.
2. **Fix 3** (λ) — one micro-check; fold into Fix 1's `p_j`.
3. **Fix 2** (honest in-flight peak) + **margin→0 default** — calibrate modeled==true vs
   oracle, then drop margin. This is what makes tight caps win across the board.
4. Full grid (sdxl/sd3/llama × caps × small W sweep) vs RT + sliding_window; confirm
   wins, no false-feasible aborts.
