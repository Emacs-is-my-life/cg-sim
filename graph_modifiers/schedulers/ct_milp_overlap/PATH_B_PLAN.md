# ct_milp_overlap — Path B (faithful model) + λ plan

## Why (established, not hypothesized)

The MILP reaches **proven optimality** on sd3@11 in ~3 min (LP relaxation is
near-integral, 11 fractional/36k), and the **proven-optimal plan still loses**:
true peak 10952 < cap 11264 (under-fill), streams 19.7 GB vs RT's 16.8, mk 1.269
vs 1.205. So it is **not** a solver problem — the model's true optimum is a
losing plan. The optimizer is trustworthy; **the model it optimizes is not
faithful to sim**. Two faithfulness gaps, both measured:

- **Peak over-predicts true** by ~300–575 MB → at the cap the LP stops filling
  while sim still has room → under-fill → over-stream (reuse amplifies it).
- Even with peak filled (Path A inflation → true peak 11240 ≈ cap), it **still
  over-streams vs Belady** (19 vs 16.8 GB) and loses → the **L/objective also
  doesn't match sim** (it under-prices the makespan cost of that streaming).

Goal: make `modeled_peak(x) == true_sim_peak(x)` and `modeled_makespan(x) ==
true_makespan(x)` for *arbitrary* plans `x`. Then margin→0 is safe and the
optimizer's argmin is the true argmin.

Honest payoff bound: sd3/llama are bandwidth-bound where Belady is near-optimal,
so the target there is **parity (stop losing)**; the genuine *win* is the sdxl
mid-regime. Path B's value = don't-lose-on-sd3/llama + a stable margin→0 across
all caps.

---

## Step 0 — instrumentation (do first; makes the rest measurable)

Without attribution we're guessing which term over-counts. Build a one-shot
**modeled-vs-true peak diff at the binding instant**:

1. **Model side** (already have the data): at solve end, find the sample that
   achieves `_alive_peak` (the binding sample), and dump its decomposition —
   `const_addons` split into {forced_cold, extras, intermediates, in-flight bw·W}
   plus the per-tid `size·c`/`size` variable terms that are "on" in the solved
   plan. `peak_sample_terms` already holds (const, [(col,coef)]) per sample; add
   a helper that returns the categorized breakdown at argmax.
2. **Sim side**: hook DAV at its peak instant to record the **resident tensor
   set** (tid → bytes) when `peak_num_used_pages` updates to a new max. DAV
   already has `_dump_abort_diag_consumers` / region iteration; add a
   `_record_peak_residents()` that snapshots `_vram.space` tensors at the new-max
   tick. Expose via the sim result / a live-object read.
3. **Diff**: `modeled_alive_set − sim_resident_set` at the (approximately) same
   instant → names the over-counted tensors and which **category** dominates
   (weights vs intermediates vs in-flight). One run on sd3@11 tells us where the
   ~312 MB lives. This converts Step 2 from guessing to fixing the top item.

Validation harness already reads true peak (`overlap_harness.py`,
`peak_num_used_pages`); extend it to also dump the resident-set diff.

---

## Step 1 — λ (cheap, do alongside Step 0)

**Confirm the phantom latency.** `TransferJob` (sim/core/job/transfer_job.py)
charges `fixed_latency_micros` **only when an endpoint is a `BaseStorage`** (SSD);
RAM→VRAM H2D has neither endpoint a BaseStorage → fixed latency = 0, cost is pure
bandwidth (`4·num_pages KB / rate`). But the scheduler's
`tau_h2d_ns = h2d_latency_ns + size/bw` includes `h2d_latency_ns = 5000` (5 µs)
per tid (hw_pcie4.json). For sd3's ~thousands of channel jobs that's a large
phantom term in the channel cost `p_j` → L over-/mis-priced.

**Verify** with a 2-transfer micro-bench: issue two RAM→VRAM transfers back to
back in sim, read total time; compare to `2·size/bw` (pipelined/no-latency) vs
`2·(5µs+size/bw)`. Confirms λ.

**Fix**: set the channel processing `p_j = size/bw + λ` with λ = measured
per-transfer non-pipelined latency (expected ≈ 0 for RAM→VRAM). Apply the same to
`tau_h2d_ns` where it feeds the channel/lateness side. Keep latency only where
sim actually charges it (SSD-sourced streaming, `DAV_STREAM_FROM_SSD`). This
removes a systematic bias from L (and from `gap_feasibility`/`c_feasibility`,
which use τ).

Code: `_build_pool` (~189-403, `tau_h2d_ns`), `effective_h2d_bw`, channel `p_ms`
in §2b, and the bucketed `W_b` (uses `tau_h2d_ns/S`).

---

## Step 2 — peak fidelity (the core)

Fix the over-count sources Step 0 ranks, in order. Known/likely sources and the
fix for each:

### 2a. Intermediate-activation overlay (likely the biggest)
`_build_intermediate_residencies` (~404-492) builds activation residency from
**baseline** sim-times and adds it to `const_addons` at every sample. Risks:
(i) the chosen plan shifts timing, so the overlay is from a different timeline;
(ii) activations from different graphs/iterations summed at a sample where they
don't co-reside. Audit showed it contributing up to ~134 MB.
**Fix**: compute the overlay as the true **max co-resident** activation set (not
the cross-iter sum); recompute on the realized (plan-shifted) timeline, or
cap it at the per-graph max. Validate the overlay's max ≈ sim's true activation
residency at peak (Step 0 diff isolates this).

### 2b. "Always-alive" over-coverage
In the per-sample peak loop (~1610-1700), a tid is charged full `size` as
resident when: it's the consumer node, in `[consumer−τ, consumer)`, or in an
infeasible gap. If the sample grid bunches consumers, or τ is wider than sim's
real claim window, this counts weights as co-resident that sim has already
evicted / not yet loaded. **Fix**: tighten each region to match sim's
claim/release (claim at issue, release at evict/last-use); confirm against the
Step-0 resident-set diff that no weight is charged in a window sim has it absent.

### 2c. In-flight pool (bw·W) — calibrate
The `constant_floor += bw·W` term (Fix 2) bounds the early-arrived-waiting pool.
Step 0 says whether bw·W matches the true in-flight bytes at peak; adjust the
coefficient (it may need to be the *max concurrent in-flight* observed, not the
full bw·W) so it neither over- nor under-shoots.

### 2d. Sampling
Confirm the binding instant is actually sampled (the grid is event-aligned, so it
should be). If Step 0 shows the true peak falls between samples, add that event.

**Validation gate for Step 2**: `|modeled_peak − true_sim_peak| / cap < ~2%`
across sdxl/sd3/llama × {tight, mid, loose} caps, for **multiple distinct plans**
(vary W and target) — not one cell. Use the oracle. Only when this holds is the
peak model faithful.

---

## Step 3 — margin → 0

Once Step 2 passes, set `safety_margin_frac` default 0.05 → **0** (or ≤1% residual
pad if calibration shows a small one-sided under-count). The honest overrun-repair
(already in `_stream_cold_tensors_to_cover_overrun`) catches any real overrun and
flags `target_infeasible` rather than shipping an aborting plan — so margin→0 is
safe, not reckless. Re-validate: no false-feasible aborts on the grid.

---

## Step 4 — L fidelity (the over-streaming residual)

After peak is faithful and margin→0, re-test sd3@11. If it still over-streams vs
Belady at equal residency, the **L model under-prices streaming**: the channel
model says a plan's L is small (streaming hidden) when sim actually stalls. Test:
build a plan, compare **modeled L** to **true makespan extension** (sim makespan −
floor). If modeled L < true, the channel model is optimistic about overlap for
that plan. Likely causes and fixes:
- λ wrong (Step 1) → fix first; re-check.
- bucket release `r_b` too optimistic (uses `left_edge − W`; jobs at the right
  edge release later) → use a per-bucket release that reflects the bucket's job
  distribution, or finer buckets near binding regions.
- EDF-order assumption ≠ sim's emit/issue order → make the emit issue in the
  model's EDF order so realization matches.
**Gate**: `|modeled_L − true_extension| / makespan < ~5%` across plans. Then the
objective ranks plans the way sim does, and the optimizer stops choosing
over-streaming plans.

---

## Order & exit criteria

1. **Step 0** (instrument modeled-vs-true diff) — 1 run, names the dominant term.
2. **Step 1** (λ) — micro-bench + 1-line p_j fix.
3. **Step 2** (peak sources, top-down by Step-0 ranking) — until modeled≈true.
4. **Step 3** (margin→0) — re-validate no aborts.
5. **Step 4** (L fidelity) — until modeled-L≈true.
6. **Full grid** vs RT + sliding_window (sdxl/sd3/llama × caps × small-W sweep).

**Exit / honesty check**: if after Step 2-3 sd3/llama reach **parity** (tie RT)
and sdxl keeps its mid-regime **win**, Path B succeeded at its realistic ceiling.
A *beat* on bandwidth-bound sd3/llama is not expected — there the channel is the
bottleneck and Belady is near-optimal; don't chase it past parity.
