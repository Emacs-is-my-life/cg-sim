# ct_milp_lateness_simtime — improvement options

Analysis from the sm0p10 sweep (`--max-peak-samples 256 --arc-queue-factor 2
--safety-margin-frac 0.10 --time-limit-s 600 --phase1-time-limit-s 60`).
Each option tagged **[verified-from-code]** (confirmed by reading
`scheduler.py` + saved `--audit` logs) or **[verify-against-sim]** (mechanism
plausible, needs a sim check before acting).

---

## 0. Headline finding — the "MILP" rarely solves; you're shipping fallbacks

From the saved audit logs, the LP **relaxation itself** times out on every
large model:

| config        | gpu_consumer | arc_start | total grid | phase1 LP | final plan |
|---------------|-------------:|----------:|-----------:|-----------|------------|
| llama8b 8g    | 8 989        | 22 124    | **22 380** | time-out  | MILP incumbent (time-limit) |
| llama8b 10g   | 8 989        | 22 124    | **22 380** | time-out  | **hard-fallback stream-everything** |
| llama8b 14g   | 8 989        | 22 124    | **22 380** | time-out  | MILP incumbent (time-limit) |
| sd3-med 14g   | 21 585       | 69 434    | **69 690** | time-out  | **rounded LP relaxation** (`fell_back`) |
| sdxl-turbo 4g | 8 660        | 22 752    | **23 008** | optimal   | phase-2 MILP time-out → warm-start |

So the sd3-med 5/8/11 and llama8b 10/12 "identical plateau plans" the sweep
flagged are **not optimal MILP plans** — they are the hard-fallback
`stream-everything` path taken *after* the solver burned the full 600 s and
failed. Fixing solve time isn't a nicety; it's what turns these from fallbacks
into real plans.

---

## 1bis. NUMERICAL CONDITIONING — the blocker that 1a uncovered [verified empirically]

After implementing 1a (arc-cap → grid 22 380 → 768 on llama8b 8g), the LP no
longer times out — but **HiGHS then returns `model_status=Unknown,
primal_status=Infeasible`** in ~8 s, even though the LP is provably always
feasible (`s_P` and `L_window_i` are unbounded slacks that absorb any
peak/lateness). With `presolve=off` it degrades to `Solve error`. HiGHS's own
diagnostic (output_flag on):

```
Coefficient ranges:  Matrix [1e+00,1e+08]  Cost [1e+00,1e+08]  RHS [1e+00,2e+10]
WARNING: Problem has some excessively large costs
WARNING: Problem has some excessively large row bounds
WARNING:   Consider scaling the objective by 1e-3 ... the bounds by 1e-5
HEkkPrimal::shiftBound LB = 1.6e+10 ... feasibility = 1.6e-07 ...
```

The simplex shifts bounds at 1.6e10 where float64 absolute precision (~1e-6)
collides with the 1e-7 feasibility tolerance → it can't separate feasible from
infeasible. **Root cause:** the model is built in raw **bytes** (sizes ~7e8,
RHS/peak ~2e10) and mixes a `PEAK_SLACK_PENALTY=1e6` cost — a ~10¹⁶ objective
dynamic range. This was previously *masked*: at 22 k samples HiGHS timed out
mid-simplex (status "Time limit reached") and fell back to stream-everything,
so the numerical failure never surfaced. The prior `milp_arc_sample_sweep` logs
already show sporadic `Solve error` on llama3b — same cause.

**Confirmed fixes (both verified on llama8b 8g):**
- **(quick) HiGHS scale options** — `user_bound_scale=-14,
  user_objective_scale=-7` (exactly what HiGHS recommends). LP → **Optimal in
  10 s**, peak 7717 MB ≤ cap. Uniform power-of-2 scaling, so the optimum is
  unchanged; the magic exponents are mildly model-dependent (the matrix range
  [1,1e8] is untouched — only cost/RHS are scaled — but that range alone was
  tolerable).
- **(principled) rescale the model to MB** — divide every byte quantity (sizes,
  const_addons, RHS, peak target, `_bytes_per_ns_late`) by 1e6 and retune
  `PEAK_SLACK_PENALTY` to preserve the `s_P ≫ ΣL ≫ ε·streaming` hierarchy.
  Fixes Matrix, Cost, AND RHS ranges uniformly, model-independent. More edit
  surface but no magic constants.

**This is now the #1 prerequisite for everything else** — without it the
warm-start path is dead: phase-1 LP can't reach Optimal, so phase-2 MILP runs
cold and (verified) returns a junk incumbent that overruns the cap by ~7.9 GB,
rescued only by `overrun_repair`. With scaling on, phase-1 reaches Optimal
(408/11835 binaries fractional) and hands phase-2 a real warm-start.

**Residual after both fixes:** phase-2 MILP still hits its time limit (408
fractional binaries = genuine B&B work) and returns the rounded-LP warm-start;
the plan still overruns modeled peak (driven up by `arc_queue_factor=2`) and
leans on `overrun_repair`, with lateness ~1.1 s. That points back to the
formulation items in §3 (arc widening, per-window lateness) and to MIP-gap /
rounding-heuristic tuning — not to sampling.

**STATUS: 1a + scaling LANDED** (both LP build paths, no env flags). Verified
out-of-the-box, `--cores 4 --time-limit-s 120 --phase1-time-limit-s 90`:

| config       | grid (before→after) | phase-1 LP        | phase-2                 | peak ≤ cap | lateness |
|--------------|---------------------|-------------------|-------------------------|-----------:|---------:|
| llama8b 8g   | 22 380 → 768        | Optimal (408 frac)| warm-start (time limit) | 7717       | 1095 ms  |
| sd3-med 14g  | 69 690 → 768        | Optimal (21 frac) | **proven incumbent**    | 13529      | 368 ms   |

sd3-med 14g **improved** vs the original sweep: rounded-LP fallback
(`fell_back=True`, 382 ms) → genuine integer solution (`fell_back=False`,
368 ms). The hardcoded scale exponents don't regress the smaller-weight model.
Next: re-run the full sweep to confirm the abort triage (§2) and re-measure
sim e2e on the now-real plans.

---

## 1. SPEED — why 256 samples still times out, and how to fix it

### 1a. The sample grid ignores `--max-peak-samples` entirely [verified-from-code]
`max_peak_samples` thins only **consumer** events. Section 6 adds one
`arc_start` sample per *consumer of every feasible tid* (scheduler.py
~1271-1280), and the down-sample guard (~1324-1332) keeps **all** `nid==-1`
arc samples and thins only the consumer set:

```
threshold = max_peak_samples*8 = 2048;  total 22380 > 2048 → guard runs
keep_arc_post = all 22124 arc samples      # never thinned
keep_consumer = 8989 → 256
final = 22124 + 256 = 22380                # matches audit "total" exactly
```

`seen_t` only dedupes arc_starts that collide on the *exact* ns; multi-iter
weights (N tokens / N steps) have distinct consumer starts, so ≈14 arc
samples per tid survive (22124 / 1595 ≈ 14). **`--max-peak-samples 256` is a
no-op against the term that dominates the row count.**

**Options (pick one or combine):**
- **Down-sample arc samples too**, the same way consumers are: keep the
  arc_starts of the largest-`size` tids (they move peak), uniformly thin the
  rest to `max_peak_samples`. Smallest, highest-leverage change.
- **Quantize/cluster arc_starts** to a coarse time grid (e.g. round to the
  nearest 1 % of timeline, or bucket by lateness-window) and keep one
  representative per bucket — adjacent arc events constrain nearly the same
  alive-set, so this loses almost no fidelity.
- **Threshold arc samples by size**: only emit an arc_start for tids with
  `size ≥ some MB`; a 2 MB tid's in-flight window doesn't change a GiB-scale
  peak. Cuts the long tail of small weights.
- Make the existing `* 8` cap apply to arc samples as well (one-line change to
  the guard).

### 1b. Model build is O(samples × tids × consumers) in pure Python [verified-from-code]
The per-sample peak loop (~1408-1490) scans `feasible_tids`, and for each does
a **linear** gap scan `for k in range(len(pt.consumers)-1)` (line ~1460) to
find which gap the sample falls in. With 22 380 samples × 1 595 tids × ~14
consumers that's ~5×10⁸ Python ops *before HiGHS even starts* — hence the
"LP build progress" prints exist at all.
- **Binary-search the gap** (`bisect` over consumer starts) instead of the
  linear scan — O(log n) per (sample, tid).
- Build the constraint matrix with vectorized numpy / batched
  `addRows` instead of per-entry `rows.append/cols.append/vals.append`.
- After 1a shrinks samples to ~512, this matters less, but it's free.

### 1c. Large-budget configs should short-circuit, not solve [verified-from-code]
When `cap ≥ all-resident peak` (every pool tid cold + intermediates + extras),
the all-cold plan trivially fits and is optimal under the ε tiebreaker. Detect
this and **emit all-cold directly** — no LP. This kills the wasted sd3-med
5/8/11 and llama8b 10/12 solves (the "large budget can be faster" ask) and
also removes their accidental fallback-to-streaming. Symmetrically, detect
"all-streamed fits trivially" to skip straight to a minimal LP.

### 1d. Warm-start across a cap sweep [verify-against-sim]
A sweep solves the same model at monotonically changing caps. Feed the
previous cap's integer solution as the `setSolution()` warm-start for the
next — tighter→looser is nearly always still feasible. Cheap, and turns
600 s cold solves into seconds.

### 1e. Give phase-1 a real chance, or skip it [verified-from-code]
Today phase-1 LP gets 60 s and *still* times out (grid too big), so phase-2
always runs cold with no warm start — the worst case. After 1a the LP will
solve in <1 s and phase-1 becomes useful again. If you can't shrink the grid,
it's strictly better to skip phase-1 and give all 600 s to phase-2.

---

## 2. TIGHT BUDGET — triage first; only some aborts are margin/formulation

Split the 5 aborts by **modeled peak vs cap** (sweep table) — they need
different fixes:

### Class A — modeled < cap but sim aborts → the LP↔sim peak gap (FIXABLE without margin)
- llama8b 8g: modeled 7703 < cap 8590; sim aborts.
- llama3b 3g: modeled 2886 < cap 3221; sim aborts.

Root cause is documented in the README "LP–sim peak gap under hybrid": for
c-infeasible tids the LP credits dead-zone savings `size·(1−e)`, but the
injector's `coverage_repair` **demotes those tids back to fully resident**
(cold-start isn't treated as a gate before the first refetch). Sim keeps the
bytes → sim peak > LP peak → overflow.

**Options:**
- **(best) Fix the injector** so cold-start residency counts as a gate within
  `[layout, first_evict_node]`. Then the hybrid savings are real and the
  margin can drop. (README already names this as the clean fix.)
- **(LP-side, sound) Conservative peak accounting for c-infeasible hybrid
  tids**: in the dead-zone branch (~1472-1486), for c-infeasible tids keep
  them `size`-alive across the gap instead of crediting `−size·e`. This makes
  modeled peak match what sim actually does (no demotion surprise) and
  eliminates Class-A aborts without any margin. Gate behind a flag; it gives
  up the hybrid peak benefit, so only the injector fix recovers both.
- **(closed loop) Sim-feedback repair**: an overrun-repair pass already exists
  (`_stream_cold_tensors_to_cover_overrun`) but it repairs against the LP's
  *own* peak estimate. Instead: solve → sim once → feed the observed peak
  overrun back as `extra_static_bytes` (or a per-region addon) → re-solve.
  Converges to a sim-feasible plan with no blanket margin. ~2 solves.

### Class B — modeled ≥ cap / structurally infeasible (margin & samples can't help)
- llama8b 6g: modeled **13434** ≫ cap 6442.
- llama3b 2g: modeled 2395 > cap 2147 (weights ≈ 4977 MB).

The pinned-resident floor is large: audit shows `forced_cold + c_feas_false_
bound_cold` = 2101 + 2952 = **5054 MB** for llama8b (the same 5054 MB seen as
`layout_cold` in the 10g fallback). Any cap near/below that floor + peak
intermediates cannot be satisfied by *any* streaming plan.

**Options:**
- **Attack the pin, not the margin.** Tids are pinned `c=1` because no async
  issuer fires `≥ τ_h2d` before `consumer[0]` (`c_feasibility`, ~354-360).
  The issuer search is already global/cross-graph, but the **first** consumers
  of the run have nothing before them. Levers: insert earlier synthetic issuer
  anchors, split the layout load across the warm-up, or allow a bounded
  synchronous initial prefetch the injector *does* gate.
- **Report infeasible honestly.** llama3b 2g (~5 GB weights in 2 GiB) is just
  infeasible — the hard-fallback's optimistic peak (`constant_floor +
  cold_now_bytes`, ~1920-1925) under-reports and hides it, so `sched peak`
  looks plausible while sim aborts. Compute a sound lower-bound peak
  (pinned floor + max intermediate overlay) and surface "cap < floor" before
  emitting.

### 2c. In-flight prefetch claim race — replace the blunt `arc_queue_factor` [verify-against-sim]
The memory note ("in-flight H2D claim race hits the cap") and the need for
`arc_queue_factor=2` both point at the same thing: with `h2d_streams=1`, queued
prefetches hold their dst VRAM from issue→land, so K simultaneous issues claim
K×dst at once. `arc_queue_factor` is a global constant multiplier on every
tid's residency arc — over-charges loose configs, under-charges bursty ones.
- **Model it directly**: add a constraint bounding concurrent in-flight H2D
  dst bytes per sample (the lateness rows already compute per-window PCIe queue
  depth — reuse it to scale the arc width *locally* instead of globally).

---

## 3. FORMULATION — caveats to audit (don't assume; some are deliberate choices)

### 3a. Lateness windows model no PCIe backlog carryover [verify-against-sim]
20 fixed equal **wall-clock** windows, each with an independent slack and a
budget = window length (~1543-1621). A real serial PCIe queue carries overflow
from window *i* into *i+1* (and shifts those deadlines). The per-window form
neither carries backlog forward nor lets early slack absorb later load — it's a
heuristic. **A cumulative / release-time LP** — at each sample T, cumulative
H2D bytes whose deadline ≤ T must be ≤ cumulative PCIe throughput by T (one
"conveyor" constraint per sample) — models the serial queue exactly and
**replaces the 20 arbitrary windows + the lateness→peak coupling** with one
principled structure. This is the standard single-machine cumulative-flow
formulation; worth prototyping.

### 3b. `N=20` and equal-width windows are hardcoded [verified-from-code]
Windows can split an iteration; a weight whose deadline lands 1 ns into a
window is fully charged there while the prior window sits empty. Align windows
to graph/iteration boundaries (or make N a CLI arg) for a more faithful budget.

### 3c. `δ_t` sums per-tid H2D *latency* across many tids in a window [verify-against-sim]
`δ_t = h2d_latency + size/bw`; the window budget sums δ over all tids whose
deadline lands there. If sim's queued H2D **pipelines** setup latency under the
prior transfer's bandwidth phase, the summed latency is phantom and the LP sees
false lateness (→ over-streams to cold, or calls tight caps infeasible). If sim
charges latency strictly sequentially, the sum is correct. **Check sim's H2D
cost model before changing** — the authors already fixed a 10× budget bug here
(comment ~1565-1570), so this side has been iterated.

### 3d. lateness→peak coupling — audit the within-window attribution [verify-against-sim]
Every peak row in window *i* adds `bw·L_window[i]` to P (~1502-1505). Semantics
("bytes that arrive late get demoted to resident, so they count toward peak")
are intended and only bite if they push P over cap (else free under the cap).
Not a double-count bug — but verify it doesn't *over*-attribute: L_i is a
single window-wide slack added to *every* sample in the window, so a localized
late burst inflates peak at samples where those bytes aren't actually resident.

### 3e. Intermediate residency overlay is stale [verify-against-sim]
`_build_intermediate_residencies` uses baseline sim_times. The chosen plan
shifts compute timing (streaming stalls move consumers), so the intermediate
overlay added to peak rows is from a *different* timeline than the plan
produces — another LP↔sim peak-gap source on tight caps.

### 3f. Minor [verified-from-code]
- `import bisect` inside the gap loop (line ~347) → hoist to module top.
- Warm-start rounding can emit `c=1,e=1` when both LP values are exactly 0.5
  (uses `>=0.5` for both), violating `c+e≤1`; harmless (HiGHS repairs the
  hint) but the README's "proof" assumes the symmetric `c+e=1` that hybrid
  lifted.
- README "Sampling" section still says "uniform 256 stride"; code does
  event-aligned sampling. Stale docs.

---

## Suggested order of attack
1. **1a (cap arc samples)** — unblocks every large-model solve; turns
   fallbacks back into real MILP plans. Highest leverage, smallest change.
2. **1c (short-circuit loose caps)** — instant wins on the plateau configs.
3. **2 triage + Class-A fix (injector gate or conservative peak)** — removes
   the fixable aborts without bumping margin.
4. **3a (cumulative-flow LP)** — the principled rewrite, once the above make
   iteration fast enough to experiment.
