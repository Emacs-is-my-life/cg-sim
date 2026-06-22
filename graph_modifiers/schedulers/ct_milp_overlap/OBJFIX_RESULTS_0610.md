# ct_milp_overlap objective/fidelity experiments — 2026-06-10

Question (from review): "a principled MILP would solve the best schedule at
each given time goal — ours doesn't (target−1GiB plans beat target plans), so
the model is flawed. Where?" Harness: `scripts/objfix_ab.py` (shared
RT(Belady) baseline per cell; reports modeled_L vs true makespan extension
and modeled vs true peak). Cells: sdxl-turbo@6 (the WIN cell), llama8b@6
(the LOSE cell). W=5ms throughout.

## Established (measured, not hypothesized)

1. **L_max is the RIGHT objective shape; L-fidelity is already good.**
   modeled_L vs true extension: sdxl 60.35 vs 60ms; llama 8050 vs 7834ms
   (~3%). The "stalls absorb" model (serial compute shifts, later deadlines
   shift with it ⇒ extension = max lateness, not sum) matches sim.
   `MILP_SUM_LATENESS=1` (per-bucket Σℓ_b) and `MILP_EPS_PHYS=1` (volume at
   physical 1/bw instead of legacy 25×) are second-order:
   - sdxl@6: control 0.227s → eps_phys 0.219 → both 0.216 (RT 0.211). Small
     consistent WIN — keep eps_phys(+sum_lat) for diffusion.
   - llama@6: all four identical (~8.1s) — objective not binding there.

2. **The binding flaw on llama tight-cap is the PEAK MODEL + missing
   runtime safety net**, chain of three:
   - modeled peak 5968 vs true 4357MB (+1.6GB phantom) → LP under-fills
     (cold 2.3GB vs RT residency 6.1GB) → 200GB streamed vs RT 118GB →
     bandwidth-bound mk 8.10s vs 4.86s (in this regime mk = floor +
     bytes/bw exactly).
   - `MILP_HONEST_INFLIGHT=1` fixes the phantom (plan: cold 4.3GB, 25GB
     less volume) but ABORTS in sim: plans packed to ~cap need the
     reactive-Belady fallback for residency-timing drift. NOT a
     fragmentation issue (best_fit doesn't help; gap genuinely absent) —
     needs `DAV_REACTIVE_EVICT=1`. With it: **8.10 → 6.30s, no abort**,
     and only ~200-500 reactive evictions (plan deviations are modest).
   - margin>0 structurally excludes Belady-class plans (RT lives at 100.0%
     of cap), so tight-cap solves want margin=0 + reactive net.

3. **g-mode seed honesty**: with margin 0 + honest peak the greedy-Belady
   seed still models +130MB over cap (categories: consumed 118 +
   boundary_forced 118 at the binding sample) and was REJECTED as
   incumbent. `MILP_GSEED_RAISE=1` lets the auto-calibrated raise through
   (bounded by the seed's own over-count, ~2%): mk 6.20s, mip_gap 0.15%.

   **Best llama8b@6 config now**: `MILP_HONEST_INFLIGHT=1 MILP_GMODE=1
   MILP_GSEED_RAISE=1 DAV_REACTIVE_EVICT=1` + `--margin 0` + best_fit:
   **6.20s vs old best 8.10s (−23%)**. Still loses to RT 4.86s.

4. **The residual gap is the FEASIBLE SET, not the solver or objective.**
   mip_gap 0.15% AND the LP relaxation bound itself ≈ 5.7s ⇒ even
   fractionally the model cannot express a ≤118GB-volume rotation
   (≈ L 4.3s). New audits (greedy seed volume + dropped-evict counter)
   show: the offline Belady walk itself = 150GB (init 10.1 + refetch
   140.3), and 1443 of its evictions (10.1GB) fall on gaps with NO e-var
   (`gap_feasibility` pruned them) and are silently dropped at encoding
   (scheduler.py `_eset` walk). RT's runtime Belady achieves 118GB ≈ the
   theoretical (16GB−cap) per-iteration rotation floor.

## Round 2 (same day): the no-reactive-eviction standard

Review pushback (correct): if the plan needs reactive eviction, the MILP is
still unfaithful — 404 unplanned evictions / 11GB is the runtime silently
repairing the plan. Acceptance test redefined: **abort-free with ZERO
unplanned evictions, realized peak ≤ cap.**

Root cause of the unfaithfulness, identified and partially fixed:

1. **W lives on the wrong axis.** Emit places issuers `max(τ,W)` ahead on
   the BASELINE axis; the honest peak bounds early residency by `bw·W`
   assuming W of REALIZED time. At 17× stretch (llama@6), 5ms baseline
   ≈ 90ms realized — the channel races ahead of stalled compute and
   delivers GBs early. Fix: `DAV_PACED_PREFETCH_MB` (new, in DAV) — the
   executor holds the prefetch FIFO while claimed/delivered-but-unconsumed
   bytes exceed the budget, enforcing the model's in-flight invariant on
   the realized axis (with a deadlock intercept). An order/byte-based
   lookahead (not ms) would fix this structurally.
2. **Single big tensors break the bw·W pool.** A 1002MB embedding claims
   its full size for its whole δ=39ms service window — `bw·W`=130MB can't
   cover it. Existing flags `MILP_CINFEAS_INFLIGHT=1` +
   `MILP_CINFEAS_SINGLE_STREAM=1` charge those arcs so the LP
   cold-resides them (Belady-like). Fixed the 1002MB abort.
3. **Seed must fit the REAL cap** — `MILP_GSEED_RAISE` is dishonest by
   construction (plan modeled at cap+130 ⇒ realized overrun with no net).
   Replaced by `MILP_GSEED_AUTOCAL=1` (new): shrink the greedy cold
   budget until the seed's own modeled peak ≤ target. Converges in 1
   iteration (6269→6139 @ 5889MB cold).

Status of the acceptance test (variant `faithful`, margin=pool/cap≈2.12%):
abort moved 286→339→362ms, realized peak 6065 vs modeled 6014 — a
**residual ~51MB unmodeled residency** remains. PROVEN NOT the early pool
(executor budget 130→90MB leaves realized peak byte-identical at 6065).
Candidates: deferred evict releases (`_d2h_pending_vram` BEING_READ
regions), activation claims, or a model under-charge at consumer samples.

**Decisive next step (do this first): resident-set diff.** Run the
faithful plan under the MCP debugger (BREAK_ON_ABORT is default-on), dump
sim's VRAM resident tensor set at the abort, diff against the model's
alive-set at the nearest peak sample (dump via MILP_PEAK_CATEGORIZE /
peak_sample_terms with the solved x). The diff names the 51MB exactly.

## Round 3: the resident-set diff names the real bug — SAMPLE THINNING

Built the diff (scripts/abort_forensics.py post-mortem classification +
DAV_PEAK_SNAPSHOT hook in DAV._record_peak_residents +
scripts/peak_diff_forensics.py tid-by-tid model-vs-sim diff at the
realized peak). Verdict chain:

1. At the realized peak, sim-resident ≈ model's OWN residency semantics
   (6306MB alive at t*; true drift — sim-resident/model-absent — is only
   121MB: one 117MB evict-lag param + dust). The executor is fine.
2. But the LP's reported peak was 5945MB: **the 256-sample thinned grid
   under-reads the model's own plan by ~360MB,** and worse, the exact
   full-event sweep of the SOLVED plan reads **8.6GB**: the solver
   EXPLOITS the thinning — evicts exactly at sampled instants, overflows
   between them by 2.6GB. The g-mode decode also SKIPS the overrun
   repair (it assumed the reactive net). Sim aborts exactly there.
3. Fixes implemented:
   - `MILP_EXACT_PEAK=1`: exact O(events) line-sweep plan-peak evaluator
     (`_exact_sweep`); drives autocal, the reported peak, and the
     overrun repair. Boundary semantics: all deltas at a timestamp apply
     before the max (handoffs don't co-reside).
   - Lazy peak-row generation (`MILP_LAZY_ROUNDS`, default 3 when exact):
     solve → exact-sweep → add violated instants as never-thinned peak
     rows → re-solve. Violation selection = worst-64 + uniform spread
     (`MILP_LAZY_K`, default 512) because the overflow is a plateau.
   - `MILP_EVAR_ALL_GAPS` must be ON for any exact-fit config: pruned
     gaps force residency the exact grid then (correctly) flags — the
     autocal seed could not fit even at cold=3.4GB without it.

Acceptance state at time of writing: lazy rounds shrink the exact
overrun (8612 → 8469 → 8381 with worst-64-only selection); the
spread-selection + allgaps run is the live candidate.

## Round 4: seed fallback + wait-on-full; the irreducible order-axis gap

- With allgaps, the autocal seed fits the exact grid in ONE iteration
  (cold 5887 @ margin 2.12%) — a known exact-feasible Belady incumbent.
- The LP cannot converge to exact-feasibility by lazy rows alone (the
  overflow plateau has thousands of binding instants; 512 rows/round
  moved 2204→2023→1717MB). **Seed fallback** landed (`MILP_SEED_FALLBACK`,
  default on): if lazy rounds exhaust with the LP plan exact-infeasible,
  ship the incumbent (B&B discipline). Result: modeled exact 6011 ≤
  target 6014, realized peak 6066 — model-vs-real gap collapsed from
  2.1GB to ~55-130MB.
- `DAV_PF_WAIT_ON_FULL` landed (DAV): a prefetch dst-claim that can't fit
  requeues at the FIFO head and retries after planned evicts fire (real
  CachingAllocator block-on-free semantics) instead of aborting.
- **Remaining blocker (~120-220MB, invariant to margin):** the abort
  happens via the deadlock-intercept — compute is blocked on the very
  tensor whose claim fails, and the planned evicts that would free space
  have triggers DOWNSTREAM of the blocked node. The seed plan is
  exact-feasible on the BASELINE TIME axis but not in REALIZED ORDER:
  claims tick on the channel clock, evict triggers on the compute clock,
  and around the binding instant they interleave differently than the
  baseline timeline says. Margin cannot fix this (measured: realized
  binding co-residency ~6050-6066 regardless of target 5837/5929/6014).

## The principled endgame (next session): ORDER-AXIS residency model

Replace baseline-time residency windows with consumer-ORDER windows:
claims/releases happen at order positions (consumer event indices), the
exact sweep + peak rows + autocal evaluate on order instants, and the
emit pins refetch issue order behind the evicts that free its space
(pairing derivable from the same order axis). Time stays ONLY in the
channel-lateness model (which is proven faithful, ~3%). This is
drift-free by construction — the executor IS order-driven — and removes
margin/pads entirely. It is a structural rewrite of the peak side;
worth a fresh folder (ct_milp_orderax/) reusing the pool builder,
bucketed channel rows, solver glue, and emit; the peak-row construction
and the seed walk move to the order axis.

Open separately: seed walk volume 149-153GB vs RT's 118GB at equal
budget (~10.3GB/iter optimal rotation vs walk's ~13.9GB/iter) — the
makespan-parity blocker once feasibility is settled. The "dropped
evicts" audit line mostly counts benign post-final-use evictions; the
volume gap is walk quality (tie-breaking/cold-set), not e-var coverage.

## Next lever (not yet implemented)

e-var coverage / gap pruning: offer refetch vars on the gaps Belady
actually uses (relax `gap_feasibility` to "issuer exists", let lateness +
injector sync_fallback price/handle tight gaps instead of pruning them),
then re-check the LP bound drops toward (118GB/bw − floor). Watch the
`dropped evicts (no e-var)` audit line — it should go to ~0. Secondary:
the 2.1GB `infeasible_forced` at the binding sample (pinned hybrids)
shrinks rotation room; revisit pin policy under relax+reactive.

## Env-flag bug found on the way

`DAV_ALLOC_BESTFIT` env is DEAD — sim only reads scheduler arg
`alloc_policy: best_fit` from the input yaml (device_aware_vanilla_async
`_find_free_page`). Several sweep scripts set the env and silently run
first-fit. `objfix_ab.py` patches the yaml; other scripts should migrate.

## Artifacts

- exp_results/0610_objfix/*.log / *.json — all runs
- exp_results/0610_objfix/sdxl_vanilla_sim_result.json — regenerated sdxl
  baseline (old one truncated; see harness comment)
- scheduler.py: MILP_EPS_PHYS / MILP_EPS_NS_PER_BYTE, MILP_SUM_LATENESS,
  MILP_GSEED_RAISE, greedy-seed volume + dropped-evict audits
