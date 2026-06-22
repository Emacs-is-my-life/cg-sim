# ct_milp_orderax — order-axis residency MILP

## Why (measured, 2026-06-10; see ct_milp_overlap/OBJFIX_RESULTS_0610.md)

ct_milp_overlap models residency on the BASELINE TIME axis. Two fatal,
measured consequences at tight caps (llama8b@6):

1. The LP exploits the 256-sample thinned peak grid: solved plans satisfy
   sampled instants and overflow 2.6GB between them (exact sweep 8.6GB vs
   sampled 5.9GB). Lazy rows alone converge too slowly (plateau with
   thousands of binding instants).
2. Even an exact-on-time-axis-feasible plan (the Belady seed) aborts in
   sim: claims tick on the CHANNEL clock, evict triggers on the COMPUTE
   clock, and around the binding instant they interleave differently than
   the baseline timeline says (~120-220MB, margin-invariant — the blocked
   consumer's freeing evicts are downstream of itself).

The executor is ORDER-driven (single GPU stream, nodes retire in order;
evicts fire at consumer-node retire; gates park consumers). Hence:

## The model

- **Residency lives on the consumer-ORDER axis.** Global position p =
  index of each (tid, k) consumer event sorted by baseline start. Per
  tid: cold ⇒ alive [0, last]; streamed ⇒ alive [first, last] minus the
  interiors of evicted gaps (e_{t,k}=1 ⇒ absent in (pos_k, pos_{k+1})).
  Position windows are stretch-invariant: however much the run stalls,
  order is preserved, so model residency == executor residency by
  construction (up to the bounded in-flight pool below).
- **Time enters ONLY the channel-lateness model** (bucketed single-server
  EDF with release times, unchanged from ct_milp_overlap — proven
  faithful to ~3%). Objective = L_max + (1/bw)·streamed volume.
- **In-flight pool**: the executor runs with DAV_PACED_PREFETCH_MB=B and
  DAV_PF_WAIT_ON_FULL=1 (claims wait for planned evicts instead of
  aborting), so claimed-but-unconsumed bytes ≤ B at all times; the model
  carries B as a constant in the floor. Tensors with size > B cannot be
  bounded by the pool: they get NO e vars (no refetch) and, if
  c-infeasible, are pinned cold — exactly Belady's treatment of the
  1051MB embeddings.
- **Exact evaluation, same axis**: an O(events) position-space sweep
  evaluates any (c, e) plan exactly. It drives the greedy-Belady seed
  autocal, lazy peak-row generation (violated positions become new rows),
  and the seed FALLBACK (if the LP plan remains exact-infeasible after
  the lazy rounds, ship the known exact-feasible Belady incumbent —
  B&B discipline).

## Reuse

_build_pool / _load_baseline_sim_times / _build_intermediate_residencies
(mapped time→position by event start), _solve_two_phase_highspy,
_stream_cold_tensors_to_cover_overrun, _emit_neutral (+ env
MILP_GMODE/MILP_EVAR_ALL_GAPS/MILP_CINFEAS_INFLIGHT for the lifetime/
all-gaps/pinned-tag emit semantics) — all from ct_milp_overlap.scheduler.

## Acceptance test — PASSED (2026-06-10)

llama8b@6GiB, margin 0.005, NO DAV_REACTIVE_EVICT:
  makespan 6.322s, abort=False, realized peak 5791MB ≤ modeled 6114MB ≤
  cap 6144MB, zero unplanned evictions, modeled_L 5900ms vs true
  extension 6056ms (2.6%). H2D 152GB (RT: 118GB, 4.857s).
(exp_results/0610_objfix/llama8b6_orderax7.log; harness
 scripts/objfix_ab.py variant "orderax".)

Executor pieces this needed (all in device_aware_vanilla_async):
  - DAV_PACED_PREFETCH_MB pool with decrement at GATE-PASS (retire-based
    misses view-like in-place retires → pool leak → drain wedge),
  - DAV_PF_WAIT_ON_FULL (claim-miss waits, mid-batch budget pacing),
  - deadlock-intercept with deferred-release retry, stale-batch skip,
    and DEMAND-PRIORITY out-of-order issue of the parked front's own
    batch (head-of-line blocking: an unfittable head batch starves the
    32MB batch the front actually needs).
  - Model floor covers B + one inverted claim (sync-fallback arrivals
    enter the FIFO out of order).

## Walk quality investigation (2026-06-10/11): WALK EXONERATED, RT BASELINE INVALID

scripts/walk_lab.py measured the policy-free volume LOWER BOUND from the
trace: every iteration touches 16.07GB against a ≤6.44GB device ⇒ any
legal execution loads ≥ ~148.6GB. The belady walk sits at **149.1GB
(LB + 0.4GB — essentially optimal)**; a static-partition policy ties
(seed policies selectable via MILP_SEED_POLICY=belady|static|best).

**RT (SwapAdvisorRuntime, 118.3GB / 4.857s) BEATS the mathematical
floor ⇒ it is unfaithful.** scripts/rt_gate_audit.py counted **1278
consumer-executions with an ABSENT weight (92.5GB of ungated uses, 109
weights — incl. the 1051MB embedding used 52× while evicted).** Every
RT-baseline comparison must be restated; the legal no-prefetch optimum
at this cap is ≈ 0.27 + 148.6/25.8 ≈ **6.03s**, and the faithful
orderax plan (6.32s) is within ~5% of it.

Also found: the harness UNIT BUG — peak targets used cap_gib·1024·1e6
bytes on a cap_gib-GiB (·1024³) device, wasting 298MB of residency
(~4.6GB volume over the run). `HARNESS_GIB_TARGET=1`
(overlap_harness) plans against the true device size; orderax variant
sets it. Status: GiB-target runs reach 96% of the run (5.87s, 141GB
done) before a tail-end claim abort — the wait/intercept machinery
needs one more headroom pass at the tighter slack. The margin-0.005
legacy-target acceptance (6.322s, abort-free, zero reactive evicts)
remains the validated faithful result.
