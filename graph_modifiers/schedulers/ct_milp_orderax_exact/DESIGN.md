# ct_milp_orderax_exact — exact-peak, no-seed measurement variant

A fork of `ct_milp_orderax` with two deliberate changes, both aimed at
**measuring the pure MILP solution** — no caching heuristic propping it
up, and no sampling approximation to repair after the fact.

## Change 1 — exact incremental residency recurrence (replaces sampled peak)

`ct_milp_orderax` enforced the peak at ≤512 *sampled* order positions.
Because residency between samples is unconstrained, the LP routinely
produced plans that satisfied every sample yet overflowed in the gaps
(measured 2.6GB on llama8b@6). That forced the lazy-row rounds, the
exact-sweep violation feedback, and ultimately the Belady seed fallback.

On the consumer-order axis residency is **piecewise-constant with
breakpoints only at consumer events**, so we can enforce the peak at
*every* breakpoint and it is genuinely exact — there is no "between
samples". The trick is to keep it sparse via a running-level recurrence
instead of re-summing the alive set at each position:

```
R_q ≥ R_{q-1} + Δ_q       ∀ breakpoints q     (R_{-1} ≡ constant_floor)
P   ≥ R_q                 ∀ q
```

Each tensor (positions `ps`, size `s`) contributes O(1)+O(#evictable
gaps) signed deltas:

| component                  | interval            | deltas                         |
|----------------------------|---------------------|--------------------------------|
| streamed-lifetime base     | `[ps[0], ps[-1]+1)` | `+s @ ps[0]`, `-s @ ps[-1]+1`  (constant) |
| cold pre-use               | `[0, ps[0])`        | `+s·c @ 0`, `-s·c @ ps[0]`     |
| cold post-use¹             | `[ps[-1]+1, M+1)`   | `+s·c @ ps[-1]+1`, `-s·c @ M+1`|
| evicted gap `k`            | `[ps[k]+1, ps[k+1])`| `-s·e @ ps[k]+1`, `+s·e @ ps[k+1]` |

¹ only when `MILP_COLD_NO_RELEASE=1`.

So the whole chain carries **O(M + n_e) nonzeros** — the same class as
the ~25k binary e-vars that already dominate the B&B. We add cheap
continuous vars/rows, not hardness. Because the deltas are signed and
`R_q` is driven down by the downward pressure on `P` (the s_P penalty),
the chain binds to equality at the optimum, and by induction
`R_q ≥ constant_floor + true_cumulative(q)` in *every* feasible point —
so `P` is an honest upper bound on the realized peak, no sampling.

The decomposition was verified delta-for-delta against the order-space
`_exact_sweep` over 20000 random `(c,e)` assignments (both
`cold_no_release` modes, un-evictable gaps, intermediates, c⇒¬e
coupling): zero mismatches.
`scripts/_test_orderax_exact_recurrence.py`.

Consequences: **no lazy rounds, no violated-position feedback, no
exact-sweep repair loop** — a single solve. `_exact_sweep`/`_exact_peak`
survive only to *report* the realized peak of the final plan.

## Change 2 — no Belady seed, no seed fallback (pure MILP)

`ct_milp_orderax` warm-starts the MILP with a greedy-Belady incumbent
and, when the LP plan is exact-infeasible, *ships that incumbent*. The
`0610_objfix` logs show the fallback firing across nearly every cell —
the shipped plan was usually Belady, not the solver's.

Here `_solve_two_phase_highspy(... feasible_fallback=None)`: the only
warm start is the intrinsic phase-1 LP relaxation rounding. Whatever the
MILP returns is emitted. If the solve fails entirely, we emit the
trivial stream-everything plan (`c=forced_cold only`, no evicts) and
report it honestly — there is no caching heuristic to hide behind.

`MILP_SEED_POLICY`, `MILP_NO_SEED`, `MILP_SEED_FALLBACK`,
`MILP_LP_PEAK_BUFFER_MB`, `MILP_LAZY_ROUNDS` are all **inert** here.

## Unchanged (inherited from ct_milp_overlap)

Pool build, c/e classes (`forced_cold` for the >B / c-infeasible class),
the c⇒¬e coupling rows, the bucketed single-server EDF channel model
(`L_max` objective + 1/bw·streamed-volume), the intermediates overlay,
the paced-pool `constant_floor`, and `_emit_neutral`. Pairs with the
same executor knobs (`DAV_PACED_PREFETCH_MB`, `DAV_PF_WAIT_ON_FULL`).

## How to run

```
python3 scripts/objfix_ab.py llama8b 6 --variants orderax_exact,orderax \
    --margin 0.005 --tl 150 --p1 45 --out exp_results/orderax_exact_llama8b6.json
```

`orderax_exact` vs `orderax` isolates: (a) exact peak vs sampled+lazy,
(b) pure MILP vs Belady-seeded. The audit line prints
`M / coords / vars (c,e,R) / rows / floor / target / NO-SEED`.
