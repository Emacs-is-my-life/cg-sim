# ct_milp_peak_hybrid

Pool-first MILP weight-streaming scheduler that **drops the `c + e = 1`
coupling** of `ct_milp_peak`. A tensor can simultaneously be cold at
layout (`c = 1`) *and* evict mid-run with refetch (`e_{t,k} = 1`),
enabling four residency patterns instead of two:

| `c` | `e_{t,k}` | Pattern |
|---|---|---|
| 1 | 0 | cold from layout, never evicted (resident the whole run) |
| 1 | 1 | **hybrid**: cold at layout, evict mid-run, refetch before next use |
| 0 | 0 | JIT prefetch, stays resident across this gap |
| 0 | 1 | JIT prefetch + per-iter evict-refetch cycle |

The win: tids with long dead zones between sparse uses (e.g. small
tensors consumed at iteration boundaries, weights shared by widely-
separated launches) can reclaim VRAM in the middle even when c-
infeasibility pins them to cold-at-layout. The non-hybrid variant
keeps them resident the entire timeline.

## ⚠ Injector compatibility caveat

The `inject_schedule` pass' `coverage_repair` is all-or-nothing: once
any prefetch fires for a tid, every gpu consumer is expected to be
gated by an async arrival. **Cold-start residency does NOT count as a
gate.** So consumers BEFORE a mid-run evict get demoted back to cold
by the injector, which inflates sim VRAM beyond the LP's plan (silent-
patch overhead).

Until the injector recognizes cold-start residency as a gate for the
`[layout, first_evict_node]` range, **hybrid plans may sim with peak >
LP-modeled peak**. This variant emits the plan unconditionally —
**verify in sim** before treating as safe.

## Differences from `ct_milp_peak`

Three changes in `_solve_milp`:

1. **Coupling rows dropped.** `c + e_{t,k} = 1` rows are no longer
   emitted. `c` and every `e_{t,k}` are free binaries.
2. **Dead-zone peak coefficient updated.** For samples in the dead-
   zone of feasible gap `k_in`, the contribution becomes
   `size · (1 − e_{t,k_in})` instead of `size · c_t`. Same value when
   `c + e = 1` holds, but now correctly tracks residency under any
   `(c, e)` combination.
3. **Cold-floor cut removed.** The cut `Σ size_t · c_t + const ≤ P`
   over-constrains in hybrid mode (cold tids can be evicted, so they
   don't contribute `size · 1` at every sample). The per-sample peak
   rows still bind correctly without it.

Everything else — pool building, cum-time feasibility filters,
cumulative-by-G_cum H_used aux variables, makespan rows, emit logic
— is identical to `ct_milp_peak`. The emit code already supports
hybrid (its per-gap evict+refetch loop is independent of `c_t`); it
just never fires in `ct_milp_peak` because the coupling forbids it.

## When to use

- You suspect your workload has tensors with **long dead zones between
  sparse uses** — diffusion U-Net weights touched only by cross-iter
  attention slices, LLM weights used at decode start and end, etc.
- You want strict zero LP-stall (same as `ct_milp_peak`) but want the
  LP to consider hybrid plans during optimization.

For workloads where every tid has dense, regularly-spaced consumers
(no long gaps), hybrid won't help much — there's nothing to evict
into. On sdxl-turbo, peak improves by only ~70 MB sim (8520 → 8452);
on workloads with sparser usage patterns the win could be much
larger.

## Caveats inherited from `ct_milp_peak`

- Peak rows sample ~256 of typically ~10k gpu events. Sim peak may
  exceed `milp_peak_mb`.
- The cumulative-by-G_cum PCIe constraint assumes EDF queue order; sim
  uses water-fill fair-share. Optimistic in dense-streaming regimes.
  See `ct_milp_peak`'s README for the full discussion.
- The `--makespan-target-s` flag still works; M's lower bound is
  `G_total` so the cap can only constrain `H_total`.

## API and CLI

Same surface as `ct_milp_peak`:

```python
from graph_modifiers.schedulers.ct_milp_peak_hybrid import solve_neutral
neutral = solve_neutral(trace, hw=hw, makespan_target_s=None, ...)
```

```
python -m graph_modifiers.schedulers.ct_milp_peak_hybrid.main BUNDLE \
  --hw HW.yaml [--makespan-target-s SEC] [--audit]
```

Output (`neutral.meta`) has the same keys plus `io_model = ct_milp_peak_hybrid`.
