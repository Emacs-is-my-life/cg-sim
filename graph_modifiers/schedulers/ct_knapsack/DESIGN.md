# ct_knapsack — parametric-knapsack residency (Path 1)

MILP-free macro residency selection. Picks the resident (cold) set by
benefit-density ranking under a VRAM budget; timing is left to the order-driven
executor (same as the MILP's shipped seed plan). Built to test the hypothesis:
*in the bandwidth-bound decode regime the order-axis MILP is overkill — a greedy
density knapsack ties it, and is monotone-by-construction so a growing-KV decode
never re-solves.*

## Model
- Item density `rho_i = uses_i` (H2D reloads avoided per period, per byte).
- Resident iff `rho_i >= lambda`; `lambda` set by the budget. Greedy by
  `(density, size)` desc. c-infeasible/oversize tensors pinned (forced_cold).
- Peak floor `= B(paced pool 130MB) + max_streamed + max_intermediate_overlay +
  extras`. `max_streamed` is the in-flight headroom term (a streamed weight is
  resident while in flight) — the orderax floor's "one inverted claim".
- `parametric_curve()` returns the nested budget→resident breakpoint table; as
  the budget shrinks items leave in density order and never return.

## Executor pairing (required for the floor to hold in sim)
`DAV_PACED_PREFETCH_MB=130` + `DAV_PF_WAIT_ON_FULL=1` — bounds realized in-flight
to B. Without it streamed prefetches pile up resident and OOM (measured: peak
pinned at cap, abort). Harness sets these before building the Simulator.

## Validation — PASSED (2026-06-20, llama8b @ 6 GiB, scripts/knapsack_harness.py)

| scheduler | makespan | H2D | realized peak | abort | solve |
|-----------|----------|-----|---------------|-------|-------|
| RT (Belady) — *unfaithful baseline* | 4.857s | 118.3GB | 6144MB (=cap) | False | — |
| **ct_knapsack (Path 1)** | **6.160s** | **151.9GB** | **5848MB ≤ cap** | **False** | **instant** |
| ct_milp_orderax (DESIGN.md, same exec config) | 6.322s | 152GB | 5791MB ≤ cap | False | ~120s MILP |

Findings:
- **Knapsack ties the MILP** (6.160 vs 6.322s; identical 152GB H2D = the faithful
  volume floor ~148.6GB+pool; both abort-free, peak ≤ cap) with **no solver**.
  Confirms the bandwidth-bound regime reduces to "pack the most resident bytes" —
  Path 1 is the right tool here.
- RT is faster (4.857s/118GB) but 118GB is **below the 148.6GB legal volume lower
  bound** (known-unfaithful: ungated uses) — not a legal comparison.
- **Parametric curve: 1596 breakpoints, nested(monotone)=True** — the structural
  property holds, so the growing-KV trajectory walks the curve without re-solving.

## Scope / next
This validates the *weights-only single-pass* pool the sim represents today
(KV-cache representation is the open gap — see KV_DECODE_GAP_ANALYSIS.md). The KV
class enters as additional items with read-frequency density; the joint
weight↔KV partition and the R2 long-context regime are the next build (see
KV_DECODE_MODEL_PLAN.md §1–2).
