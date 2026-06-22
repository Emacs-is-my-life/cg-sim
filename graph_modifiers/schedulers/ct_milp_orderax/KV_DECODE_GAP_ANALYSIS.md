# KV-cached decode: investigation findings & gap analysis

_Drafted 2026-06-19. Investigation only — no implementation plan yet. Establishes
what the current stack can/can't represent for the chosen research target._

## Research target (decided)
- **Regime: true KV-cached autoregressive decode** — prefill once, then
  single-token decode steps attending to a **growing KV cache**.
- **KV writeback/prefetch co-scheduling is a core contribution** (KV offloaded to
  host and streamed back, sharing the PCIe channel + VRAM budget with weights).

## Finding 1 — the trace is single-pass, not unrolled (premise correction)
The original worry ("100k decode steps → giant MILP → long solve") does **not**
arise from the current representation:
- The LLaMA bundle captures **one forward pass** (`manifest.json`: single
  `step_0_compute_graph.dot`; `runtime_nodes.csv` `step`≡0 over 11,131 nodes).
- **79% of weights have exactly 1 consumer** (`pytorch_runtime_tensors.csv`) — the
  order axis `M` does not scale with token count.
- `greedy_generate()` (`profile_llama_common.py:597`) replays the pass per token in
  Python, outside the trace.

⇒ Weight scheduling is already solved on one period and replayed. MILP variable
count is **not** the bottleneck. The scalability concern is real only for the KV
class once it is modeled (below).

## Finding 2 — there is no KV cache in the system today
- `greedy_generate` calls `model(sequences)` with **no cache** → full recompute
  every token at fixed `seq_len=77`. K/V exist only as transient `CONTEXT`
  intermediates at fixed shape, never persisted, never grown.

⇒ The non-stationary (growing-budget) regime is **inexpressible** right now. The
first research work is *representation*, not solver scaling.

## Gap stack for the target (all 5 layers MISSING or PARTIAL)

| # | Layer | Status | Evidence |
|---|-------|--------|----------|
| 1 | Profiler / trace capture | MISSING | `profile_llama_common.py:597-611` full-recompute loop, no `use_cache`/`past_key_values`; cannot capture a seq_len=1 decode step. |
| 2 | Sim tensor model (growth) | MISSING | `sim/core/trace/tensor.py:22` `size_bytes` fixed at construction; no append/grow-over-iteration concept. |
| 3 | Multi-graph semantics | PARTIAL | `multigraph_timeline.py:109-120` `graph_multiplicity` *replicates identical* graphs; no asymmetric "prefill ×1 + decode ×N" (distinct graphs) pattern. |
| 4 | Real runtime KV offload | MISSING | `WEIGHT_STREAMING_OFFLOADING.md:91-92` "only WEIGHT tensors are streamed"; `weight_streaming_ops.cpp:320` "activations must not be pin-cloned". No intermediate writeback/prefetch primitive. |
| 5 | Scheduler KV handling | STATIC FLOOR | `ct_milp_overlap/scheduler.py:393-408` intermediates enter the peak as a fixed `const_addons` overlay, **not** decision variables; pool types = WEIGHT/LEAF/INPUT only. |

## What the gaps imply (sequencing observations, not a plan)
- Layers form a hard dependency chain: you cannot *schedule* KV (5) until the
  scheduler can treat it as a variable, which needs the trace to *represent* a
  growing KV tensor (2) captured from a real KV-cached decode (1); and validating
  on hardware needs runtime offload primitives for intermediates (4).
- The weight side is largely done/stationary; the **novel surface is entirely the
  KV class**: capture it, represent its growth, model it as a second scheduled
  class on a shared channel, and give the runtime an intermediate-offload op.
- This is a **full-stack research build**, materially larger than the
  "faithful-transfer validation" framing in `REAL_SYSTEM_ROADMAP.md` — that
  roadmap remains valid for the stationary (diffusion / recompute) regime and as
  the de-risking first milestone.

## Open questions to resolve before planning
- Decode-step graph vs prefill graph: two compiled graphs (multigraph) or one
  parameterized by a dynamic shape? (affects layers 1–3)
- KV growth model: per-iteration size snapshots vs a `base + rate·iter` formula in
  the tensor model? (layer 2)
- KV access structure to exploit: append-only writes + triangular reads (token i's
  block read by all later steps) — analytic residency vs per-block variables.
- Is the budget trajectory (Cap − KV(t)) handled by parametric/phase decomposition
  over the *decode* graph only (weights stationary, KV+budget the only movers)?
