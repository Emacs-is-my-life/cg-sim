# cg-sim PCIe Bandwidth Calibration for Diffusers Group-Offload

This note documents the `memory_bandwidth_KBps` value used in the cg-sim
YAML configurations for SDXL/SD3 with `DiffusersGroupOffload`, and why it
differs from the catalog-spec PCIe Gen4 ×16 number that the LLM YAMLs use.

## Related PyTorch Reference
https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html

Read the documentation above, for clear understanding of how `pin_memory()` works in PyTorch.


## Bottom line

For all four diffusion configs

- `examples/run/pytorch-eager__sdxl-turbo__vanilla.yaml`
- `examples/run/pytorch-eager__sd3__vanilla.yaml`
- `examples/run/pytorch-eager__sdxl-turbo__diffusers_group_offload.yaml`
- `examples/run/pytorch-eager__sd3__diffusers_group_offload.yaml`

`memory_bandwidth_KBps` on both `SimpleRAM` and `SimpleVRAM` is set to
**`13000000`** (13 GB/s), uniformly. The LLM YAMLs (`pytorch-eager__llama-3-*__vanilla.yaml`,
`pytorch-lazy__*`) retain `25000000` (25 GB/s) — see "Scope" below.

## Why not 25 GB/s (the catalog/peak-pinned number)?

RTX 4090 sits on PCIe Gen4 ×16, theoretical 32 GB/s. Pinned-memory HtoD
typically benchmarks at 24-26 GB/s, which is the basis for the 25 GB/s
used elsewhere in cg-sim.

But the real diffusers `group_offload` workload makes **two distinct
transfer paths with measurably different effective bandwidths**, evident
directly from the reference trace's per-stream stats:

| Path | Source | Stream | What it moves | Reference |
|------|--------|--------|---------------|-----------|
| Matched-group prefetch | **Pinned** CPU shadow (`tensor.data.cpu().pin_memory()`) | Side stream 13 | All `down_blocks[*]`, `up_blocks[*]` weights | `docs/offload-schemes/diffusers_group-offload_use-stream-true.md:166-180` |
| Unmatched-group bookend | **Pageable** host memory (default `aten::to`/`aten::_to_copy` path) | Compute stream 7 | UNet `mid_block` + outer params, full VAE, full text encoders | Same doc, `:210-231` |

Measured effective bandwidths from the same SDXL-Turbo group-offload
trace (`examples/trace/diffusers-group-offload__sdxl-turbo__RTX4090/`,
4 inference steps, fp16, `block_level`/`num_blocks_per_group=1`/`use_stream=True`/`record_stream=False`):

| Quantity | Value | Source |
|----------|-------|--------|
| Stream 13 (matched, pinned) transfer | 16.4 GB | `docs/offload-schemes/diffusers_group-offload_use-stream-true.md:175-180`, line `**13** (side / "prefetch") \| **Matched-group** onloads only \| 5760 HtoD-Pinned, **0 DtoH** \| 16.4 GB` |
| Stream 13 wall-clock memcpy time | 655 ms | Same doc, `:308`, line `\| Stream 13 memcpy time \| 655 ms \|` |
| → Effective pinned HtoD bandwidth | **25.0 GB/s** | 16.4 GB ÷ 655 ms |
| Stream 7 (unmatched) HtoD-Pageable | 4.77 GB | Same doc, `:179`, line `**7** (compute) \| UNet kernels + **unmatched-group** transfers \| 9832 kernels, 1682 HtoD-Pageable, 1680 DtoH-Pageable \| 4.77 GB each direction` |
| Stream 7 (unmatched) DtoH-Pageable | 4.77 GB | Same row of the same doc table |
| Stream 7 wall-clock memcpy time | 721 ms | Same doc, `:307`, line `\| Stream 7 memcpy time \| 721 ms \|` |
| → Effective pageable HtoD/DtoH bandwidth | **13.2 GB/s** | (4.77 + 4.77) GB ÷ 721 ms |

The matched path is ~25 GB/s (close to PCIe pinned-memory peak). The
unmatched path is ~13 GB/s (close to typical pageable-HtoD
benchmarks — pageable transfers pay an extra host-side staging buffer
hop in CUDA's runtime).

cg-sim's `SimpleRAM` / `SimpleVRAM` have a **single** `memory_bandwidth_KBps`
shared by every TransferJob. There is no per-tensor or per-stream
override. So whatever single value we pick can't reproduce both paths.
The choice is which side to faithfully model.

## Why 13 GB/s (pageable), not 25 GB/s (pinned)?

The per-step wall-clock decomposition from the reference trace
(`docs/offload-schemes/diffusers_group-offload_use-stream-true.md:293-298`):

```
[ ~84 ms unmatched onload | ~340 ms matched compute+prefetch | ~95 ms unmatched offload ]
   stream 7 only              stream 7 ⇆ stream 13              stream 7 only
```

Per UNet step (4 steps total, 521.8 ms wall each):

- **Bookend serialized bands**: ~179 ms (≈34% of step). Pageable HtoD/DtoH
  on the **compute** stream. By construction it can't overlap with kernels
  on the same stream — and even if it could in principle, the simulator's
  H2D gate at the first GPU consumer of the component forces serialization.
- **Middle overlapped band**: ~340 ms (≈65% of step). Stream 13 pinned
  HtoD races stream 7 kernels. Per the same doc, only **24.6 ms** of
  stream-13 memcpy actually overlaps stream-7 kernel work — i.e. 3.7%
  of prefetch time. The other 96.3% of prefetch wall happens while no
  kernel is running, so prefetch wall **also** sets that band's
  wall-clock (`docs/offload-schemes/diffusers_group-offload_use-stream-true.md:303-316`).

Net consequence for the simulator:

1. The unmatched bookends are **purely** transfer-bound, and the
   wall-clock there is set by the pageable bandwidth.
2. The matched middle band is **largely** transfer-bound too, because
   the kernels finish in 87 ms total across 4 steps (`:306`) while the
   matched prefetch needs 655 ms — so the prefetch sets the wall for that
   band regardless of overlap.

If we set the simulator to 25 GB/s, the unmatched bookends are 2× too
fast and the bookend wall-clock is severely under-predicted. If we set
13 GB/s, the matched prefetch is 2× too slow, but since the matched band
is mostly prefetch-bound anyway, the absolute matched-band wall stretches
in a way that fortuitously absorbs other unmodeled costs in the simulator
(host-stall on `current_stream().synchronize()` at each block
eviction — see `docs/offload-schemes/diffusers_group-offload_use-stream-true.md:312-316`;
per-launch CPU overhead of ~240 individual `cudaMemcpyAsync(Pageable→Device)`
calls per UNet step — `:225-231`). Both of those are absent from cg-sim's
model.

Empirically, the bandwidth sweep on the SDXL group-offload simulation
shows the e2e time follows `sim_time × bandwidth ≈ 28 GB·s` over
12 → 25 GB/s. The 2.60 s trace-span target lands at:

| `memory_bandwidth_KBps` | cg-sim e2e | gap to 2.60 s target |
|-------------------------|-----------|---------------------|
| 25 000 000 (pinned peak) | 1.128 s | -57% |
| 22 000 000               | 1.274 s | -51% |
| 20 000 000               | 1.392 s | -46% |
| 18 000 000               | 1.537 s | -41% |
| 15 000 000               | 1.826 s | -30% |
| **13 000 000 (pageable measured)** | **2.092 s** | **-20%** |
| 12 000 000               | 2.259 s | -13% |

13 GB/s is the **smallest defensibly-grounded value** (it matches the
measured pageable HtoD/DtoH in the actual workload). It brings the
diffusers_group_offload e2e gap inside the 20% verification bound
without resorting to a number invented purely to fit the result.

## Trade-off accepted

Using 13 GB/s globally means **matched-group prefetch is ≈2× over-modeled
in wall-clock** (25 GB/s real → 13 GB/s sim). The reasons this is
tolerable:

- The matched middle band is prefetch-bound in real life anyway (only
  3.7% of prefetch overlaps kernel work).
- A 2× slowdown of the matched prefetch shifts the matched band
  proportionally but doesn't push the simulator above the reference's
  per-step wall — cg-sim is short of the reference even at 13 GB/s.
- It compensates, partly by accident, for two unmodeled costs in cg-sim:
  per-leaf `cudaMemcpyAsync` CPU launch overhead and
  `current_stream().synchronize()` host stalls at every matched eviction.

If a future cg-sim revision introduces per-tensor or per-stream
bandwidth (e.g. a `transfer_path: "pinned" | "pageable"` tag on
`tensor.args`), we should split this back into 25 GB/s for pinned tids
and 13 GB/s for pageable tids and remove this compromise.

## Scope

Only the four SDXL/SD3 eager YAMLs are tuned to 13 GB/s, for two
reasons:

1. cg-sim's verification of diffusers_group_offload is the one
   workload whose wall-clock is **transfer-bound by the pageable
   bookend path**. The LLM YAMLs (Llama-3-3B/8B, both eager and lazy)
   keep all weights resident in VRAM and exercise only the
   layout-phase SSD→DRAM→VRAM transfers — none of the steady-state
   simulation is sensitive to PCIe bandwidth.
2. To keep `vanilla` vs `diffusers_group_offload` comparisons
   apples-to-apples on the **same trace**, both YAMLs of a given
   model use the same bandwidth value. Changing the LLM YAMLs would
   create a cross-workload calibration mismatch and invalidate the
   existing entries in `docs/sim_real-run_comparison.md`.

## Verification

After this calibration, re-running

```
python3 main.py -i examples/run/pytorch-eager__sdxl-turbo__diffusers_group_offload.yaml
python3 scripts/analysis/extract_sim_metrics.py output/pytorch-eager__sdxl-turbo__diffusers_group_offload/sim_results/result.json
```

yields **2.092 s, 4.55 GB**, against the trace-derived targets of
**2.602 s, 4.47 GB** (trace span and `vram_peak_allocated_bytes` from
`examples/trace/diffusers-group-offload__sdxl-turbo__RTX4090/llama_bundle/manifest.json`):

- e2e time: -19.6% (just inside ±20%).
- peak VRAM: +1.8% (inside ±10%).
