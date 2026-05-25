# `hf_accelerate` — HF / diffusers CPU-offload schedulers

One scheduler per HF offload API. Each is a thin CLI wrapper that
hardcodes the semantics that the corresponding HF function actually
exhibits (granularity, lookahead, sync behaviour, whether bytes move
on offload). There is **no `--mode` switch** and **no functional flags**
like `--d2h-evict` / `--sync-calls` / `--lookahead` — those are
properties of the HF API the scheduler models, not user knobs.

| Module | HF API | Granularity | Lookahead | D2H on offload? | Last unit resident? |
|---|---|---|---|---|---|
| `sequential` | `pipe.enable_sequential_cpu_offload()` | leaf nn.Module | 0 (sync) | no (cpu copy preserved by AlignDevicesHook) | no |
| `module` | `accelerate.cpu_offload(model)` | leaf nn.Module | 0 (sync) | no | no |
| `model` | `pipe.enable_model_cpu_offload()` | top-level component | 1 (prev_module_hook) | yes (per-param `module.to("cpu")`) | no (`maybe_free_model_hooks` D2Hs last too) |
| `module_hook` | chained `accelerate.cpu_offload_with_hook(..., prev_module_hook=)` | top-level component | 1 (prev_module_hook) | yes | yes (no successor hook for last component) |
| `group` | `diffusers.apply_group_offloading(block_level, use_stream=True, non_blocking=True)` | ModuleList/Sequential block | 1 (separate stream) | no (`use_stream=True` offload is a pointer swap) | no |

`sequential` and `module` are functionally identical except for
`offload_buffers` (sequential = True, module = False — accelerate's
defaults).

## Inputs

All schedulers operate on an **eager-mode** PyTorch profile bundle —
one captured without `torch.compile`. The bundle must include
per-node module annotations (`module_path`, `module_class`,
`module_size_bytes`, `module_has_parameters`) which the cg-sim
pytorch loader exposes when the bundle was built with
`with_modules=True` on torch profiler. No compile sidecars
(`compiled_launch_map_graph*.json`) are required.

The `group` scheduler additionally needs a `module_hierarchy.json`
describing the model's nn.Module tree, so it can locate the real
`ModuleList` / `Sequential` containers that define block boundaries.
It is auto-discovered from `<bundle>/module_hierarchy.json` or
`<bundle>/../<bundle>_module_hierarchy/module_hierarchy.json`;
override with `--module-hierarchy PATH`.

## CLI

```bash
# enable_sequential_cpu_offload-equivalent
python3 -m graph_modifiers.schedulers.hf_accelerate.sequential \
  /path/to/eager/bundle/llama_bundle \
  --output /path/to/sequential_schedule.json

# accelerate.cpu_offload(model) (per-leaf, no D2H, buffers stay)
python3 -m graph_modifiers.schedulers.hf_accelerate.module \
  /path/to/eager/bundle/llama_bundle \
  --output /path/to/module_schedule.json

# enable_model_cpu_offload-equivalent (depth:1, D2H on offload)
python3 -m graph_modifiers.schedulers.hf_accelerate.model \
  /path/to/eager/bundle/llama_bundle \
  --output /path/to/model_schedule.json

# Manual cpu_offload_with_hook chain (last component stays resident)
python3 -m graph_modifiers.schedulers.hf_accelerate.module_hook \
  /path/to/eager/bundle/llama_bundle \
  --output /path/to/module_hook_schedule.json

# apply_group_offloading(block_level)
python3 -m graph_modifiers.schedulers.hf_accelerate.group \
  /path/to/eager/bundle/llama_bundle \
  --num-blocks-per-group 1 \
  --output /path/to/group_schedule.json
```

The shared flags are:

| flag | meaning |
|---|---|
| `--output` / `-o` | Schedule output path. Default: `<bundle>/hf_accelerate_output/schedule.json`. |
| `--keep ...` | Comma-separated module-path substrings to keep cuda-resident (cold-start). Not part of real HF — user-side experimental override. |

`group` adds:

| flag | default | meaning |
|---|---|---|
| `--num-blocks-per-group N` | `1` | Coalesce N consecutive ModuleList/Sequential entries into one offload group (real `apply_group_offloading` argument). |
| `--module-hierarchy PATH` | _auto-discover_ | Path to `module_hierarchy.json`. Required to identify block boundaries. |

Each CLI writes a self-contained `schedule.json` containing
`xfer_arrivals`, `evict_after_node`, `streamed_tids`, and module
metadata.

## Picking a scheduler when the harness reinterprets `--mode`

`scripts/sim_sweep_script.py` maps `(harness, --mode)` → scheduler
because some harnesses use different HF APIs for the same `--mode`
label:

- SDXL / SD3 (diffusers pipelines): `--mode {sequential, module,
  model, module-hook, group}` map 1:1 to the schedulers above.
- llama (`run_llama_accelerate_cpu_offload.py`): `--mode {model,
  module, sequential}` all call `accelerate.cpu_offload` underneath
  (no `enable_model_cpu_offload` exists for plain HF text models), so
  they all map to the `module` scheduler. `--mode module-hook`
  uses the chained-`cpu_offload_with_hook` path; `--mode group` uses
  `apply_group_offloading`.

## Applying the schedule to a sim

Point a YAML's trace loader at the schedule via
`inject_eager_schedule_path` and use the async scheduler:

```yaml
trace:
  type: "PytorchProfile"
  args:
    profile_dir: "/path/to/bundle"
    bundle_manifest: "llama_bundle/manifest.json"
    inject_eager_schedule_path: "/path/to/schedule.json"

scheduler:
  type: "DeviceAwareVanillaAsync"
  args:
    cpu_compute: "cpu"
    cuda_compute: "gpu0"
    cuda_device: "cuda:0"
```

`DeviceAwareVanillaAsync` reads `trace.args["xfer_arrivals"]` to fire
async RAM→VRAM transfers on issuer-retire and gates consumer nodes on
the resulting `_xfer_state` transitions. `trace.args["evict_after_node"]`
frees VRAM mirrors at each run's last consumer.

## How ownership and runs are determined

A **unit** is a parameter-carrying module (or the depth-N / block
group it falls into). A **weight tensor** is attributed to a unit by
inspecting its GPU consumers' `module_path` tags:

1. If exactly one consumer-path resolves to a known unit, the tensor
   belongs to that unit.
2. If multiple disjoint units consume it, the tensor is marked
   **shared** and stays cuda-resident (streaming it for any one unit
   would drag that unit's window across the trace).
3. If the consumers all live in a parent module with a single
   param-carrying child, the tensor belongs to that unique child.
4. Otherwise the tensor is ambiguous → it is streamed as its own
   per-tid pseudo-unit (mirrors real HF, which moves every parameter
   regardless of profiler attribution).

### Per-tensor (not per-module) run scheduling

Scheduling happens at the **weight-tensor** granularity, not at the
module-hook granularity accelerate uses. Each tid's GPU consumer
events are bucketed into **runs** by walking the CPU dispatch stream
and detecting `module_path` transitions (the actual `nn.Module`
forward boundaries that the profiler records when `with_modules=True`).
Each run gets its own prefetch + evict pair, fired/evicted at the
tensor's own first/last consumer node in that run. Prefetches with
the same `(issuer, consumer)` pair are batched into a single arrival
entry, so co-firing modules get a single transfer event.

This diverges from accelerate's "move every param of the module in
one hook fire" semantics on traces where torch profiler
mis-attributes fused kernels to parent modules (e.g. a CUTLASS matmul
kernel reading a sibling Conv2d weight ends up tagged as `norm1`).
Per-tid attribution keeps each weight with its actual run instead of
prefetching it on every parent-tagged hook fire.

## Cross-mode comparison vs real HF (SDXL-Turbo)

Source bundle (sim input): `sdxl-turbo-modulecols-eager` — vanilla
eager run with no offload applied. Real numbers come from running
`scripts/run_accelerate_cpu_offload.py sdxl-turbo --offload-mode X
--fusion none --steps 1 --height 128 --width 128 --warmup-runs 1`,
which prints `inference_seconds` (non-profiled wall) and
`max_memory_allocated` (non-profiled VRAM peak).

The numbers below were taken with the older single-CLI scheduler at
`--lookahead 1` for every mode. After the per-mode split, `sequential`
and `module` schedulers run at `lookahead=0` (matching real
AlignDevicesHook semantics) and will produce different sim numbers
than what's tabulated below — peak should drop slightly, wall should
grow.

| Mode | Real wall (inference) | Sim wall (old, lookahead=1 for all) | Real VRAM peak | Sim VRAM peak |
|---|---|---|---|---|
| `eager` baseline (no offload) | 49 ms | 256 ms | 6846 MB | 7273 MB (+6%) |
| `sequential` | 974 ms | 490 ms | 129 MB | 505 MB |
| `module` | 946 ms | 490 ms | 129 MB | 505 MB |
| `model` | 1422 ms | 480 ms | 4963 MB | 5286 MB (+6%) |
| `module-hook` | 446 ms | 480 ms | 6238 MB | 5286 MB (-15%) |

### VRAM peak — the scheduler's job

For the modes that bundle weights at the *component* level (`model` /
`module_hook`), sim VRAM peak matches real within ±15%: the
prefetch+evict of a whole top-level component (UNet ≈ 5 GB) dominates,
and the ~400 MB activation footprint is a small relative add.

For the per-leaf modes (`sequential` / `module`), the scheduler
correctly bounds *weight* residency at peak to ~111 MB. Real HF
measures 129 MB peak. The 300 MB overshoot in sim total is **cg-sim's
intermediate-tensor accounting**, not a scheduler bug — real
PyTorch's caching allocator reuses freed activation pages immediately
while cg-sim tracks them as distinct regions.

### Wall time — limits of the comparison

Sim wall is in the right order of magnitude across all modes
(~250–500 ms) but **does not track the mode-dependent overhead profile
of real HF**:

- Sim is ~5× *slower* than real for the eager baseline because
  cg-sim sums CPU dispatch durations sequentially while real PyTorch
  overlaps them with GPU work via the cuLaunch queue.
- Sim is ~2–3× *faster* than real for `sequential` / `model` because
  cg-sim doesn't model the Python hook overhead (~400 µs per
  `set_module_tensor_to_device` × 2300+ params for sequential, ~600 µs
  per `enable_model_cpu_offload` bookkeeping op for model).
- Sim happens to land close to real for `module-hook` (480 vs
  446 ms): the chained variant only fires 4 hooks per inference, so
  Python overhead is small and H2D-bound wall dominates.

These gaps mean sim wall is **directionally useful for comparing
schedules against each other** but not for predicting absolute
real-time on hardware. Closing the gap would require modeling
per-launch CPU/Python overhead per mode in cg-sim — a separate
project.


## Known caveats

- **Ambiguous weight tensors** are streamed as standalone tids (each
  becomes its own pseudo-unit, prefetched on first use and evicted on
  last use). This mirrors real HF, which moves every parameter
  regardless of profiler attribution.
- **Activation-aware buffering** is not modeled — VRAM peak is
  determined by which weights are concurrently resident across the
  lookahead window plus the activations that the sim's
  DeviceAwareVanillaAsync already accounts for.
- **Wall-time gap vs real HF**: cg-sim does not model per-hook Python
  overhead. Real HF spends a non-trivial fraction of wall time in
  Python hook dispatch (one hook fire per nn.Module per forward call).
  Use sim wall for *relative* schedule comparisons, not absolute
  predictions.
- **Sync mode (`sequential` / `module`, lookahead=0) with ambiguous
  streaming can deadlock** on some traces — the per-tid synchronous
  issuers create fan-out that the dependency graph can't satisfy.
  If you hit this, fall back to the `module_hook` or `model`
  scheduler (lookahead=1) for that bundle.
