# Diffusers `enable_group_offload` Behavior

Notes on what `enable_group_offload` actually does, based on reading
`diffusers/hooks/group_offloading.py` and `diffusers/pipelines/pipeline_utils.py`
(diffusers installed at
`/home/kwpark/.conda/envs/huggingface/lib/python3.14/site-packages/diffusers/`).

## Entry points

- `DiffusionPipeline.enable_group_offload(...)` — `pipeline_utils.py:1374`.
  Iterates `self.components` and forwards to each component that is an
  `nn.Module` (line 1487-1492). For each, it calls
  `component.enable_group_offload(**kwargs)` if available, else
  `apply_group_offloading(module=component, **kwargs)`.
- `ModelMixin.enable_group_offload(...)` — `modeling_utils.py:520`. Thin wrapper
  around `apply_group_offloading(self, ...)`.
- `apply_group_offloading(module, ...)` — `group_offloading.py:613`. Dispatches
  to `_apply_group_offloading_block_level` or `_apply_group_offloading_leaf_level`
  based on `offload_type`.

So `pipe.enable_group_offload(...)` is **per-component** — UNet, VAE, text
encoders are each treated independently as the root `module` passed into the
block/leaf-level functions.

## `offload_type="block_level"`

Implementation: `_apply_group_offloading_block_level` at
`group_offloading.py:762`.

**Key fact**: it iterates `module.named_children()` (line 784) — **direct
children only, not recursive**.

For each direct child:

1. **Listed in `block_modules`** → recurse into it via
   `_apply_group_offloading_block_level(submodule, ...)` (line 786-794). This
   is the only way to descend into a non-`ModuleList` child.
2. **`torch.nn.ModuleList` or `torch.nn.Sequential`** → its elements are
   sliced into chunks of `num_blocks_per_group` and each chunk becomes one
   offload group (line 796-820).
3. **Anything else** → goes into a single "unmatched" group that lives
   permanently on the onload device with the top-level module (line 841-860).
   No recursion happens here — the whole subtree stays loaded as one lump.

### Implication: nested ModuleLists are invisible

Only `ModuleList`/`Sequential` at the **top level of the component** are
matched. ModuleLists nested inside other modules (e.g. inside a custom block
class) are not auto-detected and remain part of their parent's unmatched lump.

### Concrete example: SDXL-Turbo UNet (`UNet2DConditionModel`)

Direct children of `pipe.unet`:

| Direct child       | Type                       | Block-level treatment        |
|--------------------|----------------------------|------------------------------|
| `time_proj`        | `Timesteps`                | unmatched (stays loaded)     |
| `time_embedding`   | `TimestepEmbedding`        | unmatched                    |
| `add_time_proj`    | `Timesteps`                | unmatched                    |
| `add_embedding`    | `TimestepEmbedding`        | unmatched                    |
| `conv_in`          | `Conv2d`                   | unmatched                    |
| `down_blocks`      | **`ModuleList`**           | **matched** — each inner block is its own group |
| `mid_block`        | `UNetMidBlock2DCrossAttn`  | unmatched (entire mid_block stays loaded) |
| `up_blocks`        | **`ModuleList`**           | **matched**                  |
| `conv_norm_out`    | `GroupNorm`                | unmatched                    |
| `conv_act`         | `SiLU`                     | unmatched                    |
| `conv_out`         | `Conv2d`                   | unmatched                    |

Everything visible under `pipe.unet.mid_block.attentions[0].transformer_blocks`
(itself a `ModuleList`), under `…ff.net` (`ModuleList`), etc. is **not**
separately offloaded — `mid_block` is a single unmatched lump.

### Identifying matched ModuleLists from `sdxl_turbo_model_info.csv`

A row's `class_name` is `ModuleList` or `Sequential` **and** its `path` has
exactly two dot segments after the root (`pipe.<component>.<name>`). Any
deeper ModuleList is ignored by block-level offloading.

### `num_blocks_per_group` and streams

If `use_stream=True` is set, `num_blocks_per_group` is forcibly set to 1
(line 771-775) with a warning. Stream-based prefetch only makes sense one
block at a time.

## `offload_type="leaf_level"`

Implementation: `_apply_group_offloading_leaf_level` at
`group_offloading.py:863`.

This one **does** recurse: it iterates `module.named_modules()` (every
descendant, not just direct children) and creates a one-module group for each
submodule whose type is in `_GO_LC_SUPPORTED_PYTORCH_LAYERS`
(`hooks/_common.py:43`):

- `Conv1d`, `Conv2d`, `Conv3d`
- `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`
- `Linear`
- `Embedding`

Notable exclusions: `LayerNorm` and `GroupNorm` — see the inline TODO at
`_common.py:52-53` (caused issues with CogVideoX double-invocation).

Parameters/buffers that don't sit inside one of those leaf types get attached
to their closest module ancestor and offloaded with it (line 892-936). Leaf
level offers the smallest VRAM footprint but at the cost of many more
host↔device transfers per step.

### Leaf-level + streams

`use_stream=True` is fully supported (and recommended) under `leaf_level`:

- `apply_group_offloading` creates the stream unconditionally for any
  `offload_type` (line 715-722).
- The "num_blocks_per_group forced to 1" restriction at line 771-775 is
  block-level only; there is no analogous restriction in leaf-level.
- The leaf-level function passes `config.stream` into every per-leaf
  `ModuleGroup` (line 883, 930) and additionally attaches a lazy prefetching
  hook to the root module when a stream is provided (line 938-958), so
  transfer can overlap with compute.
- Per the function's own docstring (line 863-868): leaf-level has high
  host↔device sync overhead, *but* "When using devices that support streams
  to overlap data transfer and computation, this method can reduce memory
  usage without any performance degradation."

`num_blocks_per_group` is ignored under `leaf_level` (only validated/used in
the block-level path).

## How to override the default matching

`apply_group_offloading(...)` accepts `block_modules: list[str] | None`
(also exposed on `ModelMixin.enable_group_offload`). Each name in that list
must be a direct-child attribute of the root module; the function recurses
into it as if it were itself the root. This is the supported way to reach
nested ModuleLists like `mid_block.attentions[0].transformer_blocks` —
either pass `block_modules=["mid_block"]` (and rely on what's a ModuleList
inside `mid_block`'s direct children), or restructure your call to start
from a deeper module.

Note: `enable_group_offload` on `ModelMixin` accepts `block_modules`, but the
pipeline-level `DiffusionPipeline.enable_group_offload` does **not** forward
it through `group_offload_kwargs` (see `pipeline_utils.py:1476-1486`). To use
it, call `enable_group_offload` directly on the component (e.g.
`pipe.unet.enable_group_offload(..., block_modules=[...])`).

## Concrete count: SDXL-Turbo with `block_level`, `num_blocks_per_group=1`

Computed directly from the loaded pipeline (see `count_blocks.py`):

| Component                                       | Matched groups | Unmatched | Total |
|-------------------------------------------------|----------------|-----------|-------|
| `vae` (AutoencoderKL)                           | 0              | 1         | **1** |
| `text_encoder` (CLIPTextModel)                  | 0              | 1         | **1** |
| `text_encoder_2` (CLIPTextModelWithProjection)  | 0              | 1         | **1** |
| `unet.down_blocks` (ModuleList, len=3)          | 3              | —         | 3     |
| `unet.up_blocks` (ModuleList, len=3)            | 3              | —         | 3     |
| `unet` everything else (mid_block, conv_in/out, time/add embeds, norm, act) | — | 1 | 1 |
| **GRAND TOTAL**                                 |                |           | **10** |

Out of ~2717 nn.Modules (~6.94 GB), only 6 UNet down/up blocks are
offloaded with any granularity. VAE, both CLIP text encoders, and the entire
UNet `mid_block` each live as a single big unmatched lump. To split those,
use `offload_type="leaf_level"` or call
`pipe.unet.enable_group_offload(..., block_modules=["mid_block"])` directly
(pipeline-level entry doesn't forward `block_modules`).

## Runtime behavior (validated against profiling trace)

Trace source: `tmp/offload-trace/sdxl_offload_group/` — SDXL-Turbo, 4 inference
steps, `block_level`/`num_blocks_per_group=1`/`use_stream=True`/`record_stream=False`,
RTX 4090, fp16. The kineto JSON was streamed and bucketed by `cat` and stream
ID; module windows were resolved from the `nn.Module: <ClassName>_<N>` python
function annotations emitted by `with_modules=True`.

### Two CUDA streams — but matched and unmatched groups use different ones

| Stream | Role | Events | Bytes |
|--------|------|--------|-------|
| **7** (compute) | UNet kernels + **unmatched-group** transfers | 9832 kernels, 1682 HtoD-Pageable, 1680 DtoH-Pageable | 4.77 GB each direction |
| **13** (side / "prefetch") | **Matched-group** onloads only | 5760 HtoD-Pinned, **0 DtoH** | 16.4 GB |

Stream 13 has zero DtoH events in 16.4 GB of matched-group "offloading"
traffic. See next item for why.

### Matched-group offload is a pointer swap, not a transfer

`_offload_to_memory` (line 315-335), `if self.stream is not None` branch, does
**not** call `.to(cpu)`. It just rebinds `param.data = self.cpu_param_dict[param]`
back to the always-resident pinned CPU shadow:

```python
def _offload_to_memory(self):
    if self.stream is not None:
        if not self.record_stream:
            self._torch_accelerator_module.current_stream().synchronize()
        for group_module in self.modules:
            for param in group_module.parameters():
                ...
                param.data = self.cpu_param_dict[param]   # pointer swap only
```

The pinned shadow was built once in `ModuleGroup._init_cpu_param_dict` /
`_to_cpu` (line 175-199): `tensor.data.cpu().pin_memory()`. It is permanent
for the life of the pipeline — that's the CPU-side memory cost.

With `record_stream=False` (the default), every matched-group `post_forward`
also calls `current_stream().synchronize()` (line 318) — a host stall waiting
for the compute stream to drain before the rebind. With `record_stream=True`,
the host doesn't stall, at the cost of caching-allocator memory overhead.

### Unmatched group hard-codes `stream=None`, `non_blocking=False`

The unmatched group constructed at line 794-809 ignores the user's
`use_stream=True`:

```python
unmatched_group = ModuleGroup(
    modules=unmatched_modules,
    ...
    non_blocking=False,   # hard-coded
    stream=None,          # hard-coded — ignores config.stream
    record_stream=False,
    ...
)
```

Consequence: every unmatched-group onload is `cudaMemcpyAsync(Pageable →
Device)` issued on the **compute stream** (stream 7), one memcpy per
parameter and buffer. Because the source is pageable, the runtime makes it
effectively synchronous (you see `cudaStreamSynchronize` calls and
`cudaPointerGetAttributes` correlating with these). Offload is the symmetric
real DtoH-Pageable on stream 7 (line 337-344, the `else` branch).

`_apply_lazy_group_offloading_hook` is still registered on the top-level
module (line 813), but it only wires `next_group` pointers between the
**matched** groups — the unmatched lump itself remains an island that
synchronously serializes with compute.

### Per-step alignment (from `nn.Module:` annotations in the trace)

One SDXL-Turbo UNet step (520 ms wall-clock):

```
UNet2DConditionModel_0          t=[ 497.3, 1019.0] ms  (521.8 ms total)
├─ <UNet outer pre_forward>          t=[ 497.3,  581.1]   83.8 ms
├─ DownBlock2D_0                     t=[ 581.1,  592.8]   11.7 ms
├─ CrossAttnDownBlock2D_0            t=[ 592.8,  639.2]   46.4 ms
├─ CrossAttnDownBlock2D_1            t=[ 639.2,  758.9]  119.7 ms
├─ UNetMidBlock2DCrossAttn_0         t=[ 758.9,  788.1]   29.3 ms   ← unmatched, 0 hooks fire
├─ CrossAttnUpBlock2D_0              t=[ 788.1,  892.6]  104.5 ms
├─ CrossAttnUpBlock2D_1              t=[ 892.6,  919.8]   27.2 ms
├─ UpBlock2D_0                       t=[ 919.8,  924.4]    4.6 ms
└─ <UNet outer post_forward>         t=[ 924.4, 1019.0]   94.6 ms
```

Transfer events per module window:

| Window | Stream 7 HtoD | Stream 7 DtoH | Stream 13 HtoD |
|--------|---------------|---------------|----------------|
| UNet outer pre_forward (83.8 ms) | **240 ev, 802 MB** | — | 22 ev, 10 MB |
| DownBlock2D_0 (11.7 ms) | — | — | 116 ev, 115 MB |
| CrossAttnDownBlock2D_0 (46.4 ms) | — | — | 327 ev, 1058 MB |
| CrossAttnDownBlock2D_1 (119.7 ms) | — | — | 763 ev, 2688 MB |
| **UNetMidBlock2DCrossAttn_0** (29.3 ms) | — | — | **0 ev** |
| CrossAttnUpBlock2D_0 (104.5 ms) | — | — | 176 ev, 203 MB |
| CrossAttnUpBlock2D_1 (27.2 ms) | — | — | 36 ev, 21 MB |
| UpBlock2D_0 (4.6 ms) | — | — | 0 ev |
| UNet outer post_forward (94.6 ms) | — | **240 ev, 802 MB** | — |

Key observations from this alignment:

- **The unmatched-group onload fires *inside* UNet's outer `pre_forward`**
  (~84 ms, 802 MB on stream 7), serially blocking before any down-block
  starts. A small trickle of matched-block prefetch (22 events, 10 MB) is
  also issued during this window — see line 414-415 of
  `GroupOffloadingHook.pre_forward`, where the lazy hook on the UNet root
  has wired `next_group` to the first matched block.
- **Mid_block has zero hooks.** Its weights are already resident from the
  outer unmatched onload, so neither stream sees activity during its forward.
  This confirms: only direct-child `ModuleList`/`Sequential` children get
  per-element `GroupOffloadingHook`; everything else rides the outer module's
  unmatched hook.
- **Stream-13 events during block N are the prefetch of block N+1**
  (issued by `next_group.onload_()` inside block N's `pre_forward`, line
  413-415). The per-block prefetch volume scales with the *next* block's
  weight size, not the current block's.
- **UNet outer `post_forward` fires the unmatched DtoH** (240 events / 802 MB
  on stream 7) at the END of the step — not earlier. Prefetch for the next
  UNet step cannot start until this serial 95 ms drains, because the
  prefetch chain is scoped within one UNet call.

### Step wall-clock structure

```
[ ~84 ms unmatched onload | ~340 ms matched compute+prefetch | ~95 ms unmatched offload ]
   stream 7 only              stream 7 ⇆ stream 13              stream 7 only
```

The `use_stream=True` overlap window is only the middle band (~65% of the
step). The bookend ~180 ms (~34% of step latency) is purely serialized
stream-7 transfer — no overlap, never on stream 13.

### Overlap is small even within the middle band

| Quantity | Value |
|----------|-------|
| Stream 7 kernel time (all 4 UNet steps) | 87 ms |
| Stream 7 memcpy time | 721 ms |
| Stream 13 memcpy time | 655 ms |
| Stream 13 ⇆ stream-7-kernel overlap | **24.6 ms** (28% of kernel time; 3.7% of prefetch time) |

Most prefetch wall-time happens while no compute kernels are running, because
compute is much faster than the transfers it is supposed to hide.
Additionally, `_onload_from_memory` calls `self.stream.synchronize()` at its
start (line 282-283) — a host stall — so when the prefetch for block N+1 is
slower than block N's compute, the host blocks at block N+1's `pre_forward`.

### Implications for downstream modeling

1. **Matched-group eviction is free.** Model it as a pointer swap plus an
   optional `current_stream.synchronize()` (only when `record_stream=False`).
   Do *not* count it as a DtoH.
2. **The pinned shadow copies are permanent CPU memory cost** equal to total
   matched-group weight size.
3. **The unmatched lumps are first-class transfers** that run on the compute
   stream and serialize with kernels. For SDXL these are: each text encoder
   in full, the VAE in full, and UNet's `mid_block` + `conv_in/out` +
   time/add embeds + norms — all bundled into a single per-component
   unmatched group, but transferred per-parameter as individual
   `cudaMemcpyAsync(Pageable → Device)` calls.
4. **The prefetch chain restarts every UNet forward.** No cross-step overlap.

## Summary

- `block_level` matches only **direct-child** `ModuleList`/`Sequential` of the
  component. Nested ones aren't auto-detected; their parent becomes one
  loaded lump.
- `leaf_level` matches **every** leaf `Conv*`/`Linear`/`Embedding` anywhere in
  the tree.
- Use `block_modules` (component-level call only) to manually recurse into
  non-`ModuleList` direct children.
- With `use_stream=True`, `num_blocks_per_group` is forced to 1.
- **Matched and unmatched groups use different streams and different
  transfer paths.** Matched: stream 13, pinned source, pointer-swap eviction.
  Unmatched: compute stream, pageable both ways, real DtoH on eviction.
- **Matched-group offload is a `param.data` rebind, not a transfer.** The
  pinned CPU shadow is permanent.
- **Each UNet step has serialized bookends** (unmatched onload at start,
  unmatched offload at end) that cannot overlap with anything.
