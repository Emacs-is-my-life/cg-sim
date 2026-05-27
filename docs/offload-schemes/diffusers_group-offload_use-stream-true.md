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

## Summary

- `block_level` matches only **direct-child** `ModuleList`/`Sequential` of the
  component. Nested ones aren't auto-detected; their parent becomes one
  loaded lump.
- `leaf_level` matches **every** leaf `Conv*`/`Linear`/`Embedding` anywhere in
  the tree.
- Use `block_modules` (component-level call only) to manually recurse into
  non-`ModuleList` direct children.
- With `use_stream=True`, `num_blocks_per_group` is forced to 1.
