# `accelerate.cpu_offload(..., offload_buffers=True)`

All file/line refs are against `accelerate==1.13.0`.

## TL;DR

`offload_buffers=True` extends the offload paging loop to cover **persistent
buffers** in addition to parameters. **Non-persistent buffers are never paged**
regardless of this flag — they are moved to the execution device once at hook
attach time and stay there.

For each module that owns direct tensors, the behavior table is:

| element type            | `offload_buffers=False` (default) | `offload_buffers=True`                |
|-------------------------|-----------------------------------|---------------------------------------|
| direct params           | paged every forward               | paged every forward (unchanged)       |
| persistent buffers      | resident on `execution_device`    | paged every forward                   |
| non-persistent buffers  | resident on `execution_device`    | resident on `execution_device`        |

The set of modules that receive an `AlignDevicesHook` is identical in both
modes: any module whose `named_module_tensors(...)` (params + buffers,
non-recursive) is non-empty (`hooks.py:520`). For Llama-3-8B that's 292
hooks (291 param-owning leaves + 1 `LlamaRotaryEmbedding`).

## Mechanism

Three sites in `AlignDevicesHook` (`accelerate/hooks.py`) read
`self.offload_buffers`. The fourth control point is what populates
`weights_map`.

### 1. `weights_map` source — `big_modeling.py:209`

```python
state_dict = {n: p.to("cpu") for n, p in model.state_dict().items()}
```

`model.state_dict()` includes **persistent** buffers but excludes
non-persistent ones. So the CPU master copy already contains persistent
buffers regardless of the flag — the flag only controls whether they get
*used* by the per-leaf hooks.

### 2. `init_hook` — `hooks.py:298–343`

```python
# (a) set tensors to meta:
for name, _ in named_module_tensors(
    module, include_buffers=self.offload_buffers,
    recurse=self.place_submodules, remove_non_persistent=True
):
    set_module_tensor_to_device(module, name, "meta")

# (b) place buffers on execution device:
if not self.offload_buffers and self.execution_device is not None:
    for name, _ in module.named_buffers(recurse=self.place_submodules):
        set_module_tensor_to_device(module, name, self.execution_device, ...)
elif self.offload_buffers and self.execution_device is not None:
    for name in get_non_persistent_buffers(module, recurse=self.place_submodules):
        set_module_tensor_to_device(module, name, self.execution_device, ...)
```

With `offload_buffers=True`:
- (a) sets **params + persistent buffers** to `meta` (non-persistent ones are
  explicitly excluded by `remove_non_persistent=True`).
- (b) explicitly moves the non-persistent buffers to `execution_device`, so
  they have real storage and are immediately usable by every forward.

With `offload_buffers=False`:
- (a) sets only **params** to `meta`.
- (b) moves **all** buffers (persistent + non-persistent) to
  `execution_device`. Persistent ones live there for the lifetime of the
  model — paging never touches them.

### 3. `pre_forward` — `hooks.py:346–389`

```python
for name, _ in named_module_tensors(
    module, include_buffers=self.offload_buffers,
    recurse=self.place_submodules, remove_non_persistent=True,
):
    value = self.weights_map[name]                       # CPU master copy
    set_module_tensor_to_device(module, name,
                                self.execution_device, value=value, ...)
```

With `offload_buffers=True` the iteration yields params + persistent
buffers — they get the same just-in-time H2D copy as parameters. With
`=False`, only params.

The args/kwargs alignment (`send_to_device(args, execution_device)`) runs
either way.

### 4. `post_forward` — `hooks.py:392–417`

```python
for name, _ in named_module_tensors(
    module, include_buffers=self.offload_buffers,
    recurse=self.place_submodules, remove_non_persistent=True,
):
    set_module_tensor_to_device(module, name, "meta")
```

Same iterator: persistent buffers are dropped back to `meta` alongside the
params with `=True`; with `=False` they are untouched.

## Per-forward timeline (with `offload_buffers=True`)

For each module that owns direct tensors, in execution order:

1. Module's `forward` is called → PyTorch dispatches through accelerate's
   wrapped forward (`hooks.py:186–193`).
2. `pre_forward` runs: H2D copy of every param + persistent buffer, args
   moved to `execution_device`. Synchronous, `non_blocking=False`, default
   stream (verified at `utils/modeling.py:343`).
3. Original `forward` runs on GPU.
4. `post_forward` runs: every param + persistent buffer is set to `meta`,
   freeing the GPU storage at the next caching-allocator op.

Non-persistent buffers are skipped at every step — they sit on
`execution_device` permanently, paid for once at attach time.

## Effect on Llama-3-8B

Llama-3-8B contains **two buffers**, both **non-persistent** (verified by
`check_buffers.py`):

```
model.rotary_emb.inv_freq          (non-persistent)
model.rotary_emb.original_inv_freq (non-persistent)
```

Source: `transformers/models/llama/modeling_llama.py:89-90` and
`transformers/modeling_rope_utils.py:73,79,109,116` — all use
`register_buffer(..., persistent=False)`.

Consequences for `cpu_offload(model, offload_buffers=True)`:

1. **Hook count: still 292.** Identical to `=False`.
2. **What changes vs `=False`:** nothing observable in steady state. The two
   non-persistent `inv_freq` tensors are placed on `cuda:0` in `init_hook`
   under either branch, and neither pre_forward nor post_forward touches
   them.
3. **PCIe traffic per forward: identical.** ~16 GB of params per pass either
   way. There are no persistent buffers to add to the paging budget.
4. **Throughput: identical** (~0.65 tok/s observed for `=False`).

In other words, **`offload_buffers=True` is a no-op for Llama-3-8B**. The
flag's effect depends entirely on whether the model has persistent buffers.

## When `offload_buffers=True` actually matters

You need a model that holds large *persistent* buffers. Typical cases:

- **BatchNorm-heavy CNNs**: each `BatchNorm2d` has `running_mean` and
  `running_var` registered as persistent buffers (this is the default).
  In ResNet-152 these add up to ~76 KB total — trivial — so even there
  the flag is performance-neutral, but it does change *behavior*: at any
  given instant the BN layers' running stats live in `weights_map` (CPU),
  not on GPU.
- **Positional encodings stored as persistent buffers**: some older
  Transformer implementations (`SinusoidalPositionalEmbedding`, certain
  T5 variants' relative-position buckets, some custom RoPE caches with
  `persistent=True`) keep a `(max_len, d_model)` buffer that can be
  hundreds of MB.
- **Quantized models** (e.g., `bitsandbytes` `Linear8bitLt`): the `SCB`
  scale buffer is persistent and is paged together with the int8 weight.
  See the special-case handling in `pre_forward` at `hooks.py:360–362`
  and the `Linear8bitLt` branch in `post_forward` at `hooks.py:401–403`.
- **Models with KV cache implemented as a persistent buffer** (rare in
  the HF ecosystem, but possible in custom code).

If the persistent buffer is small (kilobytes), `=True` vs `=False` is
cosmetic. If it's hundreds of MB or larger, `=True` *increases* per-token
PCIe traffic by that amount; in exchange it frees that much GPU memory
between modules.

## Edge cases

- **`weights_map` storage with `=False`:** since `state_dict()` includes
  persistent buffers, they end up in `weights_map` (CPU) even when the
  flag is `False`. They just go unused by the per-leaf hooks — pure CPU
  RAM waste in that mode. With `=True` that storage is what gets paged
  in, so it's not wasted.
- **Tied params/buffers** (`tied_params_map`): the tied-weight memoization
  in `pre_forward`/`post_forward` (`hooks.py:367–376`, `407–416`) iterates
  the same `named_module_tensors(include_buffers=self.offload_buffers, ...)`,
  so tied **persistent buffers** are also memoized correctly under `=True`.
- **`set_module_tensor_to_device(..., 'meta')` on a buffer:** works
  identically to the param case — the module's `_buffers[name]` is
  replaced by a meta tensor.
- **No stream/prefetch overlap is added by `=True`.** The per-leaf,
  default-stream, synchronous copy model from `=False` carries over —
  with strictly more tensors per leaf to move.

## Pointer cheatsheet

| What                                  | Where                                      |
|---------------------------------------|--------------------------------------------|
| Entry point                           | `accelerate/big_modeling.py:179` `cpu_offload` |
| CPU master copy build                 | `big_modeling.py:209-210`                  |
| Hook attach recursion                 | `accelerate/hooks.py:479` `attach_align_device_hook` |
| Per-module hook                       | `hooks.py:242` `AlignDevicesHook`          |
| init: set tensors to meta             | `hooks.py:317-330`                         |
| init: place buffers on exec device    | `hooks.py:332-341`                         |
| pre_forward: H2D                      | `hooks.py:352-385`                         |
| post_forward: drop to meta            | `hooks.py:392-400`                         |
| Tensor iterator (params + buffers)    | `accelerate/utils/modeling.py:427` `named_module_tensors` |
| Non-persistent buffer set helper      | `utils/modeling.py:457` `get_non_persistent_buffers` |
