# `accelerate.cpu_offload(..., offload_buffers=False)` (default)

All file/line refs are against `accelerate==1.13.0`.

## TL;DR

`offload_buffers=False` is the default. The offload paging loop covers
**parameters only** — every buffer (persistent or not) is moved to the
execution device once at hook attach time and stays there for the lifetime
of the model.

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
non-persistent ones. So the CPU master copy contains persistent buffers
regardless of the flag — but with `=False` those entries are never read
back by the per-leaf hooks, so they sit in CPU RAM unused.

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

With `offload_buffers=False`:
- (a) `include_buffers=False` → only **params** are set to `meta`.
- (b) `module.named_buffers(...)` iterates **all** buffers (persistent +
  non-persistent) and moves them to `execution_device`. Persistent buffers
  live there for the lifetime of the model — paging never touches them.

With `offload_buffers=True` (for contrast):
- (a) sets **params + persistent buffers** to `meta`
  (`remove_non_persistent=True` excludes the non-persistent ones).
- (b) explicitly moves only the non-persistent buffers to
  `execution_device`.

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

With `offload_buffers=False` the iteration yields **params only** — only
parameters get the just-in-time H2D copy. Persistent buffers, already
resident on the execution device, are skipped here entirely.

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

Same iterator: with `=False` only params are dropped back to `meta` after
the forward. Buffers — persistent and non-persistent alike — remain
untouched on the execution device.

## Per-forward timeline (with `offload_buffers=False`)

For each module that owns direct tensors, in execution order:

1. Module's `forward` is called → PyTorch dispatches through accelerate's
   wrapped forward (`hooks.py:186–193`).
2. `pre_forward` runs: H2D copy of every **param** (not buffers); args
   moved to `execution_device`. Synchronous, `non_blocking=False`, default
   stream (verified at `utils/modeling.py:343`).
3. Original `forward` runs on GPU.
4. `post_forward` runs: every param is set to `meta`, freeing the GPU
   storage at the next caching-allocator op. Buffers stay where they are.

All buffers — persistent and non-persistent — are skipped at every step.
They sit on `execution_device` permanently, paid for once at attach time.

## Trace-validated behavior (PyTorch profiler, accelerate 1.13.0)

The source-level analysis above is correct but doesn't capture **how**
each pre/post_forward looks at the CUDA-runtime level. The following is
empirical: from PyTorch-profiler traces of `cpu_offload(model)` on
Llama-3.2-3B-Instruct and Llama-3.1-8B (bf16, batch=1, prompt_len=20,
15 generated tokens, `examples/trace/llama{3,8}b_offload_model/`).

### Per-leaf operation sequence

For one Linear leaf (e.g. `model.layers.0.self_attn.q_proj`, weight
~32 MB bf16), one token's CPU-side trace is:

```
pre_forward  → aten::to                  (args.to(exec_device), ~6 µs)
             → cuMemCreate × N           (expandable-segment alloc, ~10–35 µs each)
             → cuMemMap × N              (~0.3–2 µs each)
             → cudaStreamIsCapturing     (no-op check)
             → cudaMemcpyAsync           (submit H2D, blocks GPU stream)
             → cudaStreamSynchronize     (CPU waits for H2D to finish)
             → detach / aten::to / aten::to   (rebind module._parameters[name] → live GPU tensor)
compute      → aten::view / as_strided / cudaMemsetAsync / cuLaunchKernel ...
post_forward → detach / aten::to         (rebind module._parameters[name] → meta — no physical copy)
```

Three details that the source-only doc above does not call out:

1. **`cuMemCreate`/`cuMemMap` fire on every pre_forward.** PyTorch's
   default allocator on CUDA ≥ 11.4 uses expandable segments
   (`PYTORCH_CUDA_ALLOC_CONF` defaults that path on). After
   `post_forward` rebinds the param to `meta`, the storage is dropped;
   the next pre_forward (same module, next token) has to allocate
   afresh. Cost: ~10–35 µs per chunk, and large params chunk into
   many segments — e.g. `model.embed_tokens` and `lm_head` (788 MB
   each on 3B) issue ~37 back-to-back `cuMemCreate` + `cuMemMap` pairs
   before their single `cudaMemcpyAsync`.

2. **The H2D is followed by a hard `cudaStreamSynchronize`.** The
   `non_blocking=False` in `set_module_tensor_to_device` shows up as
   an explicit CPU-side wait that blocks until the Memcpy completes,
   *before* any compute kernel is launched. This is the actual
   serialization mechanism — there is no implicit ordering trick;
   the CPU thread literally pauses.

3. **`post_forward`'s drop-to-meta produces no Memcpy event.**
   Across both Llama traces the only direction observed is
   `Memcpy HtoD (Pageable -> Device)` — no D2H, no D2D. Setting a
   tensor to `meta` is a pure descriptor rebind: `_parameters[name]`
   points at a meta tensor, the previous CUDA storage's refcount
   drops to zero, and the allocator may unmap it later (lazily).

### Quantitative facts from the traces

| Quantity                                | Llama-3-3B | Llama-3-8B |
|-----------------------------------------|------------|------------|
| Hooked leaves with paged params         | 255        | 291        |
| `cudaMemcpyAsync` calls (15 tokens)     | 3826       | 4366       |
| `cudaMemcpyAsync` calls per token       | 255 + 0\*  | 291 + 0\*  |
| H2D bytes per token                     | 7.21 GB    | 16.06 GB   |
| H2D bytes total (15 tokens)             | 108.2 GB   | 240.9 GB   |
| `Memcpy DtoH` events                    | 0          | 0          |
| Distinct CUDA streams used for H2D      | 1          | 1          |
| Peak VRAM (allocator side)              | 810 MB     | 1.07 GB    |
| Throughput (inference)                  | 1.65 tok/s | 0.77 tok/s |

\* Plus exactly one "orphan" `cudaMemcpyAsync` at trace start — the
input-id tensor moving to GPU via user-code `input_ids.to(device)`,
not part of accelerate's hook machinery.

Per-class H2D cost (Llama-3-3B, averaged over 15 tokens × N modules):

| Module class           | n modules | bytes/H2D | GPU H2D time | CPU submit time |
|------------------------|-----------|-----------|--------------|-----------------|
| `Embedding` / `Linear` (vocab) | 2 (embed_tokens + lm_head) | 788 MB | ~57 ms | ~57 ms (blocked) |
| `Linear` (attention/MLP layers)| 196       | ~32 MB    | ~2.3 ms      | ~2.4 ms         |
| `LlamaRMSNorm`         | 57        | 6 KB      | ~0.55 µs     | ~8.4 µs         |

Two implications:

- **Embedding paging is wasteful.** A 788 MB H2D fires every token to
  do a gather over ~20 indices. accelerate hooks operate at module
  granularity; they have no concept of "only fetch the rows you need."
- **Small modules are submit-overhead-bound, not bandwidth-bound.**
  RMSNorm's 6 KB H2D would take ~0.5 µs at line rate; the CPU submit
  path (cuMemCreate + cuMemMap + cudaMemcpyAsync + cudaStreamSync)
  spends ~8 µs in user-mode CUDA driver code. Stacking 57 RMSNorms
  per token × 15 tokens = ~7 ms of pure launch overhead.

### `Pageable -> Device` — the CPU master copy is **not** pinned

Every Memcpy event records the kind as `Memcpy HtoD (Pageable -> Device)`.
That is, the CPU source buffer (`weights_map[name]`, populated at
`big_modeling.py:209`) lives on **pageable** memory, not pinned host
memory. Consequences:

- PCIe DMA cannot run at peak bandwidth; the driver must stage through
  a pinned bounce buffer or page-pin on the fly.
- Multi-stream overlap with compute is also impossible with pageable
  source memory (cuMemcpyAsync from pageable host is effectively
  synchronous w.r.t. the issuing thread anyway).
- accelerate has no flag to make this pinned. Pinning would require
  reworking `state_dict = {n: p.to("cpu") for n, p in model.state_dict().items()}`
  at `big_modeling.py:209` to use `pin_memory=True`. This is the
  largest single-knob performance opportunity the source-level doc
  does not surface.

#### Bandwidth distribution: unimodal, not bimodal

A natural follow-up question is whether the per-transfer bandwidth shows
any *bimodality* — i.e. some transfers hitting a pinned-fast path (e.g.
~22–24 GB/s on PCIe 4.0 x16) and others stuck on a true-pageable slow
path (~5–10 GB/s). The trace says **no**.

Computed per-transfer bandwidth = `module_size_bytes / Memcpy_event_duration_ns`
over the 2130 H2D transfers ≥ 10 MB in the 3B trace, and 2430 in the
8B trace:

| Statistic                | Llama-3-3B (n=2130) | Llama-3-8B (n=2430) |
|--------------------------|---------------------|---------------------|
| mean                     | 14.17 GB/s          | 14.91 GB/s          |
| std                      | 1.49 GB/s           | 0.79 GB/s           |
| p1 / p50 / p99           | 9.4 / 14.8 / 16.6   | 11.6 / 15.1 / 15.8  |
| min / max                | 7.68 / 17.0         | (similar range)     |
| skewness                 | −1.32               | −2.64               |
| **excess kurtosis**      | **+1.33**           | **+9.10**           |

ASCII histogram (3B, large transfers only, 0.5 GB/s bins, bar = ∝ count):

```
   7.5- 8.0 GB/s  n=   2
   8.0- 8.5 GB/s  n=   3
   8.5- 9.0 GB/s  n=   5
   9.0- 9.5 GB/s  n=  10  #
   9.5-10.0 GB/s  n=  15  #
  10.0-10.5 GB/s  n=  23  ##
  10.5-11.0 GB/s  n=  34  ###
  11.0-11.5 GB/s  n=  62  #####
  11.5-12.0 GB/s  n=  81  ######
  12.0-12.5 GB/s  n= 105  ########
  12.5-13.0 GB/s  n=  90  #######
  13.0-13.5 GB/s  n= 102  ########
  13.5-14.0 GB/s  n= 156  ############
  14.0-14.5 GB/s  n= 250  ###################
  14.5-15.0 GB/s  n= 391  #############################
  15.0-15.5 GB/s  n= 532  ######################################## ← mode
  15.5-16.0 GB/s  n= 260  ####################
  16.0-16.5 GB/s  n=   9  #
```

Both runs are **unimodal** with a left-tail. Excess kurtosis is positive
on both (bimodal distributions have **negative** excess kurtosis — the
empirical positive values rule out a two-mode mixture). The left-tail
events all come from the Linear class (mid-sized 18–50 MB transfers),
scattered across tokens — i.e. they are transient PCIe / DRAM
contention, not a separate transfer class.

**Interpretation.** Every H2D goes through the same code path. The CUDA
driver internally stages the pageable source through a pre-allocated
pinned bounce buffer; the on-the-wire DMA from that bounce buffer sees
~14–15 GB/s — well above a true-pageable copy (5–10 GB/s) but well
below a true-pinned-source copy (~22–24 GB/s on PCIe 4.0 x16, which the
trace's source-class label of `(Pageable -> Device)` correctly reports
as *not* the path being used). The `(Pageable -> Device)` label is about
the **source classification**, not the actual DMA performance regime —
the regime is "pageable-with-driver-staged-bounce", a single unimodal
mode shared by all H2Ds.

The submit-side (CPU cudaMemcpyAsync wall-time) bandwidth tracks the
GPU-side number within 1% on average — i.e. the CPU thread is not
spending meaningful time pinning pages on the fly; the bounce buffers
are reused. The CPU-side overhead per H2D is dominated by
`cudaStreamSynchronize`, not by staging.

### Stream usage: single-stream, fully serialized

All 3826 / 4366 H2Ds in both traces share a single `stream_id`. With
the default-stream model and pageable source memory, there is **no
overlap** between:

- H2D of module *N* and compute of module *N* (synchronize-after-copy)
- H2D of module *N* and H2D of module *N+1* (same stream)
- H2D of module *N* and compute of module *N-1* (synchronize fences any
  in-flight kernel, but the next module's H2D can't start until that
  kernel and its own param-rebind finishes on the CPU thread)

The serialization is end-to-end, per module. Wall-clock per token ≈
`Σ_modules (alloc_overhead + H2D_time + launch_overhead + kernel_time)`.

### Allocator churn: `cuMemUnmap` / `cuMemRelease`

The trace also shows 2088 `cuMemUnmap` + 2088 `cuMemRelease` calls per
3B run, attributed mostly to the *next* module's pre_forward window
(LlamaRMSNorm hosts most of them by virtue of running right after the
big Linears). These are the expandable-segment allocator reclaiming
freed-but-still-mapped chunks. The asymmetry (`cuMemCreate` count
3826 vs `cuMemUnmap` 2088) shows that ~45% of allocations are reused
from the caching pool without unmapping, so the alloc overhead is
amortized partially but never eliminated.

### Buffer behavior — empirically confirmed

`model.rotary_emb` (the only `LlamaRotaryEmbedding` instance, with two
non-persistent `inv_freq` buffers) generates **405 events** in the
Llama-3-3B trace — `cudaLaunchKernel`, `aten::as_strided`,
`aten::to`, etc. — but **zero `cudaMemcpyAsync`**. Buffers placed at
`init_hook` time stay on `cuda:0`; the rotary forward runs on GPU
without any paging, exactly as the source analysis predicted. The
same holds on Llama-3-8B (0 of 4366 H2Ds go to `rotary_emb`).

## Effect on Llama-3-8B (cross-check)

Llama-3-8B contains **two buffers**, both **non-persistent** (verified by
`check_buffers.py`):

```
model.rotary_emb.inv_freq          (non-persistent)
model.rotary_emb.original_inv_freq (non-persistent)
```

Source: `transformers/models/llama/modeling_llama.py:89-90` and
`transformers/modeling_rope_utils.py:73,79,109,116` — all use
`register_buffer(..., persistent=False)`.

Consequences for `cpu_offload(model, offload_buffers=False)`,
confirmed against the `llama8b_offload_model` trace:

1. **Hook count: 292.** Same as `=True`.
2. **Buffer placement:** the two `inv_freq` tensors are moved to `cuda:0`
   in `init_hook` (step (b), the `not self.offload_buffers` branch) and
   stay there forever. `pre_forward`/`post_forward` never touch them.
   Trace shows zero memcpy events attributed to `model.rotary_emb`.
3. **PCIe traffic per forward:** **16.06 GB** of params per pass
   (trace-measured; matches "~16 GB" in the source-level doc). Buffers
   contribute zero — there are no persistent buffers.
4. **Throughput:** 0.77 tok/s (15 tokens / 19.59 s; profile run).
   Allocator+submit overhead dominates the small-module fraction;
   PCIe bandwidth dominates the large-module fraction (lm_head and
   embed_tokens alone contribute ~1.5 GB of the 16 GB per-token traffic).

Because Llama-3-8B's only buffers are non-persistent, the `=False` and
`=True` modes are observationally identical for this model — the flag's
effect hinges entirely on persistent-buffer presence.

## When the `=False` default is the wrong choice

`=False` keeps **all buffers** permanently on the execution device. This
is fine when buffers are small or absent, but wasteful (or limiting) when
persistent buffers are large. Typical cases where `=True` is preferable:

- **BatchNorm-heavy CNNs**: each `BatchNorm2d` has `running_mean` and
  `running_var` registered as persistent buffers (default). In ResNet-152
  these total ~76 KB — trivial — so `=False` costs nothing observable;
  `=True` would just move that 76 KB to the paging budget.
- **Positional encodings stored as persistent buffers**: some older
  Transformer implementations (`SinusoidalPositionalEmbedding`, certain
  T5 variants' relative-position buckets, some custom RoPE caches with
  `persistent=True`) keep a `(max_len, d_model)` buffer that can be
  hundreds of MB. With `=False` that storage stays pinned to the GPU.
- **Quantized models** (e.g., `bitsandbytes` `Linear8bitLt`): the `SCB`
  scale buffer is persistent. With `=False`, every `SCB` tensor in the
  model is GPU-resident at all times — the int8 weight is paged but its
  scale is not. See special-case handling in `pre_forward` at
  `hooks.py:360–362` and the `Linear8bitLt` branch in `post_forward` at
  `hooks.py:401–403` (only relevant under `=True`).
- **Models with KV cache implemented as a persistent buffer** (rare in
  the HF ecosystem, but possible in custom code): under `=False` the
  entire cache stays GPU-resident.

If the persistent buffer is small (kilobytes), `=False` vs `=True` is
cosmetic. If it's hundreds of MB or larger, `=False` *pins* that much GPU
memory; `=True` would free it between modules at the cost of paging it in
on every forward.

## Edge cases

- **CPU RAM waste in `weights_map`:** since `state_dict()` includes
  persistent buffers, they end up in `weights_map` (CPU) even though the
  per-leaf hooks never read them back under `=False`. The CPU master
  copy is built unconditionally, so persistent-buffer bytes are
  duplicated: once on the execution device (live) and once in
  `weights_map` (dead weight). Under `=True` that CPU copy is the
  authoritative one being paged in.
- **Tied params/buffers** (`tied_params_map`): the tied-weight memoization
  in `pre_forward`/`post_forward` (`hooks.py:367–376`, `407–416`) iterates
  `named_module_tensors(include_buffers=self.offload_buffers, ...)`. Under
  `=False` the iterator yields params only, so tied buffers are not
  memoized — but they don't need to be, because they're already resident
  and shared via the single GPU storage placed in `init_hook`.
- **`set_module_tensor_to_device(..., 'meta')` on a buffer:** doesn't
  happen under `=False` — buffers are never moved to `meta`. (Under
  `=True` it works identically to the param case.)
- **Single stream, synchronous copy.** Per-leaf, default-stream,
  blocking H2D for every parameter; no prefetch overlap. This is the
  same copy model under `=True` — `=False` just has fewer tensors per
  leaf to move.

## Pointer cheatsheet

| What                                  | Where                                      |
|---------------------------------------|--------------------------------------------|
| Entry point                           | `accelerate/big_modeling.py:179` `cpu_offload` |
| CPU master copy build                 | `big_modeling.py:209-210`                  |
| Hook attach recursion                 | `accelerate/hooks.py:479` `attach_align_device_hook` |
| Per-module hook                       | `hooks.py:242` `AlignDevicesHook`          |
| init: set tensors to meta (params)    | `hooks.py:317-330`                         |
| init: place buffers on exec device    | `hooks.py:332-341` (the `not self.offload_buffers` branch) |
| pre_forward: H2D (params only)        | `hooks.py:352-385`                         |
| post_forward: drop to meta (params)   | `hooks.py:392-400`                         |
| Tensor iterator (params + buffers)    | `accelerate/utils/modeling.py:427` `named_module_tensors` |
| Non-persistent buffer set helper      | `utils/modeling.py:457` `get_non_persistent_buffers` (unused under `=False`) |
