# `accelerate.cpu_offload` — Specification for Simulator Implementation

A self-contained spec for faithfully replicating `accelerate.cpu_offload` behavior in a custom simulator. Derived from reading the source (`accelerate/big_modeling.py`, `accelerate/hooks.py`) and from empirical probes against Meta-Llama-3-8B.

Companion files in this directory:
- `accelerate-cpu-offload.md` — narrative description of the algorithm
- `llama-3-8B-mp.py` — monkey-patched generation script that emits the event log
- `llama-3-8B-Offload-inspect.py` — JSON-emitting inspector (static structure + event trace)

---

## 1. Scope and locked configuration

The simulator targets exactly one accelerate offload API:

```python
accelerate.cpu_offload(
    model,                              # required: the nn.Module to offload
    execution_device=torch.device("cuda:0"),
    offload_buffers=False,              # buffers stay on GPU permanently
    state_dict=None,                    # built automatically from model.state_dict()
    preload_module_classes=None,        # no coarse grouping; per-leaf granularity
)
```

The two algorithmically meaningful parameters are `offload_buffers` and `preload_module_classes`. Defaults are listed above. Their effects are described in section 9.

Outside the offload call: the simulator targets a single execution device (one GPU). All other model components, datasets, and pipeline state are out of scope.

---

## 2. Algorithm in two phases

### Phase 1 — Attach (one-time, when `cpu_offload` returns)

1. **Build the CPU master copy.** If the user didn't supply `state_dict`, accelerate runs `{n: t.to("cpu") for n, t in model.state_dict().items()}`. This is a single dict holding every parameter and buffer on **pageable** CPU memory (no `pin_memory()`).

2. **Attach the wrapper hook.** `add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)` is applied to the top-level model. This hook has `offload=False` and exists only to move the output back to the caller's input device at the end of the full forward. It carries no parameters of its own and contributes no transfer events.

3. **Recursively walk the module tree** with `attach_align_device_hook`:
   - For each module `m`, check `len(list(named_module_tensors(m))) > 0` (i.e., does `m` own at least one direct parameter or buffer at this level — equivalent to `recurse=False`).
   - If yes: attach an `AlignDevicesHook(execution_device=…, offload=True, weights_map=PrefixedDataset(state_dict, m.name))` to `m` via `add_hook_to_module`. This hook is the one that actually moves data.
   - If `m`'s class is in `preload_module_classes`: attach the hook with `place_submodules=True` (which makes the hook treat the entire subtree as its own scope) and **stop the recursion**. Otherwise recurse into `m.children()`.

4. **Per-hook initialization** (`AlignDevicesHook.init_hook`):
   - Record each direct tensor's original device into `self.original_devices` (used only if the hook is later detached).
   - Reuse the CPU `weights_map` (already on CPU).
   - For every direct tensor: `set_module_tensor_to_device(module, name, "meta")` — replaces the on-module tensor's `.data` with a zero-byte `meta` placeholder. Real GPU memory is now zero for this module's params.
   - If `offload_buffers=False` (default): for every direct buffer, `set_module_tensor_to_device(module, name, execution_device)` — the buffer is placed on GPU and stays there for the model's lifetime. It does **not** participate in the load/evict cycle.
   - If `offload_buffers=True`: buffers are treated like parameters and get the `meta` placeholder treatment.

5. **Rewrite `forward`.** `add_hook_to_module` saves the original `module.forward` as `module._old_forward`, then replaces `module.forward` with `new_forward`:

   ```python
   def new_forward(module, *args, **kwargs):
       args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
       output = module._old_forward(*args, **kwargs)   # the real computation
       return module._hf_hook.post_forward(module, output)
   ```

State after attach:
- Every direct-tensor-owning leaf module has an `AlignDevicesHook(offload=True)`.
- The top-level model has an additional `AlignDevicesHook(io_same_device=True)`.
- All offloaded parameter tensors on the model are `meta` placeholders (zero device memory).
- Buffers (under default `offload_buffers=False`) sit permanently on the execution device.
- The CPU master copy in `state_dict` is the single source of truth for weights — never modified after attach.

### Phase 2 — Per-forward (every model call, every submodule call)

Every wrapped `module(...)` call follows the same three-step pattern:

```text
pre_forward(module):
    for each direct param (and buffer if offload_buffers=True) of module:
        set_module_tensor_to_device(module, name, execution_device, value=weights_map[name])
        # i.e. allocate GPU memory and copy weights_map[name] (CPU) -> GPU.
        # Synchronous on the default stream. No non_blocking flag.
    args, kwargs = send_to_device(args, execution_device), send_to_device(kwargs, execution_device)
    return args, kwargs

old_forward(*args, **kwargs)
    # the real computation; weights are present on GPU for the duration.

post_forward(module, output):
    for each direct param (and buffer if offload_buffers=True) of module:
        set_module_tensor_to_device(module, name, "meta")
        # Frees the GPU allocation. The CPU master copy in weights_map is NOT touched.
    if io_same_device (top-level wrapper only):
        output = send_to_device(output, input_device)
    return output
```

Key properties of the per-forward path:
- All H2D copies execute on the **default CUDA stream**.
- `set_module_tensor_to_device` does a `tensor.to(device)` equivalent **without** `non_blocking=True`. So the copy is synchronous — the host blocks until the H2D transfer is complete.
- `post_forward` sets the on-module tensor to `meta`. This is an O(1) reference swap, not a copy. The actual GPU memory is freed by the allocator when the previous tensor's refcount drops to zero. There is **no copy back to CPU** — the CPU master copy in `weights_map` was never modified.
- No prefetching. No side stream. No chain of dependent groups.

---

## 3. Hook attachment rules

The recursive rule from `attach_align_device_hook` (hooks.py:479):

```text
def attach(m, prefix=""):
    if any direct param or buffer (named_module_tensors(m, recurse=False)) exists:
        attach AlignDevicesHook(offload=True, weights_map=PrefixedDataset(state_dict, prefix))
    if m.__class__.__name__ in preload_module_classes:
        # treat subtree as one unit; do NOT recurse
        attach AlignDevicesHook with place_submodules=True
        return
    for name, child in m.named_children():
        attach(child, f"{prefix}.{name}" if prefix else name)
```

Properties of the resulting hook layout (with default `preload_module_classes=None`):
- Container modules (`nn.ModuleList`, decoder layers, MLPs, attention blocks) typically own no direct parameters — only their children do. They get **no hook**.
- Weight-owning leaves (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, etc.) each get **one hook**.
- The top-level model gets one extra hook (`io_same_device=True`, `offload=False`) regardless of whether it owns direct parameters.

For Llama-3-8B, this rule produces **293 hooks**:

| Class | Hook count | Why |
|---|---:|---|
| `Linear` | 225 | 4 attention + 3 MLP per decoder layer × 32 layers + 1 lm_head |
| `LlamaRMSNorm` | 65 | 2 per decoder layer + 1 final norm |
| `Embedding` | 1 | `embed_tokens` |
| `LlamaRotaryEmbedding` | 1 | owns only a buffer; counts because of `named_module_tensors` |
| `LlamaForCausalLM` | 1 | top-level wrapper hook (`io_same_device=True`, 0 bytes) |

Under `preload_module_classes=["LlamaDecoderLayer"]`, the recursion stops at each `LlamaDecoderLayer`, and the per-decoder-layer hook treats the layer's whole subtree as one offload unit. Hook count drops to ~34 (32 layers + embed_tokens + lm_head + wrapper), and each transfer becomes ~500 MB instead of ~33 MB. The algorithm is otherwise identical.

---

## 4. Memory model

### CPU side

The `state_dict` dict holds one CPU tensor per parameter and buffer in the model. Storage characteristics:
- **Pageable** (not pinned). Pinning is never performed by `cpu_offload`.
- Allocated once at attach time and never modified afterwards.
- Lives for the lifetime of the offloaded model.

Total CPU footprint = the full model's weight size. For Llama-3-8B in fp16, ≈ 14.96 GiB.

### GPU side

At any wall-clock point during a forward pass, the GPU holds:
- Any permanently resident buffers (under default `offload_buffers=False`). For Llama-3-8B this is the `LlamaRotaryEmbedding.inv_freq` buffer at 512 bytes. Negligible.
- The direct parameters of **exactly one** leaf module being currently executed (its hook's `pre_forward` ran; `post_forward` hasn't yet).
- The intermediate activations and inputs/outputs of that module (out of scope for this spec — model-specific, additive on top).

Peak weight residency is **one leaf module's direct params**. For Llama-3-8B, the largest single load is the `lm_head` Linear at ~1.05 GB; the gate/up/down projections in each MLP are ~117 MB each; q/o_proj are ~33 MB each; k/v_proj are ~8 MB each.

Brief nesting overlap: when module A's `pre_forward` has fired but A's `old_forward` then calls submodule B, B's `pre_forward` fires before A returns. In practice this only matters for the top-level wrapper hook (which has 0 bytes anyway), since real leaf modules contain no further hooked submodules.

---

## 5. Streams and synchronization

There is **one stream**: the default CUDA stream.

Properties:
- Every H2D copy in `pre_forward` is `set_module_tensor_to_device(... value=cpu_tensor)`, which internally does `cpu_tensor.to(device)` — no `non_blocking` flag, so it's a synchronous copy on the default stream.
- Every kernel from the wrapped `old_forward` runs on the default stream.
- The synchronous copy in `pre_forward` blocks the host until the H2D is complete; then `old_forward` runs; then `post_forward` swaps the tensor to `meta` (host-only, free).
- No `record_stream` calls. No allocator deferred-reclamation behavior to model.
- No side stream. No async transfer. No prefetch.

Implication: each forward pass through a leaf module is a strict `H2D → compute → free` sequence on the same stream. The GPU is either copying or computing at any given moment, never both.

---

## 6. Determinism guarantees

`cpu_offload` is fully deterministic:

| Property | Deterministic? |
|---|:---:|
| Set of hooks attached | ✓ |
| Order of hooks fired per forward | ✓ (= module call order in `model.forward`) |
| Per-event `(action, module_name, bytes)` | ✓ |
| Total bytes moved per forward | ✓ |
| Peak GPU residency (in bytes) | ✓ |
| Wall-clock timestamp of each event | ✗ (kernel/copy execution variance, allocator state) |

The hook layout is established at attach time and unchanged thereafter. The event sequence is purely a function of the model's `forward` implementation and the hook attachment rule above.

---

## 7. Empirical reference numbers

Measured on Meta-Llama-3-8B in fp16 on RTX 3090, prompt `"Write a short story about a robot learning emotions."`, `MAX_NEW_TOKENS=4`, `do_sample=False`. Captured by `llama-3-8B-Offload-inspect.py`.

| Metric | Value |
|---|---:|
| Hooks attached (excluding the io_same_device wrapper) | 292 |
| Total hooks (including wrapper) | 293 |
| Generation tokens | 4 |
| Elapsed | 9.137 s |
| Events recorded (load + evict pairs) | 2,344 |
| Total bytes CPU→GPU | 64.24 GB |
| Bytes per generation token | ≈ 14.96 GiB (matches full model size) |
| Median load size | 4.72 MB |
| Min load size | 512 B (`LlamaRotaryEmbedding` buffer — counted but stays resident under default) |
| Max load size | 1.05 GB (`embed_tokens` and `lm_head`, separately) |
| Max concurrent loaded modules (non-zero bytes) | 1 |
| Max concurrent loaded modules (including 0-byte wrapper) | 2 |

Per-class breakdown (from one of the runs):

| Class | Unique modules | Loads in run | Bytes/load (min … max) | Total bytes |
|---|---:|---:|---|---:|
| `Linear` | 225 | 900 | 8.4 MB … 1.05 GB | 60.04 GB |
| `LlamaRMSNorm` | 65 | 260 | 8.2 KB | 2.13 MB |
| `LlamaForCausalLM` | 1 | 4 | 0 (wrapper) | 0 |
| `Embedding` | 1 | 4 | 1.05 GB | 4.20 GB |
| `LlamaRotaryEmbedding` | 1 | 4 | 512 B | 2.05 KB |

(Note: each generation forward loads each Linear exactly once. 225 Linears × 4 tokens = 900 loads. ✓)

---

## 8. Simulator implementation guide

### State

```text
HookedModule:
    name: str (e.g. "model.layers.5.self_attn.q_proj")
    class_name: str
    direct_param_bytes: int     # recurse=False sum
    direct_buffer_bytes: int    # recurse=False sum
    bytes_for_offload: int      # = direct_param_bytes + (direct_buffer_bytes if offload_buffers else 0)
    is_top_level_wrapper: bool  # the io_same_device hook on the root model

Allocator/Memory:
    cpu_state_dict_bytes: int   # sum of all weight tensors; pageable
    gpu_resident_bytes: int     # current GPU weight residency
    gpu_persistent_buffer_bytes: int  # buffers that stay forever
```

### Build pipeline

1. **Decide `offload_buffers` and `preload_module_classes`** (the only two knobs). These determine the hook layout.

2. **Walk the model tree**, recursing into children. For each module:
   - Compute `has_direct_tensor = direct_params(m, recurse=False) ∪ (direct_buffers(m, recurse=False) if offload_buffers else ∅)` is non-empty. (Note: buffer ownership counts for hook attachment even when `offload_buffers=False`; the hook still fires, it just doesn't move the buffer.)
   - If `has_direct_tensor`: attach a `HookedModule` entry.
   - If `m.class in preload_module_classes`: attach with `place_submodules=True` (sum the whole subtree's bytes into `bytes_for_offload`), and stop recursing.
   - Else recurse into children.

3. **Attach the top-level wrapper** `HookedModule(is_top_level_wrapper=True, bytes_for_offload=0)`.

4. **Initialize state**:
   - `cpu_state_dict_bytes = sum of all weight tensors in the model`
   - `gpu_resident_bytes = 0`
   - `gpu_persistent_buffer_bytes = sum of buffers if not offload_buffers else 0`

### Per-call state machine

When the simulator sees a call to a `HookedModule`:

```text
pre_forward(h):
    if h.bytes_for_offload > 0:
        emit event ("load", h.name, h.bytes_for_offload)
        gpu_resident_bytes += h.bytes_for_offload
    move args/kwargs to execution device (model-specific; outside this spec)

forward(h):
    # compute happens; the simulator's job is to schedule the compute
    # against the GPU using its compute-cost model. The H2D copy
    # finished synchronously before this step started.

post_forward(h):
    if h.bytes_for_offload > 0:
        emit event ("evict", h.name, h.bytes_for_offload)
        gpu_resident_bytes -= h.bytes_for_offload
    if h.is_top_level_wrapper:
        move output to input_device
```

All events are emitted on a single logical stream. No record_stream model is required. No deferred reclamation.

### Validation

Diff the simulator's event sequence against the JSON dump produced by `llama-3-8B-Offload-inspect.py`. The relevant field is the event list with `(action, module_class, module_id, bytes)`. Match on the `(action, module_class, bytes)` projection — accelerate names hooks by id() at runtime, so the raw `module_id` won't be portable, but the class + byte size should match exactly.

---

## 9. Tunable parameters in real accelerate under this configuration

`cpu_offload` exposes only two algorithmic knobs:

| Parameter | Default | Effect when changed |
|---|---|---|
| `offload_buffers: bool` | `False` | If `True`, every direct buffer (e.g. `LlamaRotaryEmbedding.inv_freq`) participates in the load/evict cycle. Under `False` (default), buffers are placed on the execution device at attach time and stay there permanently. For typical LLMs, the bytes involved are negligible (Llama-3-8B has ~512 B of buffers); the knob matters only for models with large persistent buffers. |
| `preload_module_classes: list[str]` | `None` | When a module's class name appears in this list, hook attachment stops recursing at that module, and the single hook attached treats the subtree as one offload unit (`place_submodules=True`). For Llama-3-8B, `["LlamaDecoderLayer"]` collapses 292 leaf hooks to ~34 layer-granularity hooks and changes the median transfer size from 4.72 MB to ~500 MB. The total bytes moved per forward is unchanged; only the granularity is. |

The remaining parameters (`model`, `execution_device`, `state_dict`) are structural inputs, not algorithmic options. They don't change the per-forward schedule.

There are no other tunable parameters. Specifically, there is no exposed way in real accelerate to enable streams, prefetching, pinning, or alternate eviction policies under `cpu_offload`. Anything outside the two parameters above is fixed by the implementation.

---

## 10. One-line summary

**`cpu_offload` keeps one pageable CPU copy of the full state_dict, attaches one synchronous `AlignDevicesHook` per direct-tensor-owning leaf module, and on every leaf forward call loads that leaf's parameters CPU→GPU on the default stream, runs the forward, then frees them by reassigning the tensor to `meta` — with no prefetch, no streams, no pinning, and no write-back.**
