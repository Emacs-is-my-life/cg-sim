# Group + Async Offload — Specification for Simulator Implementation

A self-contained spec for replicating Diffusers's group-offloading-with-async-stream behavior in a custom simulator. Derived from reading the source (`diffusers/hooks/group_offloading.py`, `accelerate/hooks.py`) and from empirical probes against Stable Diffusion 3 Medium (`stabilityai/stable-diffusion-3-medium-diffusers`).

Companion files in this directory:
- `locked-config-trace.md` — full per-event report for the locked spec
- `locked-config-trace.json` — raw event log + structural metadata
- `diffusers-offload-modes.md` — exhaustive comparison of all Diffusers offload modes (broader context)
- `SD3-Medium-Offload-inspect.py` — the probe script that produced the above

---

## 1. Scope and locked configuration

The simulator targets exactly one Diffusers offload variant:

```python
apply_group_offloading(
    pipe.transformer,                   # target module (only the transformer)
    onload_device=torch.device("cuda:0"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",         # group by ModuleList/Sequential children
    num_blocks_per_group=1,             # forced to 1 by streams; explicit here
    use_stream=True,                    # async transfer via side stream
    record_stream=True,                 # skip per-eviction sync
    non_blocking=True,                  # forced True by streams; explicit here
    low_cpu_mem_usage=False,            # eager-pin all weights once
    offload_to_disk_path=None,          # RAM-only
    block_modules=None,                 # default ModuleList/Sequential rule
    exclude_kwargs=[],                  # don't auto-infer _skip_keys
)
```

Non-offloaded components (`text_encoder`, `text_encoder_2`, `text_encoder_3`, `vae`) are pinned to the execution device for the full pipeline lifetime. Only the transformer is group-offloaded.

---

## 2. Algorithm in two phases

The simulator assumes the user runs **one warmup forward externally** between `apply_group_offloading(...)` and the start of profiling. Diffusers's internal lazy-chain-construction machinery runs during that warmup forward and is outside the simulator's scope. From here on, "Phase 1" describes the state at the moment profiling begins (post-warmup), and "Phase 2" describes every profiled forward call.

### Phase 1 — Initial state (when profiling begins)

By this point: `apply_group_offloading(...)` has been called with the locked config, and one transformer forward has executed externally. The runtime state the simulator must reproduce is:

**25 `ModuleGroup` objects**, formed by walking the transformer's `named_children()`:

- `ModuleList` / `Sequential` children → chunked into groups of `num_blocks_per_group=1`. Each `transformer_blocks[i]` becomes one `ModuleGroup`. → 24 block groups.
- Everything else (`pos_embed`, `time_text_embed`, `context_embedder`, `norm_out`, `proj_out`) plus any top-level params/buffers → bundled into the single **unmatched_group**.

**Per-group fields** at the moment profiling begins:

| Field | Block groups (1..24) | Unmatched_group |
|---|---|---|
| `modules` | `[transformer_blocks[i]]` | `[pos_embed, time_text_embed, context_embedder, norm_out, proj_out]` |
| `onload_leader = offload_leader` | the single member | the top-level transformer |
| `stream` | the side CUDA stream | `None` (always sync, default stream) |
| `non_blocking` | `True` | `False` |
| `record_stream` | `True` | `False` |
| `onload_self` | `False` | `True` |
| `next_group` | the next block in the chain | first block in execution order |
| `cpu_param_dict` | pinned copy of all member weights | empty (no pinning) |

**Prefetch chain**: `unmatched_group → transformer_blocks[0] → transformer_blocks[1] → … → transformer_blocks[23]`. The unmatched_group is the chain head (only one with `onload_self=True`). The 24th block group is the chain tail (`next_group=None`).

**Memory state**: all groups are on CPU. GPU weight residency is 0 (apart from the unmatched_group's permanently-resident pieces that don't go through pinning — though they too end up CPU between forwards). Pinned CPU footprint = ~4.02 GB (block weights only; unmatched_group is pageable).

### Phase 2 — Steady state (every transformer forward call)

Each transformer forward executes this pattern:

```text
transformer.pre_forward (unmatched_group):
    onload(unmatched_group)         # sync, no stream, blocking
    # Note: there's no prefetch of next_group here because the
    # unmatched_group's pre_forward fires before the first block. The
    # prefetch chain begins when the unmatched_group is itself a
    # next_group, but it IS the head — instead it triggers the next
    # group's onload via the chain.

block_0.pre_forward:
    # block_0.onload_self == False, so it expects to have been prefetched.
    # But on the very first call, no one has prefetched it yet —
    # the unmatched_group's pre_forward at the top should have done so.
    onload(block_1)                 # ← prefetched on side stream

    record_stream(default_stream) on block_0's tensors
    (block_0's weights are ready — either prefetched by unmatched, or sync-loaded if not)

block_0.forward runs on default stream.

block_0.post_forward (offload_leader):
    offload(block_0)                # weights back to CPU, .data swapped to cpu_param_dict copy

block_1.pre_forward:
    onload(block_2)                 # prefetch
    record_stream on block_1's tensors

block_1.forward runs on default stream.

block_1.post_forward:
    offload(block_1)

…  repeats for blocks 2..23 …

block_23.pre_forward:
    # next_group is None — no prefetch
    record_stream on block_23's tensors

block_23.forward runs.

block_23.post_forward:
    offload(block_23)

transformer.post_forward:
    offload(unmatched_group)        # sync
```

Net resident-set evolution per forward:

```text
t=0  : empty
t=1  : {unmatched}                    +304 MB  → 304 MB
t=2  : {unmatched, block_0}           +170 MB  → 474 MB   (sync-loaded by head's pre_forward via chain)
t=3  : {unmatched, block_0, block_1}  +170 MB  → 644 MB   (prefetched by block_0.pre_forward)
t=4  : {unmatched, block_1}           -170 MB  → 474 MB   (block_0 offloaded)
t=5  : {unmatched, block_1, block_2}  +170 MB  → 644 MB
…
```

Peak resident: **3 groups / 644 MB** (unmatched + current + prefetched-next). This is the resident-set during the iterative phase. At the end, only the unmatched_group remains briefly resident before being offloaded.

---

## 3. Group construction rules

For SD3's transformer, with `named_children()` returning:

```text
pos_embed              PatchEmbed
time_text_embed        CombinedTimestepTextProjEmbeddings
context_embedder       Linear
transformer_blocks     ModuleList (24 × JointTransformerBlock)
norm_out               AdaLayerNormContinuous
proj_out               Linear
```

The 25 resulting groups, in the **observed forward execution order** (= the order the lazy hook discovers):

| # | group_id | members | bytes | onload_self (steady) |
|---:|---|---|---:|:---:|
| 0 | unmatched | `pos_embed`, `time_text_embed`, `context_embedder`, `norm_out`, `proj_out` (+ top-level params/buffers) | 304.40 MB | **True** (head) |
| 1 | `transformer_blocks_0_0` | `transformer_blocks.0` | 169.96 MB | False |
| 2 | `transformer_blocks_1_1` | `transformer_blocks.1` | 169.96 MB | False |
| … | … | … | 169.96 MB | False |
| 23 | `transformer_blocks_22_22` | `transformer_blocks.22` | 169.96 MB | False |
| 24 | `transformer_blocks_23_23` | `transformer_blocks.23` | 108.59 MB | False |

Notes:
- Block 23 is smaller because it uses `context_pre_only=True` and drops the context projections.
- The unmatched_group is the chain **head**, not the tail. The lazy hook makes the first-observed real group its `next_group` and sets that group's `onload_self=False`.

---

## 4. Memory model

### CPU side

Pinned RAM holds the master copy of every offloaded weight:

```text
pinned_cpu = sum over block_groups of (params + buffers)
            ≈ 24 * 170 MB - (savings for block_23) = 4.02 GB
```

The unmatched_group does **not** contribute to pinned RAM (it has `stream=None`, so `_init_cpu_param_dict` returns `{}` and weights are managed by ordinary `.to("cpu")` calls).

Pageable (non-pinned) RAM holds:
- The unmatched_group's tensors when not on GPU.
- Tokenizer state, scheduler state, etc.

### GPU side

Resident weights at any wall-clock point in steady state:
- Always: `unmatched_group` (304 MB) — resident throughout the entire transformer forward.
- Current block (170 MB).
- Prefetched-next block (170 MB), if not at chain tail.
- Possibly the just-finished block (170 MB) until its memory is reclaimed by the allocator (with `record_stream=True`, reclamation is deferred until default stream passes the recording point).

Practical peak: **~644 MB**. Worst-case transient peak considering allocator lag: **~814 MB** (one extra block's worth of "freed but not yet reclaimable" memory).

Activations and KV-state are separate and additive on top of this. The simulator should treat them as a model-specific budget independent of the offload weights.

---

## 5. Streams and synchronization

Two CUDA streams are involved:

- **Default stream** — executes all kernels (the actual transformer compute).
- **Side stream** — executes all H2D copies for block groups (created by `torch.cuda.Stream()` once, reused).

The unmatched_group's transfers run on the **default stream** (because its `stream=None`). They are synchronous within the default stream timeline — no overlap with anything.

Block group transfers on the side stream:
- Each `onload_()` issues a sequence of `pinned_tensor.to(device, non_blocking=True)` calls in a fixed (insertion-order) iteration over `cpu_param_dict`. All issued on the side stream.
- Immediately after each tensor's `.to()`, `tensor.data.record_stream(default_stream)` registers the cross-stream dependency.
- Because of `record_stream=True`, no host-side `synchronize()` is needed at eviction time. The allocator handles deferred reclamation.

Synchronization points that DO fire:
- `unmatched_group.onload_()` — synchronous (default stream, blocks until complete).
- `unmatched_group.offload_()` — synchronous.
- (None for block groups under steady-state.)

Synchronization point that fires *only at chain tail*: at the last group's `pre_forward`, the code checks `should_synchronize = not onload_self and stream is not None and not should_onload_next_group`. For the tail (`next_group is None`), this is True, and the side stream is synced before the forward proceeds.

---

## 6. What is deterministic vs not

### Deterministic (given fixed model + seed)

- The set of 25 groups and their members.
- The prefetch chain (after warmup).
- The CPU `cpu_param_dict` iteration order within each group.
- The ordered `(action, group_id, bytes)` event sequence per generation step.
- The total bytes loaded and evicted per step.
- The peak resident-bytes value.

Empirically confirmed: two consecutive steady-state runs produced **byte-identical** event sequences.

### Not deterministic (wall-clock only)

- Wall-clock timestamp of each event (small jitter, on the order of ms).
- Exact moment the allocator reclaims a freed-but-recorded block.
- Whether a new allocation hits the cached pool or grows it.
- Cross-stream completion ordering (which stream finishes its current op first when running concurrently).

Sources of wall-clock jitter:
1. GPU SM scheduler decisions when multiple streams have ready work.
2. `cudaMallocAsync` cache state.
3. PCIe contention with other host traffic.
4. Kernel launch overhead accumulation.

None of these reorder events; they only shift timestamps.

---

## 7. Empirical reference numbers

Measured on RTX 3090, SD3 Medium, fp16, prompt `"A cinematic photo of …"`, `num_inference_steps=2`, `guidance_scale=7.0`, seed=0. Two transformer forwards per inference step (classifier-free guidance), so 4 transformer forwards total per measured run. Numbers are from a steady-state run (post-warmup); two consecutive steady-state runs produced identical sequences.

| Metric | Value |
|---|---:|
| Elapsed (wall-clock) | ~1.49 s (varies ±10 ms across runs) |
| Events recorded | 150 |
| Onloads | 50 |
| Offloads | 50 |
| Peak resident groups | 3 |
| Peak resident bytes | 644 MB |
| Total bytes onloaded | 8.64 GB |
| Median onload size | 169.96 MB |

Per-forward sanity check: 4 forwards × (1 unmatched_group onload + ~12 block onloads observed per forward) ≈ 50 onloads. Refer to the JSON log (`locked-config-trace.json`, field `steady_state_A.events`) for the exact per-event sequence.

---

## 8. Simulator implementation guide

### State

```text
ModuleGroup:
    id: stable string (e.g. "unmatched", "block_0", …, "block_23")
    bytes: int (sum of member weights)
    members: list of module references
    onload_leader: module ref
    offload_leader: module ref
    onload_self: bool
    next_group: ref to next group or None
    stream: side stream | None
    location: "cpu" | "gpu" | "in-flight"
```

### Build pipeline

1. **Group construction (static):** walk `transformer.named_children()`. For each `ModuleList`/`Sequential` element, create one group. Bundle the rest into `unmatched_group`. Set leaders.
2. **Pre-pin CPU master copy:** mark every block-group member's weights as `pinned`. Mark unmatched_group's weights as `pageable`.
3. **Build prefetch chain (static — your simulator deviates from Diffusers's lazy approach):** `unmatched → block_0 → … → block_23`. Set `onload_self = True` on `unmatched`, `onload_self = False` on every block.
4. **Initial offload:** all groups in state "cpu", GPU empty.

### Per-forward state machine

Issue events in this order at each transformer forward call:

```text
forward_enter(transformer)
onload(unmatched_group, stream=DEFAULT, sync=True)
  → unmatched.location = "gpu"

for i in range(24):
    pre_forward(block_i):
        if i == 0:
            # First block — prefetched by unmatched? No, unmatched is the head
            # but its pre_forward doesn't prefetch (its next_group machinery
            # didn't fire because the chain was set up by the lazy hook). In
            # Diffusers steady state, block_0 gets prefetched at the same time
            # the unmatched_group's onload runs, because the unmatched_group's
            # pre_forward (= transformer's pre_forward) handles `should_onload_next_group`.
            onload(block_0, stream=SIDE, sync=False)  ← actually issued by transformer's pre_forward
        if i < 23:
            onload(block_{i+1}, stream=SIDE, sync=False)
        record_stream(block_i tensors on DEFAULT)
        (wait for block_i to be ready if not yet — usually it is)

    compute(block_i) on DEFAULT

    post_forward(block_i):
        offload(block_i, stream=SIDE, sync=False)
        → block_i.location = "cpu" (logically; physically deferred by allocator)

post_forward(transformer):
    offload(unmatched_group, stream=DEFAULT, sync=True)
    → unmatched.location = "cpu"
```

### Allocator / record_stream model

Track per-GPU-block:
- `allocated_at_stream`: which stream allocated it (always SIDE for block groups, DEFAULT for unmatched).
- `recorded_streams`: set of streams that have called `record_stream` on it.
- `freed_logical`: bool — has the program logically freed it?

The block is **reclaimable** when `freed_logical` AND every recorded stream has progressed past the recording event. Until then, the memory counts toward resident.

This is what makes `record_stream=True` semantically meaningful: deferred reclamation.

### Validation

Diff your simulator's output against `locked-config-trace.json`'s `steady_state_A.events` field:

```python
sim_events  = [(e['action'], e['group_id'], e['bytes']) for e in sim_run]
real_events = [(e['action'], e['group_id'], e['bytes']) for e in trace['steady_state_A']['events']]
assert sim_events == real_events
```

If your sim's sequence matches the reference, your group construction, chain, and event firing are correct.

---

## 9. Tunable parameters in real Diffusers under this configuration

**Under the locked configuration, no offload-algorithm parameter remains tunable in real Diffusers.** Every parameter in section 1 is pinned, and the two that streams would have overridden anyway (`num_blocks_per_group → 1`, `non_blocking → True`) are explicitly locked at those forced values. The algorithmic knobs that look tunable in the broader API — prefetch depth, side-stream count, eviction policy, group-construction rule — are not exposed by `apply_group_offloading` at all; they're hard-coded inside `group_offloading.py`.

What remains variable is purely structural / deployment-level, not algorithmic:

| What can vary | What it changes | What stays the same |
|---|---|---|
| Target module passed to `apply_group_offloading` (`pipe.transformer` vs another component) | Number of groups (= number of `ModuleList`/`Sequential` children + 1 unmatched_group); per-group size | The algorithm, the chain construction, the steady-state schedule pattern |
| `onload_device` (e.g. `cuda:0` vs `cuda:1`) | Which physical GPU receives transfers | The host↔device transfer mechanics |

The first row matters for the simulator only insofar as you may want to run the same simulator against different target modules (e.g. a UNet instead of a DiT). The second row is irrelevant to the algorithm.

If you ever want to explore knobs beyond this (group size > 1, deeper prefetch, multiple streams, alternative eviction policies), that's strictly simulator-side work — Diffusers won't let you set them.

---

## 10. One-line summary

**For the locked config, the steady-state schedule is: load the unmatched_group synchronously on the default stream, then iterate over the 24 transformer blocks one at a time, prefetching the next block on the side stream while computing the current block, evicting each block immediately after its forward returns, and finally evicting the unmatched_group at the end of the transformer forward. The schedule is deterministic in sequence; only wall-clock timestamps vary.**
