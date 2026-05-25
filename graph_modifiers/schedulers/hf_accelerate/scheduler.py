"""HF accelerate-style module offload scheduler.

Mimics what ``accelerate.cpu_offload_with_hook`` /
``pipe.enable_sequential_cpu_offload`` does at runtime: weights for each
``nn.Module`` live in RAM and are staged to VRAM right before the
module's forward pass; after the forward they become evictable. With
``lookahead=k``, prefetch fires k modules in advance so the H2D
overlaps with the preceding modules' compute (matches the
``prev_module_hook`` chain accelerate sets up between submodules).

Operates on the eager-mode pytorch profile bundle (no compile sidecars
required). Module-level grouping is read from the per-node
``module_path`` annotation the bundle emits when ``with_modules=True``
is set on torch profiler.

Ownership of a weight tensor is derived from its **GPU consumers'
module_path tags**, not from kernel tagging walks: a weight is owned
by the deepest param-carrying module that appears among the
``module_path`` of any GPU node consuming it. This survives the
profiler's tendency to attribute fused kernels to the parent module
(e.g. a kernel reading ``q_proj.weight`` may be tagged
``self_attn``); the weight still resolves to the right leaf via its
other consumers (and even when only the parent tag is available, we
fall back to the unique descendant-leaf rule).

Output is a self-contained JSON consumable by the loader's
``inject_eager_schedule_path`` knob. The injector populates
``trace.args["xfer_arrivals"]`` and ``trace.args["evict_after_node"]``
and retargets streamed WEIGHT/LEAF tensors' device from cuda → cpu so
``DeviceAwareVanillaAsync`` streams them on demand.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sim.core.trace import Trace


_GPU_ROLES = ("gpu_runtime", "gpu_stream", "gpu")


# (block_paths, unmatched_components) — see `_block_paths_from_hierarchy`.
# We pass these together everywhere the granularity="block" path runs
# rather than threading two parallel arguments through every helper.
BlockInfo = tuple[list[str], set[str]]


@dataclass
class HFAccelerateKnobs:
    """Tunables for the module-offload planner.

    Attributes
    ----------
    lookahead : int
        Number of modules to prefetch in advance. ``0`` = synchronous
        (prefetch issued just before the module starts; full H2D cost
        on the critical path). ``1`` (accelerate's default behavior
        via ``prev_module_hook``) = prefetch one module ahead; ``N`` =
        N modules ahead. Default 1.
    granularity : str
        ``"leaf"`` (default): each parameter-carrying nn.Module is its
        own offload unit. Matches ``enable_sequential_cpu_offload``.
        ``"depth:<N>"`` (e.g. ``"depth:3"``): group modules at path
        depth N and offload each group as a unit (closer to
        ``enable_model_cpu_offload``).
    keep_substrings : tuple[str, ...]
        Module path substrings that should stay cuda-resident (cold
        start, never offloaded). Useful for embeddings / running
        buffers you don't want to stream.
    include_buffers : bool
        Whether to also stream buffer-only modules (``module_has_buffers``
        but no parameters). Default False — matches accelerate's
        ``offload_buffers=False`` default.
    sync_calls : bool
        When True (default), models real accelerate's
        ``cudaStreamSynchronize`` after each H2D by gating the
        **cpu dispatcher** (cudaLaunchKernel of the GPU consumer) on the
        prefetch tids instead of the GPU consumer itself. Because the
        cpu compute device is single-slot, this stalls the host-side
        launch chain until the H2D completes — the next module cannot
        even queue its kernels. Set False to gate only the GPU consumer
        (host can keep issuing other launches while the H2D is in
        flight), which is the previous behavior. Default True because
        the real HF accelerate profile shows 2657
        ``cudaStreamSynchronize`` per 2405 ``cudaMemcpyAsync`` —
        effectively synchronous H2D from the host's perspective.
    emit_d2h_evict : bool
        When True, emits a ``d2h_xfer_arrivals`` entry for each streamed
        tid_run that fires a VRAM→RAM ``TransferJob`` after the
        run's last GPU consumer retires. Models the per-parameter
        ``cudaMemcpyAsync`` D2H that real ``enable_model_cpu_offload``
        and ``cpu_offload_with_hook + prev_module_hook`` issue when
        moving a component back to CPU via ``module.to("cpu")``. The
        D2H consumes RAM↔VRAM bandwidth and delays the next module's
        H2D when both share the same PCIe lane. Default False because
        ``enable_sequential_cpu_offload`` (sequential/module modes)
        uses ``AlignDevicesHook`` which preserves the cpu copy and
        releases the gpu copy via the caching allocator — no DtoH.
        ``main.py`` auto-enables this knob for ``--mode model`` and
        ``--mode module-hook`` based on which real HF API they map to.
    """
    lookahead: int = 1
    granularity: str = "leaf"
    keep_substrings: tuple[str, ...] = field(default_factory=tuple)
    include_buffers: bool = False
    sync_calls: bool = True
    emit_d2h_evict: bool = False
    # Optional path to a `module_hierarchy.json` describing the model's
    # nn.Module tree with class names. When `--mode group` (block_level
    # offload) is selected and this is provided, the solver derives
    # block boundaries directly by walking the hierarchy and finding
    # immediate children of `ModuleList` / `Sequential` containers —
    # matching diffusers' `apply_group_offloading(offload_type=
    # "block_level", num_blocks_per_group=1)` semantics. Without it,
    # group mode falls back to a depth-based heuristic that may
    # mis-group for unfamiliar architectures.
    module_hierarchy_path: str | None = None
    num_blocks_per_group: int = 1
    # When True, omit D2H emission for the LAST temporal unit in the
    # offload chain — its tids stay on GPU after the pipeline finishes.
    # Matches `accelerate.cpu_offload_with_hook` chain semantics: the
    # `prev_module_hook` fires offload of component N-1 when N starts,
    # so the *last* component (no successor) is never offloaded back
    # to CPU. Auto-enabled by `--mode module-hook`.
    #
    # Conversely, diffusers' `enable_model_cpu_offload` (`--mode
    # model`) calls `pipe.maybe_free_model_hooks` at the end of every
    # `__call__`, which invokes `self.to("cpu")` and offloads the
    # last component too — so `--mode model` keeps this knob False and
    # emits D2H for all units.
    omit_last_unit_d2h: bool = False


def _is_gpu_node(node: Any) -> bool:
    rk = str((node.args or {}).get("resource_kind") or "")
    return rk in _GPU_ROLES


def _build_cpu_by_corr_id(trace: Trace) -> dict[int, int]:
    """Index CPU dispatch nodes by their kineto ``correlation_id``.

    Used to bridge gpu_stream nodes back to the CPU dispatcher that
    submitted them. The bundle's gpu_stream nodes carry no
    ``module_path`` of their own, and the loader's
    ``_add_temporal_data_control_edges`` no longer wires CPU→GPU as a
    parent_nodes edge (it expresses tensor flow, and many GPU kernels
    don't consume a tensor produced by the CPU op that launches them).
    Pairing on ``correlation_id`` is the kineto-provided link that
    actually identifies "this CPU op submitted this GPU kernel."

    Multiple CPU nodes can share one correlation_id (typically the
    leaf-most aten dispatch *and* its enclosing cudaLaunchKernel /
    cudaMemcpyAsync). We prefer the one with ``runtime_role=="submit"``
    (the actual cuda runtime call) since that's what cg-sim treats as
    the "dispatcher" elsewhere — and its retire moment is the natural
    issuer for sync prefetches.
    """
    out: dict[int, int] = {}
    out_is_submit: dict[int, bool] = {}
    for nid, n in trace.node_map.items():
        a = n.args or {}
        rk = a.get("resource_kind")
        if rk not in ("cpu_thread", "cpu_leaf"):
            continue
        cid = a.get("correlation_id")
        if cid is None:
            continue
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        is_submit = str(a.get("runtime_role") or "") == "submit"
        prior_is_submit = out_is_submit.get(cid_int, False)
        if cid_int not in out:
            out[cid_int] = int(nid)
            out_is_submit[cid_int] = is_submit
        elif is_submit and not prior_is_submit:
            out[cid_int] = int(nid)
            out_is_submit[cid_int] = True
    return out


def _node_module_path(
    trace: Trace,
    node: Any,
    cpu_by_corr: dict[int, int] | None = None,
) -> str:
    """Return the module_path annotation for a node.

    GPU stream nodes carry no ``module_path`` of their own — the
    attribution lives on the CPU dispatcher (cudaLaunchKernel / aten
    dispatch) that submitted the kernel. After the loader fix that
    excludes alias/in-place nodes from temporal data-control edges,
    ``parent_nodes`` of a gpu_stream node points at preceding
    gpu_stream nodes, not the CPU dispatcher — so we look up the
    matching CPU node via kineto's ``correlation_id`` index and read
    its own ``module_path`` (walking that CPU's parents if needed
    when the dispatcher itself sits below a tagged forward).
    """
    own_mp = (node.args.get("module") or {}).get("module_path") if node.args else None
    if own_mp:
        return str(own_mp)
    # Bridge gpu_stream → cpu via correlation_id.
    a = node.args or {}
    if a.get("resource_kind") == "gpu_stream" and cpu_by_corr:
        cid = a.get("correlation_id")
        try:
            cid_int = int(cid) if cid is not None else None
        except (TypeError, ValueError):
            cid_int = None
        if cid_int is not None and cid_int in cpu_by_corr:
            cpu_id = cpu_by_corr[cid_int]
            cpu = trace.node_map.get(cpu_id)
            if cpu is not None:
                cpu_mp = (cpu.args.get("module") or {}).get("module_path") if cpu.args else None
                if cpu_mp:
                    return str(cpu_mp)
                # Walk the dispatcher's parents until we find a tagged
                # forward (cudaLaunchKernel itself often carries no
                # module_path; its aten parent does).
                cur = cpu
                hops = 0
                while cur is not None and hops < 12:
                    par_id = next(iter(cur.parent_nodes or []), None)
                    if par_id is None:
                        break
                    par = trace.node_map.get(int(par_id))
                    if par is None:
                        break
                    par_rk = (par.args.get("resource_kind") if par.args else "") or ""
                    if par_rk not in ("cpu_thread", "cpu_leaf"):
                        break
                    par_mp = (par.args.get("module") or {}).get("module_path") if par.args else None
                    if par_mp:
                        return str(par_mp)
                    cur = par
                    hops += 1
    # Legacy fallback: walk parent_nodes (works on bundles where the
    # loader still wires CPU→GPU as a parent edge).
    seen: set[int] = set()
    frontier = list(node.parent_nodes or [])
    while frontier:
        pid = frontier.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        p = trace.node_map.get(pid)
        if p is None:
            continue
        p_mp = (p.args.get("module") or {}).get("module_path") if p.args else None
        if p_mp:
            return str(p_mp)
        rk = (p.args.get("resource_kind") if p.args else "") or ""
        if rk in ("cpu_thread", "cpu_leaf"):
            for gp in (p.parent_nodes or []):
                if gp not in seen:
                    frontier.append(gp)
        if len(seen) > 8:
            break
    return ""


_CONTAINER_CLASSES = ("ModuleList", "Sequential", "ModuleDict")
_UNMATCHED_SUFFIX = ".<unmatched>"


def _block_paths_from_hierarchy(
    hierarchy: dict[str, Any],
    num_blocks_per_group: int = 1,
) -> tuple[list[str], set[str]]:
    """Mirror diffusers' ``apply_group_offloading(offload_type=
    "block_level", num_blocks_per_group=N)`` grouping.

    Real ``_apply_group_offloading_block_level`` walks **only the
    immediate children** of the module passed in. Children that are
    ``ModuleList`` / ``Sequential`` containers become block groups (one
    per element, grouped by ``num_blocks_per_group``); everything else
    is collected into a single ``unmatched_group`` for that module that
    is itself offloaded as one unit.

    The cg-sim hierarchy root is the *pipe* (e.g. ``StableDiffusionXLPipeline``
    or ``LlamaForCausalLM``) and its immediate children are the offload
    targets (the modules the harness calls ``apply_group_offloading`` on
    individually — ``unet`` / ``text_encoder`` / ``vae`` / ``transformer``
    / ``model`` / ``lm_head`` / etc.). We replicate the block-level rule
    inside each of those children.

    Returns:
      block_paths — explicit ModuleList block prefixes
        (e.g. ``unet.down_blocks.0``).
      unmatched_components — names of top-level components that have an
        unmatched_group. Any weight under a component in this set that
        does NOT prefix-match a block_path is owned by the synthetic
        unit ``"{component}.<unmatched>"``.
    """
    block_paths: list[str] = []
    unmatched_components: set[str] = set()
    n = max(1, num_blocks_per_group)

    root_children = hierarchy.get("children") or {}
    if not root_children:
        # Edge case: hierarchy IS the single offload target.
        unmatched_components.add(str(hierarchy.get("class", "<root>")))
        return block_paths, unmatched_components

    for comp_name, comp in root_children.items():
        comp_children = comp.get("children") or {}
        if not comp_children:
            # Leaf component (e.g. lm_head Linear). Treat as one
            # unmatched_group containing the whole component.
            unmatched_components.add(comp_name)
            continue
        has_unmatched_child = False
        for child_name, child in comp_children.items():
            cls = str(child.get("class", ""))
            child_path = f"{comp_name}.{child_name}"
            if cls in _CONTAINER_CLASSES:
                grand_names = list((child.get("children") or {}).keys())
                for i in range(0, len(grand_names), n):
                    chunk = grand_names[i : i + n]
                    if not chunk:
                        continue
                    block_paths.append(f"{child_path}.{chunk[0]}")
            else:
                has_unmatched_child = True
        if has_unmatched_child:
            unmatched_components.add(comp_name)

    return block_paths, unmatched_components


def _block_unit_key(
    module_path: str,
    block_paths: list[str],
    unmatched_components: set[str] | None = None,
) -> str | None:
    """Find the longest block_path prefix; fall back to the unmatched
    group of the top-level component the path is under.

    Returns ``None`` only if the path is not under any known component
    (real ``apply_group_offloading`` would never touch such a weight —
    caller should cold-start it).
    """
    if not module_path:
        return None
    # 1. Longest explicit block_path prefix.
    best: str | None = None
    for bp in block_paths:
        if module_path == bp or module_path.startswith(bp + "."):
            if best is None or len(bp) > len(best):
                best = bp
    if best is not None:
        return best
    # 2. Component unmatched_group fallback.
    if unmatched_components:
        comp = module_path.split(".", 1)[0] if "." in module_path else module_path
        if comp in unmatched_components:
            return f"{comp}{_UNMATCHED_SUFFIX}"
    return None


def _module_unit_key(
    module_path: str,
    granularity: str,
    block_info: BlockInfo | None = None,
) -> str | None:
    """Map a module path to its offload unit under the granularity.

    ``leaf`` → the path itself.
    ``depth:N`` → first N components.
    ``block`` → longest block_path prefix from the supplied
    ``BlockInfo`` (typically derived from ``module_hierarchy.json``),
    falling back to ``"{component}.<unmatched>"`` when the path is
    under a component that has an unmatched_group.
    """
    if not module_path:
        return None
    if granularity == "block":
        if block_info is None:
            return None
        block_paths, unmatched = block_info
        return _block_unit_key(module_path, block_paths, unmatched)
    if granularity == "leaf":
        return module_path
    if granularity.startswith("depth:"):
        try:
            depth = int(granularity.split(":", 1)[1])
        except ValueError:
            return module_path
        if depth <= 0:
            return module_path
        parts = module_path.split(".")
        if len(parts) <= depth:
            return module_path
        return ".".join(parts[:depth])
    return module_path


# ---------------------------------------------------------------------------
# Unit discovery
# ---------------------------------------------------------------------------


def _collect_param_units(
    trace: Trace,
    granularity: str,
    include_buffers: bool,
    block_info: BlockInfo | None = None,
) -> tuple[set[str], dict[str, int]]:
    """Return ``(unit_keys, unit_bytes)`` for parameter-carrying modules.

    ``unit_bytes`` sums each leaf path's ``module_size_bytes`` once.
    """
    seen_leaf_paths: dict[str, set[str]] = defaultdict(set)
    unit_bytes: dict[str, int] = {}
    for node in trace.node_map.values():
        a = node.args or {}
        m = a.get("module") or {}
        mp = m.get("module_path")
        if not mp:
            continue
        has_params = int(m.get("module_has_parameters") or 0)
        has_buffers = int(m.get("module_has_buffers") or 0)
        if not has_params and not (include_buffers and has_buffers):
            continue
        size = int(m.get("module_size_bytes") or 0)
        if size <= 0:
            continue
        unit = _module_unit_key(mp, granularity, block_info)
        if unit is None:
            continue
        if mp in seen_leaf_paths[unit]:
            continue
        seen_leaf_paths[unit].add(mp)
        unit_bytes[unit] = unit_bytes.get(unit, 0) + size
    return set(unit_bytes.keys()), unit_bytes


# ---------------------------------------------------------------------------
# Weight → unit attribution (consumer-derived)
# ---------------------------------------------------------------------------


def _longest_common_prefix(paths: list[str]) -> str:
    """Longest dotted-path prefix shared by all entries.

    Operates on path components, not bytes (so ``a.bcd`` and
    ``a.bef`` share only ``a``).
    """
    if not paths:
        return ""
    splits = [p.split(".") for p in paths]
    out: list[str] = []
    for i in range(min(len(s) for s in splits)):
        parts = {s[i] for s in splits}
        if len(parts) != 1:
            break
        out.append(next(iter(parts)))
    return ".".join(out)


_SHARED = "__shared__"


def _resolve_weight_owner_v2(  # noqa: keep old signature too via inner call
    consumer_paths: list[str],
    unit_set: set[str],
    granularity: str,
    block_info: BlockInfo | None,
) -> str | object | None:
    """Like ``_resolve_weight_owner`` but threads block_info through."""
    if not consumer_paths:
        return None
    direct: set[str] = set()
    for p in consumer_paths:
        unit = _module_unit_key(p, granularity, block_info)
        if unit in unit_set:
            direct.add(unit)
    if len(direct) > 1:
        return _SHARED
    if len(direct) == 1:
        return next(iter(direct))
    lcp = _longest_common_prefix(consumer_paths)
    cur = lcp
    while cur:
        unit = _module_unit_key(cur, granularity, block_info)
        if unit in unit_set:
            return unit
        if "." not in cur:
            break
        cur = cur.rsplit(".", 1)[0]
    if lcp:
        prefix = lcp + "."
        descendants = [u for u in unit_set if u.startswith(prefix)]
        if len(descendants) == 1:
            return descendants[0]
    return None


def _resolve_weight_owner(
    consumer_paths: list[str],
    unit_set: set[str],
    granularity: str,
) -> str | None:
    """Pick a unit owner for a weight given its GPU consumers' paths.

    Returns:
      ``unit`` — string in ``unit_set``: weight belongs to that unit.
      ``_SHARED`` — weight is consumed by multiple disjoint units;
        caller should keep resident (cold-start) rather than try to
        stream it for any one of them.
      ``None`` — no unit could be derived (caller should cold-start
        as a safe fallback).

    Rules, in order:
      1. Gather distinct direct unit hits among the consumer paths.
         If >1 distinct hits → ``_SHARED`` (the weight is consumed by
         multiple modules, e.g. a global scalar). If exactly 1 → that
         unit.
      2. Walk the longest-common-prefix of all consumer paths up
         component-by-component until we hit a unit.
      3. If the LCP has exactly one unit-descendant (the parent the
         kernel was tagged with has a single param child), take it.
      4. Otherwise: ambiguous — return None.
    """
    if not consumer_paths:
        return None
    # 1. Direct hits in unit_set.
    direct: set[str] = set()
    for p in consumer_paths:
        unit = _module_unit_key(p, granularity)
        if unit in unit_set:
            direct.add(unit)
    if len(direct) > 1:
        return _SHARED
    if len(direct) == 1:
        return next(iter(direct))

    # 2. Walk up the LCP.
    lcp = _longest_common_prefix(consumer_paths)
    cur = lcp
    while cur:
        unit = _module_unit_key(cur, granularity)
        if unit in unit_set:
            return unit
        if "." not in cur:
            break
        cur = cur.rsplit(".", 1)[0]

    # 3. Unique descendant of LCP in unit_set.
    if lcp:
        prefix = lcp + "."
        descendants = [u for u in unit_set if u.startswith(prefix)]
        if len(descendants) == 1:
            return descendants[0]

    return None


@dataclass
class _UnitState:
    unit: str
    weight_tids: set[int] = field(default_factory=set)
    gpu_consumers: list[tuple[int, int, int]] = field(default_factory=list)
    """List of (start_ns, end_ns, node_id) over every GPU node consuming
    any weight in ``weight_tids``."""
    weight_bytes: int = 0


def _build_unit_states(
    trace: Trace,
    unit_set: set[str],
    granularity: str,
    block_info: BlockInfo | None = None,
    cpu_by_corr: dict[int, int] | None = None,
) -> tuple[dict[str, _UnitState], dict[int, str], dict[int, list[int]], int]:
    """Walk the trace once, returning per-unit state, weight→unit,
    weight→consumer-node list, and a count of ambiguous weight tensors.
    """
    # Per-weight: list of (consumer_node_id, consumer_module_path, start_ns, end_ns).
    weight_consumers: dict[int, list[tuple[int, str, int, int]]] = defaultdict(list)
    for nid, node in trace.node_map.items():
        if not _is_gpu_node(node):
            continue
        a = node.args or {}
        # In bundles where gpu_stream nodes lack module_path, fall
        # back to the parent cpu_thread / cpu_leaf dispatcher's
        # annotation (cudaLaunchKernel or the dispatching aten op).
        mp = _node_module_path(trace, node, cpu_by_corr)
        s_ns = int(a.get("start_ns") or 0)
        e_ns = int(a.get("end_ns") or s_ns)
        for tid in node.input_tensors or []:
            t = trace.tensor_map.get(int(tid))
            if t is None:
                continue
            ttype = (t.args or {}).get("tensor_type")
            if ttype not in ("WEIGHT", "LEAF"):
                continue
            weight_consumers[int(tid)].append((int(nid), mp, s_ns, e_ns))

    unit_states: dict[str, _UnitState] = {u: _UnitState(unit=u) for u in unit_set}
    weight_to_unit: dict[int, str] = {}
    weight_to_consumer_nodes: dict[int, list[int]] = {}
    ambiguous = 0
    shared = 0
    shared_tids: set[int] = set()
    # Ambiguous tids get a per-tid pseudo-unit. Real HF accelerate
    # moves *every* parameter regardless of profiler attribution — the
    # hook system doesn't depend on profiler tags. We mirror that by
    # treating each unresolved tid as its own one-tensor unit, sized
    # by its actual first/last GPU consumer. Without this, ~400 MB of
    # ambiguous weights stay cuda-resident on SDXL-Turbo, inflating
    # VRAM peak well above real HF's measured ~210 MB.
    _AMBIG_PREFIX = "__ambig__:tid"
    for tid, recs in weight_consumers.items():
        paths = [mp for (_n, mp, _s, _e) in recs if mp]
        owner = _resolve_weight_owner_v2(paths, unit_set, granularity, block_info)
        if owner == _SHARED:
            shared += 1
            shared_tids.add(int(tid))
            continue
        if owner is None:
            if granularity == "block":
                # `apply_group_offloading(block_level)` only touches
                # modules under explicit container_classes (ModuleList /
                # Sequential) or their immediate component siblings.
                # Weights with no derivable container ownership stay
                # cuda-resident in real HF — cold-start them here
                # instead of forcing per-tid streaming.
                shared += 1
                shared_tids.add(int(tid))
                continue
            ambiguous += 1
            owner = f"{_AMBIG_PREFIX}{int(tid)}"
            unit_states[owner] = _UnitState(unit=owner)
        elif (
            granularity == "block"
            and isinstance(owner, str)
            and owner.endswith(_UNMATCHED_SUFFIX)
        ):
            # Real `apply_group_offloading` wraps each top-level
            # component's "unmatched_group" with a *lazy* hook
            # (`_apply_lazy_group_offloading_hook` under
            # `use_stream=True`). The lazy hook fetches on first use
            # within a forward; the matching offload is a pointer
            # swap that drops PyTorch's reference to the GPU storage,
            # but the CUDA caching allocator retains the underlying
            # memory and reuses it across subsequent forwards. Net
            # effect: the unmatched group's bytes are effectively
            # GPU-resident from first use onward. Treat as cold-start
            # so cg-sim's per-burst eviction doesn't churn them in
            # and out — that path under-counts peak relative to real,
            # where the caching allocator never frees these bytes
            # back to the simulator's free pool.
            shared += 1
            shared_tids.add(int(tid))
            continue
        elif owner not in unit_states:
            # Defensive: resolver returned a unit not in unit_set
            # (shouldn't happen but don't crash).
            continue
        weight_to_unit[tid] = owner
        weight_to_consumer_nodes[tid] = sorted({nid for (nid, _, _, _) in recs})
        st = unit_states[owner]
        st.weight_tids.add(tid)
        t_obj = trace.tensor_map.get(int(tid))
        if t_obj is not None:
            st.weight_bytes += int(getattr(t_obj, "size_bytes", 0) or 0)
        for (nid, _, s_ns, e_ns) in recs:
            st.gpu_consumers.append((s_ns, e_ns, nid))

    return (
        unit_states, weight_to_unit, weight_to_consumer_nodes,
        ambiguous, shared_tids,
    )


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


@dataclass
class _TidRun:
    """One contiguous use window for a single weight tensor."""
    tid: int
    unit: str
    first_node: int
    last_node: int
    first_start_ns: int
    last_end_ns: int
    size_bytes: int


def _tid_runs_by_module_burst(
    trace: Trace,
    weight_to_unit: dict[int, str],
    granularity: str,
    block_info: BlockInfo | None = None,
    cpu_by_corr: dict[int, int] | None = None,
) -> list[_TidRun]:
    """Build tid_runs from CPU module-context transitions, NOT a gap
    heuristic.

    Real HF Accelerate decides when to load/evict via PyTorch forward
    hooks on each ``nn.Module``: ``pre_forward`` fires on entry, H2Ds
    the module's weights; ``post_forward`` fires on exit, evicts them.
    No timing threshold involved — it's purely a Python control-flow
    event at module entry/exit.

    With ``with_modules=True`` profile capture, every ``cpu_thread`` /
    ``cpu_leaf`` dispatch event carries the ``module_path`` of the
    enclosing forward. We can recover module forward boundaries by:

    1. Walking CPU events in temporal order.
    2. Mapping each event's ``module_path`` to a unit via the chosen
       granularity (same ``_module_unit_key`` used for unit discovery).
    3. Treating a contiguous run of events mapped to the *same* unit
       as a single forward of that unit (= one load/evict cycle).
       A transition (different unit, or no module_path) is a boundary.

    For autoregressive llama-style 15-token generation, this produces
    15 bursts per leaf module — one per token forward — matching real
    ``enable_sequential_cpu_offload`` exactly. For 1-step SDXL it
    produces 1 burst per leaf, also matching reality. No
    workload-specific tuning.

    For each burst, the function emits one ``_TidRun`` per
    weight/leaf tid that:
      - is owned by the burst's unit (per ``weight_to_unit``), and
      - is consumed by some GPU kernel submitted from a CPU event
        inside the burst.

    The run's ``first_node`` / ``last_node`` are the earliest and
    latest GPU kernels submitted within the burst, defining the
    H2D-arrival issuer (lookahead window) and the D2H-eviction
    anchor respectively.
    """
    # ---- Step 1: index gpu kernels by their cpu submitter ----
    # Two paths because the loader's parent_nodes wiring varies by
    # version: legacy bundles wired gpu_stream.parent_nodes back to the
    # CPU dispatcher (cudaLaunchKernel); the post-alias-fix loader
    # wires them only to preceding gpu_stream nodes. Prefer the
    # correlation_id pairing (kineto-supplied, version-invariant) when
    # a `cpu_by_corr` index is provided; fall back to walking
    # parent_nodes for the legacy layout.
    cpu_to_gpu: dict[int, list[int]] = defaultdict(list)
    for nid, node in trace.node_map.items():
        if not _is_gpu_node(node):
            continue
        cpu_id: int | None = None
        if cpu_by_corr is not None:
            cid = (node.args or {}).get("correlation_id")
            try:
                cid_int = int(cid) if cid is not None else None
            except (TypeError, ValueError):
                cid_int = None
            if cid_int is not None:
                cpu_id = cpu_by_corr.get(cid_int)
        if cpu_id is not None:
            cpu_to_gpu[cpu_id].append(int(nid))
        else:
            for p in node.parent_nodes or []:
                cpu_to_gpu[int(p)].append(int(nid))

    # ---- Step 2: order cpu_thread / cpu_leaf nodes by start_ns ----
    cpu_events: list[tuple[int, int, str]] = []
    for nid, node in trace.node_map.items():
        a = node.args or {}
        rk = a.get("resource_kind") or ""
        if rk not in ("cpu_thread", "cpu_leaf"):
            continue
        mp = (a.get("module") or {}).get("module_path") or ""
        if not mp:
            # No module context — counts as "outside any unit" for
            # boundary purposes. We still record so transitions back
            # into a unit form a new burst.
            cpu_events.append((int(a.get("start_ns") or 0), int(nid), ""))
            continue
        cpu_events.append((int(a.get("start_ns") or 0), int(nid), mp))
    cpu_events.sort()

    # ---- Step 3: form bursts by detecting unit transitions ----
    out: list[_TidRun] = []
    current_unit: str | None = None
    current_cpu_nids: list[int] = []

    def flush(unit: str | None, cpu_nids: list[int]) -> None:
        if not unit or not cpu_nids:
            return
        # Collect GPU kernels submitted by this burst's cpu events.
        gpu_nids: set[int] = set()
        for cnid in cpu_nids:
            gpu_nids.update(cpu_to_gpu.get(cnid, ()))
        if not gpu_nids:
            return
        first_start = None
        last_end = None
        first_node = None
        last_node = None
        tids_in_burst: set[int] = set()
        for gnid in gpu_nids:
            g = trace.node_map.get(gnid)
            if g is None:
                continue
            ga = g.args or {}
            s = int(ga.get("start_ns") or 0)
            e = int(ga.get("end_ns") or s)
            if first_start is None or s < first_start:
                first_start, first_node = s, gnid
            if last_end is None or e > last_end:
                last_end, last_node = e, gnid
            for tid in g.input_tensors or []:
                t = trace.tensor_map.get(int(tid))
                if t is None:
                    continue
                if (t.args or {}).get("tensor_type") not in ("WEIGHT", "LEAF"):
                    continue
                tids_in_burst.add(int(tid))
        if first_node is None or last_node is None:
            return
        for tid in tids_in_burst:
            if weight_to_unit.get(int(tid)) != unit:
                continue
            t_obj = trace.tensor_map.get(int(tid))
            size = int(getattr(t_obj, "size_bytes", 0) or 0) if t_obj else 0
            out.append(_TidRun(
                tid=int(tid), unit=unit,
                first_node=int(first_node), last_node=int(last_node),
                first_start_ns=int(first_start), last_end_ns=int(last_end),
                size_bytes=size,
            ))

    # Real HF Accelerate's `cpu_offload_with_hook` chain keeps a
    # component resident on GPU across short Python-only gaps between
    # consecutive forwards of the same component — for SDXL that
    # means unet stays on GPU for all 4 sample steps (the scheduler's
    # `step()` between them is plain Python with no `module_path`),
    # for sd3-medium the transformer stays on GPU for all 28 steps,
    # for llama generate-loop the model stays on GPU across all
    # generated tokens. The hook fires on every forward call but
    # is a no-op when the component is already on device.
    #
    # Treat events with no `module_path` (None) as TRANSPARENT to
    # the burst detector: they absorb into the surrounding burst
    # instead of splitting it. Transitions only fire when we see a
    # DIFFERENT named unit, matching the real hook semantics.
    # Per-leaf granularity still gets exact boundaries because
    # consecutive leaf events ARE in named modules (no None blips
    # between adjacent leaves to confuse).
    for _start, nid, mp in cpu_events:
        unit = _module_unit_key(mp, granularity, block_info) if mp else None
        if unit is None:
            if current_unit is not None:
                current_cpu_nids.append(nid)
            continue
        if unit != current_unit:
            flush(current_unit, current_cpu_nids)
            current_unit = unit
            current_cpu_nids = []
        current_cpu_nids.append(nid)
    flush(current_unit, current_cpu_nids)

    out.sort(key=lambda r: (r.first_start_ns, r.unit, r.tid))
    return out


def _find_cpu_dispatcher(
    trace: Trace,
    gpu_node_id: int,
    cpu_by_corr: dict[int, int] | None = None,
) -> int | None:
    """Find a CPU dispatcher (cuLaunchKernel) that submits a GPU node.

    Used as a synchronous-prefetch issuer when no prior GPU window
    exists: the dispatcher retires almost immediately (CPU launch is
    brief), the prefetch then fires, and the GPU consumer waits on
    the transfer's arrival. Prefer the kineto ``correlation_id`` index
    over ``parent_nodes`` because the post-alias-fix loader no longer
    wires gpu_stream nodes back to their CPU dispatcher via parents.
    """
    node = trace.node_map.get(int(gpu_node_id))
    if node is None:
        return None
    if cpu_by_corr is not None:
        cid = (node.args or {}).get("correlation_id")
        try:
            cid_int = int(cid) if cid is not None else None
        except (TypeError, ValueError):
            cid_int = None
        if cid_int is not None:
            mapped = cpu_by_corr.get(cid_int)
            if mapped is not None:
                return int(mapped)
    for parent_id in node.parent_nodes or []:
        parent = trace.node_map.get(int(parent_id))
        if parent is None:
            continue
        role = str((parent.args or {}).get("runtime_role") or "")
        if role == "submit":
            return int(parent_id)
    for parent_id in node.parent_nodes or []:
        parent = trace.node_map.get(int(parent_id))
        if parent is None:
            continue
        rk = str((parent.args or {}).get("resource_kind") or "")
        if rk in ("cpu_thread", "cpu_leaf"):
            return int(parent_id)
    return None


def _find_predecessor_cpu_node(trace: Trace, cpu_node_id: int) -> int | None:
    """Walk one step backward in the cpu thread/leaf chain.

    Used as a fallback issuer when ``sync_calls=True`` would otherwise
    make the prefetch issuer equal the gated consumer (the cpu
    dispatcher), creating a circular wait. Walking back one cpu node
    keeps the prefetch firing strictly *before* the dispatcher tries
    to launch.
    """
    node = trace.node_map.get(int(cpu_node_id))
    if node is None:
        return None
    for parent_id in node.parent_nodes or []:
        parent = trace.node_map.get(int(parent_id))
        if parent is None:
            continue
        rk = str((parent.args or {}).get("resource_kind") or "")
        if rk in ("cpu_thread", "cpu_leaf"):
            return int(parent_id)
    return None


def solve(trace: Trace, knobs: HFAccelerateKnobs) -> dict[str, Any]:
    """Build the eager-mode offload schedule."""
    # Load module hierarchy → block_info if provided (used by
    # granularity="block" to match `apply_group_offloading`'s
    # block_level grouping exactly).
    block_info: BlockInfo | None = None
    if knobs.module_hierarchy_path:
        with open(knobs.module_hierarchy_path) as f:
            hierarchy = json.load(f)
        block_info = _block_paths_from_hierarchy(
            hierarchy, num_blocks_per_group=knobs.num_blocks_per_group,
        )

    unit_set, unit_bytes = _collect_param_units(
        trace, knobs.granularity, knobs.include_buffers,
        block_info=block_info,
    )
    if not unit_set:
        raise RuntimeError(
            "[hf_accelerate] no parameter-carrying modules found in the "
            "trace. Check that the bundle was built with with_modules=True "
            "and that module_path is populated in runtime_nodes.csv."
        )

    # Index CPU dispatchers by kineto correlation_id so gpu_stream
    # nodes can find their submitting CPU op (and its module_path).
    cpu_by_corr = _build_cpu_by_corr_id(trace)

    (
        unit_states, weight_to_unit, weight_consumer_nodes,
        ambiguous, shared_tids,
    ) = _build_unit_states(
        trace, unit_set, knobs.granularity, block_info=block_info,
        cpu_by_corr=cpu_by_corr,
    )

    # Require `with_modules=True` capture. Real HF Accelerate's
    # offload semantics are driven by `nn.Module.__call__` entry/exit
    # hooks, so the only faithful way to recover load/evict points is
    # the module-path annotations PyTorch profiler attaches to each
    # CPU dispatch event. The old gap-threshold fallback (median of
    # inter-consumer gaps × 50, ceiling 100 ms) was a workload-tuned
    # heuristic that drifted on every new model — removed.
    cpu_with_modules = any(
        ((n.args or {}).get("resource_kind") in ("cpu_thread", "cpu_leaf"))
        and ((n.args or {}).get("module") or {}).get("module_path")
        for n in trace.node_map.values()
    )
    if not cpu_with_modules:
        raise RuntimeError(
            "[hf_accelerate] trace bundle has no `module_path` on CPU "
            "dispatch events — re-capture the profile with "
            "`with_modules=True` (PyTorch profiler) so we can derive "
            "load/evict boundaries from real nn.Module forward "
            "entry/exit, not from a timing heuristic."
        )
    tid_runs = _tid_runs_by_module_burst(
        trace, weight_to_unit, knobs.granularity, block_info,
        cpu_by_corr=cpu_by_corr,
    )
    if not tid_runs:
        raise RuntimeError(
            "[hf_accelerate] no weight-tensor runs derived — nothing "
            "to schedule."
        )

    keep_set: set[str] = set()
    if knobs.keep_substrings:
        for unit in unit_set:
            for sub in knobs.keep_substrings:
                if sub and sub in unit:
                    keep_set.add(unit)
                    break

    lookahead = max(0, int(knobs.lookahead))

    # Last unit in the temporal offload chain — its tid_runs end after
    # every other unit's. Used by omit_last_unit_d2h (--mode
    # module-hook) to skip emitting D2H for this unit, matching real
    # cpu_offload_with_hook chain semantics where the last component
    # has no prev_module_hook successor so never gets offloaded.
    #
    # Pick from real component units only — ambiguous-ownership weights
    # (consumed by multiple disjoint modules, unit named like
    # ``__ambig__:tidN``) aren't a chain participant and can have the
    # latest last_end_ns by coincidence.
    last_unit_name: str | None = None
    if knobs.emit_d2h_evict and knobs.omit_last_unit_d2h:
        real_unit_runs = [r for r in tid_runs if not r.unit.startswith("__ambig__")]
        if real_unit_runs:
            last_unit_name = max(real_unit_runs, key=lambda r: r.last_end_ns).unit

    cold_start_tids: set[int] = set()
    streamed_tids: set[int] = set()

    cold_start_bytes = 0
    streamed_bytes = 0

    # Shared weights (consumed by multiple disjoint units) stay
    # cuda-resident.
    for tid in shared_tids:
        t_obj = trace.tensor_map.get(int(tid))
        if t_obj is None:
            continue
        cold_start_tids.add(int(tid))
        cold_start_bytes += int(getattr(t_obj, "size_bytes", 0) or 0)

    # Pre-compute run end-times sorted for fast issuer lookup.
    sorted_by_end = sorted(
        enumerate(tid_runs), key=lambda kv: (kv[1].last_end_ns, kv[0])
    )
    end_times = [r.last_end_ns for (_i, r) in sorted_by_end]
    sorted_indices = [i for (i, _r) in sorted_by_end]

    from bisect import bisect_left
    n_ordering_violations = 0

    # Group prefetches with same (issuer, consumer) to share an arrival.
    arrival_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    evict_after_node: dict[int, set[int]] = defaultdict(set)
    # D2H eviction arrivals (model / module-hook only). Issuer is the
    # tid_run's last GPU node — when it retires, fire a VRAM→RAM
    # TransferJob for the streamed tid; the scheduler-side handler
    # releases the VRAM region after the D2H retires.
    d2h_arrival_by_issuer: dict[int, dict[str, Any]] = {}

    for r in tid_runs:
        if r.tid in cold_start_tids:
            # Shared weights stay resident across the run; record evict
            # would race the next consumer. Skip prefetch & evict.
            continue

        unit = r.unit
        if unit in keep_set:
            if r.tid not in cold_start_tids:
                cold_start_tids.add(r.tid)
                cold_start_bytes += r.size_bytes
            continue

        # Find issuer using a window across all tid-runs.
        if lookahead == 0:
            # Synchronous: prefetch fires when the consumer's CPU
            # dispatcher retires (i.e. just before the consumer needs
            # to run). Full H2D cost lands on the critical path.
            disp = _find_cpu_dispatcher(trace, r.first_node, cpu_by_corr)
            issuer_node_id = disp if disp is not None else r.first_node
        else:
            pos = bisect_left(end_times, r.first_start_ns)
            if pos == 0:
                disp = _find_cpu_dispatcher(trace, r.first_node, cpu_by_corr)
                issuer_node_id = disp if disp is not None else r.first_node
            else:
                target = max(0, pos - lookahead)
                issuer_run = tid_runs[sorted_indices[target]]
                # Default issuer = previous tid_run's last GPU node.
                # With sync_calls=True AND add_temporal_data_control_edges,
                # this can create a cycle: PyTorch profiles record cpu
                # launches with smaller node_ids than the gpu_stream
                # nodes those launches eventually schedule (cpu activity
                # is emitted before its concurrent gpu_stream retires),
                # so `prev_module.last_gpu` may have a *larger* id than
                # `cur_module.dispatcher`. Since tdc adds control edges
                # strictly forward in id (most-recent producer before
                # consumer → consumer), an issuer with id greater than
                # the gated consumer can be transitively waiting on the
                # consumer through some intermediate-tensor temporal
                # edge — deadlock. Routing through the GPU issuer's CPU
                # dispatcher (smaller id, host thread chain) breaks the
                # cycle: dispatcher retires almost immediately so the
                # prefetch still fires early in module M-1's window.
                gpu_issuer = issuer_run.last_node
                if knobs.sync_calls:
                    disp_issuer = _find_cpu_dispatcher(trace, gpu_issuer, cpu_by_corr)
                    issuer_node_id = (
                        disp_issuer if disp_issuer is not None else gpu_issuer
                    )
                else:
                    issuer_node_id = gpu_issuer
                if issuer_run.last_end_ns >= r.first_start_ns:
                    n_ordering_violations += 1
                    disp = _find_cpu_dispatcher(trace, r.first_node, cpu_by_corr)
                    issuer_node_id = disp if disp is not None else r.first_node

        # consumer_node_id selects whose ready-to-fire moment the
        # prefetch tids must gate. Default (sync_calls=True): the cpu
        # dispatcher (cudaLaunchKernel) of the GPU consumer — gating
        # the dispatcher stalls the cpu single-slot host pipeline until
        # the H2D completes, mirroring real accelerate's
        # cudaStreamSynchronize. Fallback: the GPU consumer itself
        # (sync_calls=False, or when no cpu dispatcher is found).
        if knobs.sync_calls:
            disp_consumer = _find_cpu_dispatcher(trace, r.first_node, cpu_by_corr)
            consumer_node_id = (
                disp_consumer if disp_consumer is not None else r.first_node
            )
        else:
            consumer_node_id = r.first_node

        # Avoid circular dependency: if the issuer equals the gated
        # consumer (happens in the pos==0 / lookahead==0 case where
        # _find_cpu_dispatcher resolves to the same node on both sides),
        # the prefetch can never fire because the issuer would have to
        # retire first but is itself parked on the gate. Back off to
        # the cpu node *before* the consumer in the thread chain — it
        # retires unimpeded and fires the prefetch, after which the
        # consumer's gate clears.
        if int(issuer_node_id) == int(consumer_node_id):
            pred = _find_predecessor_cpu_node(trace, consumer_node_id)
            if pred is not None:
                issuer_node_id = pred

        key = (int(issuer_node_id), int(consumer_node_id))
        entry = arrival_by_pair.get(key)
        if entry is None:
            entry = {
                "issuer_node_id": int(issuer_node_id),
                "consumer_node_id": int(consumer_node_id),
                "cgsim_tids": [],
                "size_bytes": 0,
                "reason": "hf_accelerate_prefetch",
                "units": [],
            }
            arrival_by_pair[key] = entry
        entry["cgsim_tids"].append(int(r.tid))
        entry["size_bytes"] += int(r.size_bytes)
        if unit not in entry["units"]:
            entry["units"].append(unit)

        if knobs.emit_d2h_evict and r.unit != last_unit_name:
            # Fire VRAM→RAM TransferJob when r.last_node retires (last
            # GPU consumer of the streamed tid). The scheduler-side
            # handler releases the VRAM region after the D2H completes.
            # Skipped for the last unit when omit_last_unit_d2h is on
            # (--mode module-hook): real `cpu_offload_with_hook` chain
            # leaves the final component's weights resident on GPU
            # after the pipeline finishes, because there's no further
            # `prev_module_hook` to fire its offload.
            issuer_d2h = int(r.last_node)
            d2h_entry = d2h_arrival_by_issuer.get(issuer_d2h)
            if d2h_entry is None:
                d2h_entry = {
                    "issuer_node_id": issuer_d2h,
                    "cgsim_tids": [],
                    "size_bytes": 0,
                    "reason": "hf_accelerate_evict_d2h",
                    "units": [],
                }
                d2h_arrival_by_issuer[issuer_d2h] = d2h_entry
            d2h_entry["cgsim_tids"].append(int(r.tid))
            d2h_entry["size_bytes"] += int(r.size_bytes)
            if unit not in d2h_entry["units"]:
                d2h_entry["units"].append(unit)
        else:
            evict_after_node[int(r.last_node)].add(int(r.tid))

        if r.tid not in streamed_tids:
            streamed_tids.add(int(r.tid))
            streamed_bytes += r.size_bytes

    xfer_arrivals: list[dict[str, Any]] = []
    for entry in arrival_by_pair.values():
        entry["cgsim_tids"].sort()
        entry["reason"] = (
            "hf_accelerate_prefetch:"
            + ",".join(entry.pop("units")[:3])
        )
        xfer_arrivals.append(entry)
    xfer_arrivals.sort(key=lambda a: (a["consumer_node_id"], a["issuer_node_id"]))

    d2h_xfer_arrivals: list[dict[str, Any]] = []
    for entry in d2h_arrival_by_issuer.values():
        entry["cgsim_tids"].sort()
        entry["reason"] = (
            "hf_accelerate_evict_d2h:"
            + ",".join(entry.pop("units")[:3])
        )
        d2h_xfer_arrivals.append(entry)
    d2h_xfer_arrivals.sort(key=lambda a: a["issuer_node_id"])

    unit_order = [
        {
            "tid": r.tid,
            "unit": r.unit,
            "first_node": r.first_node,
            "last_node": r.last_node,
            "first_ns": r.first_start_ns,
            "last_ns": r.last_end_ns,
            "size_bytes": r.size_bytes,
        }
        for r in tid_runs
    ]

    total_param_bytes = sum(unit_bytes.values())
    meta: dict[str, Any] = {
        "io_model": "hf_accelerate",
        "algorithm": (
            "Module-grained offload with per-iteration runs: "
            f"prefetch ~{lookahead} run(s) ahead, evict at each "
            "run's last GPU node."
        ),
        "knobs": {
            "lookahead": lookahead,
            "granularity": knobs.granularity,
            "keep_substrings": list(knobs.keep_substrings),
            "include_buffers": bool(knobs.include_buffers),
            "sync_calls": bool(knobs.sync_calls),
            "emit_d2h_evict": bool(knobs.emit_d2h_evict),
            "omit_last_unit_d2h": bool(knobs.omit_last_unit_d2h),
            "last_unit_resident": last_unit_name,
            "num_blocks_per_group": int(knobs.num_blocks_per_group),
            "block_paths_count": (
                len(block_info[0]) if block_info is not None else None
            ),
            "unmatched_components_count": (
                len(block_info[1]) if block_info is not None else None
            ),
        },
        "unit_count_in_trace": len(unit_set),
        "unit_count_with_gpu_activity": (
            len({r.unit for r in tid_runs})
        ),
        "tid_run_count": len(tid_runs),
        "ambiguous_weight_tensor_count": int(ambiguous),
        "shared_weight_tensor_count": int(len(shared_tids)),
        "ordering_violations": int(n_ordering_violations),
        "total_param_bytes": int(total_param_bytes),
        "total_param_mb": round(total_param_bytes / 1e6, 2),
        "streamed_tensor_count": len(streamed_tids),
        "streamed_bytes_mb": round(streamed_bytes / 1e6, 2),
        "cold_start_tensor_count": len(cold_start_tids),
        "cold_start_bytes_mb": round(cold_start_bytes / 1e6, 2),
        "h2d_ops": len(xfer_arrivals),
        "d2h_ops": (
            sum(len(v) for v in evict_after_node.values())
            + sum(len(e["cgsim_tids"]) for e in d2h_xfer_arrivals)
        ),
        "d2h_xfer_ops": len(d2h_xfer_arrivals),
    }

    # When D2H is emitted, the D2H *is* the eviction — the
    # scheduler's auto-eviction path (driven by evictable_tensor_ids
    # in _consume_inputs) would otherwise release the VRAM region the
    # moment the last consumer retires, beating the D2H fire (which
    # happens in the same tick but after _retire_completed_nodes) and
    # leaving the D2H with nothing to transfer. Excluding D2H tids
    # from evictable_tensor_ids forces the D2H handler to be the sole
    # release point. tids that don't have any D2H arrival (e.g.,
    # shared/cold-start weights) stay in evictable_tensor_ids.
    d2h_tids: set[int] = set()
    for e in d2h_xfer_arrivals:
        d2h_tids.update(e["cgsim_tids"])
    evictable_for_sim = sorted(streamed_tids - d2h_tids)

    return {
        "schema": "hf_accelerate_eager_v1",
        "meta": meta,
        "unit_order": unit_order,
        "xfer_arrivals": xfer_arrivals,
        "d2h_xfer_arrivals": d2h_xfer_arrivals,
        "evict_after_node": {
            str(nid): sorted(tids) for nid, tids in evict_after_node.items()
        },
        "evictable_tensor_ids": evictable_for_sim,
        "cold_start_tids": sorted(cold_start_tids),
        "streamed_tids": sorted(streamed_tids),
    }


def write_eager_schedule(path: str | Path, schedule: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(schedule, f, indent=2)
    return out


def print_summary(schedule: dict[str, Any]) -> None:
    meta = schedule.get("meta", {})
    print()
    print(f"  Variant            : {meta.get('io_model')}")
    print(f"  Knobs              : {meta.get('knobs', {})}")
    print(f"  Module units       : {meta.get('unit_count_in_trace')} "
          f"(with GPU activity: {meta.get('unit_count_with_gpu_activity')})")
    print(f"  Tid runs           : {meta.get('tid_run_count')}")
    print(f"  Total params       : {meta.get('total_param_mb')} MB")
    print(f"  Cold-start         : {meta.get('cold_start_tensor_count')} tensors "
          f"({meta.get('cold_start_bytes_mb')} MB)")
    print(f"  Streamed           : {meta.get('streamed_tensor_count')} tensors "
          f"({meta.get('streamed_bytes_mb')} MB)")
    print(f"  Ambiguous weights  : {meta.get('ambiguous_weight_tensor_count')}")
    print(f"  Shared weights     : {meta.get('shared_weight_tensor_count')}")
    print(f"  Ordering violations: {meta.get('ordering_violations')}")
    print(f"  Prefetch ops       : {meta.get('h2d_ops')}")
    print(f"  Evict ops          : {meta.get('d2h_ops')}")
