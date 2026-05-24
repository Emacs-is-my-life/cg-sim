"""Inject a weight-streaming schedule into a cg-sim Trace.

The injector consumes a *neutral* schedule emitted by
``graph_modifiers/`` (e.g. ``ct_milp_oracle.solve_neutral``). The
legacy pytorch-format ``jit_sim_prune_schedule.json`` is no longer
accepted — it routed through a different tid resolver that inflated
sim peak by ~1.65 GB on SDXL.

It populates two pieces of state on the trace:

  trace.args["xfer_arrivals"] : list[dict]
      Per-prefetch records: ``{issuer_node_id, consumer_node_id,
      cgsim_tids, duration_ns}``. ``DeviceAwareVanillaAsync`` reads this
      to fire ``sys.transfer`` on issuer-retire and gate the consumer
      until those transfers complete.

  trace.args["evict_after_node"] : dict[int, set[int]]
      Per-evict records: when ``node_id`` retires, free those cgsim
      tensors' VRAM regions. The base DAV scheduler already honors this
      map.

It also retargets every non-cold-start WEIGHT/LEAF tensor's
``args["device"]`` from cuda → cpu so DAV's layout places them in RAM
only — the schedule's prefetches then move them in on demand.

This module is the cg-sim analogue of
``torch/_inductor/codegen/wrapper.py:_inject_weight_streaming_io``: the
PyTorch wrapper inserts ``ws_ops`` calls at kernel boundaries; we
emit annotation lists the cg-sim async scheduler consumes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sim.core.trace import Trace


# --------------------------------------------------------------------- helpers


def _build_lid_to_node_id(trace: Trace) -> dict[tuple[int, int], int]:
    """Map (compiled_graph_id, compiled_launch_id) → GPU node_id.

    Each (gid, lid) appears in the trace as TWO nodes — a
    ``cpu_thread`` cuLaunchKernel dispatcher (no input tensors) and a
    ``gpu_stream``/``gpu_runtime`` kernel (the actual consumer of any
    weight tensor). We want the GPU node for both prefetch issuer and
    consumer anchoring: the dispatcher reads no tensors, so gating it
    on a weight's residency would deadlock the CPU pipeline without
    reason.

    With ``replicate_uses`` enabled, the same (gid, lid) appears in
    multiple iterations; we keep the FIRST occurrence (= iter 0). For
    cross-iter prefetches, that's the issuer for iter 0 and consumer
    for iter 0; iter N's instances are reached through the
    control-graph naturally.
    """
    out: dict[tuple[int, int], int] = {}
    for nid, node in trace.node_map.items():
        gid_raw = node.args.get("compiled_graph_id")
        lid_raw = node.args.get("compiled_launch_id")
        if gid_raw is None or lid_raw is None:
            continue
        try:
            gid = int(gid_raw)
            lid = int(lid_raw)
        except (TypeError, ValueError):
            continue
        if lid < 0:
            continue
        rk = str(node.args.get("resource_kind") or "")
        if rk not in ("gpu_stream", "gpu", "gpu_runtime"):
            continue
        key = (gid, lid)
        if key in out:
            continue
        out[key] = nid
    return out


def _valid_gpu_node_id(trace: Trace, node_id: int) -> int | None:
    """Return ``node_id`` iff it names a GPU compute node in ``trace``."""
    if node_id < 0:
        return None
    node = trace.node_map.get(int(node_id))
    if node is None:
        return None
    rk = str(node.args.get("resource_kind") or "")
    if rk not in ("gpu_stream", "gpu", "gpu_runtime"):
        return None
    return int(node_id)


def _set_h2d_streams_from_schedule(trace: Trace, doc: dict[str, Any]) -> None:
    """Pass scheduler H2D FIFO width to DeviceAwareVanillaAsync when known."""
    meta = doc.get("meta") or doc.get("summary") or {}
    knobs = meta.get("knobs") if isinstance(meta, dict) else None
    if not isinstance(knobs, dict) or "n_channels" not in knobs:
        return
    try:
        n_channels = max(1, int(knobs["n_channels"]))
    except (TypeError, ValueError):
        return
    trace.args["xfer_h2d_streams"] = n_channels


def _build_lid_to_gpu_node_ids(
    trace: Trace,
) -> dict[tuple[int, int], list[int]]:
    """Like ``_build_lid_to_node_id`` but only GPU nodes, all instances.

    A (gid, lid) pair has both a CPU dispatcher (``cpu_thread`` /
    ``cuLaunchKernel``, no input tensors) and a GPU kernel
    (``gpu_stream`` / ``gpu_runtime``, with the actual inputs). We need
    the GPU nodes when intersecting WEIGHT inputs to resolve
    ``(gid, ctid) → cgsim_tid``.
    """
    out: dict[tuple[int, int], list[int]] = defaultdict(list)
    for nid, node in trace.node_map.items():
        gid_raw = node.args.get("compiled_graph_id")
        lid_raw = node.args.get("compiled_launch_id")
        if gid_raw is None or lid_raw is None:
            continue
        try:
            gid = int(gid_raw)
            lid = int(lid_raw)
        except (TypeError, ValueError):
            continue
        if lid < 0:
            continue
        rk = str(node.args.get("resource_kind") or "")
        if rk not in ("gpu_stream", "gpu", "gpu_runtime"):
            continue
        out[(gid, lid)].append(nid)
    return out


def _shape_product(shape: Any) -> int | None:
    """Total element count from a shape list (entries may be str or int)."""
    if not isinstance(shape, (list, tuple)):
        return None
    n = 1
    try:
        for d in shape:
            n *= int(d)
    except (TypeError, ValueError):
        return None
    return n


_DTYPE_BYTES = {
    "torch.float16": 2, "torch.bfloat16": 2, "torch.half": 2,
    "torch.float32": 4, "torch.float": 4,
    "torch.float64": 8, "torch.double": 8,
    "torch.int8": 1, "torch.uint8": 1, "torch.bool": 1,
    "torch.int16": 2,
    "torch.int32": 4,
    "torch.int64": 8, "torch.long": 8,
}


def _bytes_for(numel: int | None, dtype: Any) -> int | None:
    if numel is None:
        return None
    nbytes = _DTYPE_BYTES.get(str(dtype) if dtype is not None else None)
    if nbytes is None:
        return None
    return numel * nbytes


def _resolve_tids_via_metas(
    trace: Trace,
    tensor_metas: dict[tuple[int, int], dict[str, Any]],
) -> dict[tuple[int, int], list[int]]:
    """Resolve ``(gid, ctid) → cgsim_tid list`` using authoritative
    metadata (``used_by_launch_ids``, ``shape``).

    For each (gid, ctid):
      1. Intersect the WEIGHT/LEAF inputs across the GPU nodes at
         every consumer launch in ``used_by_launch_ids``.
      2. If multiple cgsim tensors remain, disambiguate by shape
         element-count (or ``size_bytes`` if shape unavailable).
    """
    lid_to_gpu = _build_lid_to_gpu_node_ids(trace)
    out: dict[tuple[int, int], list[int]] = {}
    n_disamb = 0
    n_ambig = 0
    for (gid, ctid), meta in tensor_metas.items():
        launches = meta.get("used_by_launch_ids") or []
        if not launches:
            continue
        common: set[int] | None = None
        for lid in launches:
            per_launch: set[int] = set()
            for nid in lid_to_gpu.get((gid, int(lid)), []):
                node = trace.node_map[nid]
                for t_id in node.input_tensors:
                    tensor = trace.tensor_map.get(t_id)
                    if tensor is None:
                        continue
                    if tensor.args.get("tensor_type") not in ("WEIGHT", "LEAF"):
                        continue
                    per_launch.add(t_id)
            if common is None:
                common = per_launch
            else:
                common &= per_launch
            if not common:
                break
        if not common:
            continue
        if len(common) == 1:
            out[(gid, ctid)] = list(common)
            continue
        candidates = sorted(common)
        target_numel = _shape_product(meta.get("shape"))
        target_bytes = meta.get("size_bytes")
        target_dtype = meta.get("dtype")
        # First try numel match (works when shape is in meta).
        if target_numel is not None:
            filt = [
                t for t in candidates
                if _shape_product(trace.tensor_map[t].args.get("shape"))
                == target_numel
            ]
            if filt:
                if len(filt) < len(candidates):
                    n_disamb += 1
                out[(gid, ctid)] = filt
                continue
        # Neutral has no shape: fall back to size_bytes, computed from
        # candidate's (shape × dtype). Compare against meta.size_bytes
        # (when present) or numel × meta.dtype_bytes.
        if target_bytes is not None or target_dtype is not None:
            filt = []
            for t in candidates:
                cand_numel = _shape_product(trace.tensor_map[t].args.get("shape"))
                cand_dtype = trace.tensor_map[t].args.get("dtype") or target_dtype
                cand_bytes = _bytes_for(cand_numel, cand_dtype)
                if cand_bytes is None:
                    continue
                if target_bytes is not None and cand_bytes == target_bytes:
                    filt.append(t)
            if filt:
                if len(filt) < len(candidates):
                    n_disamb += 1
                out[(gid, ctid)] = filt
                continue
        # Truly ambiguous: skip to avoid corrupting residency state.
        # An ambiguous mapping would attach prefetch/evict actions to
        # multiple cgsim tensors at once — including tensors that are
        # legitimately handled by some OTHER (gid, ctid). Better to
        # under-model: leaving these out causes DAV's existing residency
        # check to fall back to a sync RAM->VRAM transfer at consumer
        # time, which is identical to the legacy sync injector path.
        n_ambig += 1
    if n_disamb or n_ambig:
        print(
            f"[inject_schedule] tid resolution: shape_disambiguated="
            f"{n_disamb} ambiguous_skipped={n_ambig}",
            flush=True,
        )
    return out


def _retarget_non_coldstart_weights_to_ram(
    trace: Trace,
    coldstart_cgsim_tids: set[int],
    prefetch_covered_cgsim_tids: set[int] | None = None,
) -> int:
    """Switch WEIGHT/LEAF tensors that are explicitly covered by the
    schedule's prefetch ops from cuda → cpu device, so DAV's layout
    places them in RAM only and the schedule's prefetches do real work.

    Tensors NOT in the schedule (neither cold-start nor prefetch
    covered) STAY on cuda → DAV's layout places them in VRAM at
    startup. This avoids the trap where every WEIGHT silently gets
    retargeted, including tensors the scheduler couldn't fit; those
    ended up sync-loaded by DAV's residency-miss path at consumer
    time, blowing up gpu stall. Now: if the scheduler couldn't admit
    a tensor's prefetch, it stays cuda-resident (== implicit
    cold-start fallback).

    When ``prefetch_covered_cgsim_tids`` is None, falls back to
    legacy behavior (retarget every non-coldstart weight) for
    callers that haven't migrated yet.
    """
    n = 0
    for tid, tensor in trace.tensor_map.items():
        if tensor.args.get("tensor_type") not in ("WEIGHT", "LEAF"):
            continue
        if tid in coldstart_cgsim_tids:
            continue
        # New gate: only retarget if the schedule has a prefetch op
        # for this tensor. Otherwise it's neither cold-start nor
        # prefetched — keep it cuda-resident as implicit cold-start.
        if (
            prefetch_covered_cgsim_tids is not None
            and tid not in prefetch_covered_cgsim_tids
        ):
            continue
        device = str(tensor.args.get("device", "")).lower()
        if not device.startswith("cuda"):
            continue
        tensor.args["device"] = "cpu"
        n += 1
    return n


# --------------------------------------------------------- format detection


def _require_neutral_format(doc: dict[str, Any]) -> None:
    """Validate that ``doc`` is a neutral schedule (the only format we
    inject into cg-sim). The legacy ``jit_sim_prune_schedule.json``
    pytorch-runtime format is no longer accepted — it inflated peak
    VRAM by ~1.65 GB through a different tid-resolution path."""
    if "prefetches" in doc and "tensors" in doc:
        return
    if "io_operations" in doc:
        raise ValueError(
            "Legacy pytorch-format schedule (`jit_sim_prune_schedule.json`) "
            "is no longer accepted by the cg-sim injector. Pass the "
            "neutral `schedule.json` instead — it carries the metadata "
            "the sim needs without going through the lossy "
            "compiled_tensor_map fallback."
        )
    raise ValueError(
        "Unrecognized schedule format: expected 'prefetches'+'tensors' "
        "(neutral schedule.json)."
    )


# ------------------------------------------------------------- neutral path


def _build_lid_ctid_order(
    tensor_metas: dict[tuple[int, int], dict[str, Any]],
) -> dict[tuple[int, int], list[int]]:
    """Per-(gid, lid) ctid order, sorted ascending by ``graph_input_idx``.

    Compiled inductor kernels read their args from a flat tuple of
    function inputs; the ``graph_input_idx`` of each ctid is its slot
    in that tuple. Within a launch, the kernel reads its weight inputs
    in ascending slot order, so this gives a canonical per-(gid, lid)
    ordering of the ctids that a kernel at lid can see.

    Used as the position-based tiebreaker in
    ``_resolve_tid_for_node`` when shape alone can't disambiguate two
    same-shape weights consumed by the same kernel.
    """
    pairs: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for (gid, ctid), meta in tensor_metas.items():
        gidx = meta.get("graph_input_idx")
        if gidx is None:
            gidx = ctid  # fall back to ctid order if graph_input_idx absent
        try:
            gidx_int = int(gidx)
        except (TypeError, ValueError):
            gidx_int = int(ctid)
        for lid in meta.get("used_by_launch_ids") or []:
            try:
                lid_int = int(lid)
            except (TypeError, ValueError):
                continue
            pairs[(int(gid), lid_int)].append((gidx_int, int(ctid)))
    out: dict[tuple[int, int], list[int]] = {}
    for key, lst in pairs.items():
        lst.sort()
        out[key] = [c for _g, c in lst]
    return out


def _resolve_tid_for_node(
    trace: Trace,
    node_id: int,
    gid: int,
    lid: int,
    ctid: int,
    tensor_metas: dict[tuple[int, int], dict[str, Any]],
    lid_ctid_order: dict[tuple[int, int], list[int]],
) -> int | None:
    """Pick the cgsim_tid in ``node_id``'s WEIGHT/LEAF inputs that
    corresponds to ``(gid, ctid)``. The mapping is per-node — different
    iterations of the same launch carry their own per-iter cgsim_tid for
    the same logical ctid, and we want the one this specific node reads.

    Resolution order:
      1. **Shape-unique**: filter the node's WEIGHT/LEAF inputs to those
         whose ``numel`` matches ``ctid``'s shape. If exactly one
         survives, return it. (~95% of weights.)
      2. **Position tiebreaker**: when multiple inputs share the
         ctid's shape (two same-shape weights consumed by the same
         kernel), break the tie by paired sort: rank the same-shape
         ctids of this lid by ``graph_input_idx``, rank the same-shape
         candidate cgsim_tids by ``id``, take the kth.

    Returns ``None`` if nothing matches (e.g. the node doesn't actually
    read this ctid — the schedule's wait_node is wrong, or the trace's
    edge ordering dropped the input).
    """
    node = trace.node_map.get(int(node_id))
    if node is None:
        return None
    meta = tensor_metas.get((gid, ctid))
    if meta is None:
        return None
    target_numel = _shape_product(meta.get("shape"))
    target_bytes = meta.get("size_bytes")
    target_dtype = meta.get("dtype")

    def _byte_match(t_id: int) -> bool:
        t = trace.tensor_map.get(t_id)
        if t is None:
            return False
        cand_numel = _shape_product(t.args.get("shape"))
        cand_dtype = t.args.get("dtype") or target_dtype
        if target_numel is not None and cand_numel == target_numel:
            return True
        if target_bytes is not None:
            cand_bytes = _bytes_for(cand_numel, cand_dtype)
            if cand_bytes is not None and cand_bytes == target_bytes:
                return True
        return False

    candidates: list[int] = []
    for tid in node.input_tensors:
        t = trace.tensor_map.get(int(tid))
        if t is None:
            continue
        if t.args.get("tensor_type") not in ("WEIGHT", "LEAF"):
            continue
        if _byte_match(int(tid)):
            candidates.append(int(tid))

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    # Tiebreaker: match same-shape ctids of this lid against same-shape
    # candidates by canonical sort (graph_input_idx ↔ tid id). We accept
    # either a numel match (shape present) or a size_bytes match (shape
    # absent in serialized schedules) so synthetic entries that share a
    # shape with real ctids still get consistent positions.
    lid_ctids = lid_ctid_order.get((gid, lid), [])

    def _meta_same_shape(other_meta: dict[str, Any]) -> bool:
        other_numel = _shape_product(other_meta.get("shape"))
        if target_numel is not None and other_numel is not None:
            return other_numel == target_numel
        other_bytes = other_meta.get("size_bytes")
        if target_bytes is not None and other_bytes is not None:
            return int(other_bytes) == int(target_bytes)
        return False

    same_shape_ctids = [
        c for c in lid_ctids
        if _meta_same_shape(tensor_metas.get((gid, c), {}))
    ]
    if ctid not in same_shape_ctids:
        return sorted(candidates)[0]
    pos = same_shape_ctids.index(ctid)
    sorted_cands = sorted(candidates)
    if pos >= len(sorted_cands):
        return sorted_cands[-1]
    return sorted_cands[pos]


def _inject_neutral(trace: Trace, doc: dict[str, Any]) -> None:
    """Primary injection path: schedule emitted by ``solve_neutral``.

    Self-contained — no bundle CSV reads. Each tensor entry carries
    ``graph_id``, ``compiled_tensor_id``, ``shape`` (via dtype + size_bytes
    or absent), and ``used_by_launch_ids``.
    """
    _set_h2d_streams_from_schedule(trace, doc)

    tensors_by_uid: dict[int, dict[str, Any]] = {
        int(t["uid"]): t for t in doc.get("tensors", [])
    }
    tensor_metas: dict[tuple[int, int], dict[str, Any]] = {}
    for t in tensors_by_uid.values():
        gid = int(t.get("graph_id", -1))
        ctid = int(t.get("compiled_tensor_id", -1))
        if gid < 0 or ctid < 0:
            continue
        tensor_metas[(gid, ctid)] = {
            "used_by_launch_ids": t.get("used_by_launch_ids") or [],
            "shape": t.get("shape"),
            "size_bytes": t.get("size_bytes"),
            "dtype": t.get("dtype"),
            "graph_input_idx": t.get("graph_input_idx"),
        }

    # Legacy resolver kept for retarget-fallback diagnostics, but the
    # arrival/evict loops below resolve cgsim_tid PER-NODE (consumer
    # for prefetches, evict-anchor for evicts) so each schedule
    # operation maps to exactly one cgsim_tid — the specific iter
    # instance that node reads. This eliminates the
    # `(gid, ctid) -> [tid_a, tid_b, ...]` bundling that caused
    # multi-tensor batches and un-gated consumers in the runtime.
    tid_to_cgsim = _resolve_tids_via_metas(trace, tensor_metas)
    lid_ctid_order = _build_lid_ctid_order(tensor_metas)
    lid_to_node_id = _build_lid_to_node_id(trace)

    # Pre-compute (gid, lid) -> [gpu_node_id] once. The cold-start
    # fallback resolver below used to walk trace.node_map per cold-
    # start tensor (O(n_unresolved_cs · n_uses · n_trace_nodes)),
    # which for sdxl-turbo's UNet with ~50k trace nodes and ~200
    # unresolved synth tensors stalled the inject step for >15
    # minutes. One-time precompute makes the inner lookup O(1).
    _lid_to_gpu_nodes: dict[tuple[int, int], list[int]] = defaultdict(list)
    for nid, node in trace.node_map.items():
        if (str(node.args.get("resource_kind") or "")
                not in ("gpu_stream", "gpu", "gpu_runtime")):
            continue
        ng = int(node.args.get("compiled_graph_id") or -1)
        nl = int(node.args.get("compiled_launch_id") or -1)
        if ng < 0 or nl < 0:
            continue
        _lid_to_gpu_nodes[(ng, nl)].append(nid)

    coldstart_cgsim: set[int] = set()
    # Diagnostics on cold-start resolution paths. Each schedule entry
    # falls into exactly one of:
    #   - "pre_resolved": the schedule embedded cgsim_tids that exist
    #     in trace.tensor_map (zero injector heuristic). Healthy.
    #   - "per_node_resolved": no embedded ids, but per-node resolver
    #     found at least one matching trace tid. Some scheduler-side
    #     pre-resolution gap, but the injector recovered cleanly.
    #   - "bundled_fallback": neither path worked, fell back to the
    #     bundled ``tid_to_cgsim[(gid, ctid)]`` resolver — this is the
    #     OVER-COLDSTART path that bundles multiple iters/aliases and
    #     inflates VRAM. Track bytes so the scheduler can prioritise
    #     fixing pre-resolution.
    #   - "unresolved": no path produced any cgsim_tid. The tensor
    #     stays cuda-resident by default but it's silently invisible
    #     to coldstart_cgsim — the injector cannot evict it later.
    cs_diag = {
        "pre_resolved_entries": 0,
        "pre_resolved_bytes": 0,
        "per_node_entries": 0,
        "per_node_bytes": 0,
        "bundled_fallback_entries": 0,
        "bundled_fallback_bytes": 0,
        "bundled_fallback_extra_tids": 0,
        "unresolved_entries": 0,
        "unresolved_bytes": 0,
        "samples_bundled_fallback": [],
        "samples_unresolved": [],
    }

    def _tensor_size_for_ctid(gid_q: int, ctid_q: int) -> int:
        meta_q = tensor_metas.get((int(gid_q), int(ctid_q)))
        if meta_q is None:
            return 0
        sb = meta_q.get("size_bytes")
        try:
            return int(sb) if sb else 0
        except (TypeError, ValueError):
            return 0

    for cs in doc.get("cold_starts", []):
        uid = int(cs.get("tensor_uid", -1))
        meta = tensors_by_uid.get(uid)
        if meta is None:
            continue
        gid = int(meta.get("graph_id", -1))
        ctid = int(meta.get("compiled_tensor_id", -1))
        anchor_lid = int(cs.get("anchor_launch_id", -1))
        size_b = _tensor_size_for_ctid(gid, ctid)

        # Fast path: the scheduler already resolved cgsim_tids at emit
        # time (see ``resolve_neutral_cgsim_tids``). Use them directly
        # without any disambiguation heuristic.
        pre_resolved_cs = [int(t) for t in cs.get("cgsim_tids", [])]
        if pre_resolved_cs:
            any_pre = False
            for cg in pre_resolved_cs:
                if cg in trace.tensor_map:
                    coldstart_cgsim.add(cg)
                    any_pre = True
            if any_pre:
                cs_diag["pre_resolved_entries"] += 1
                cs_diag["pre_resolved_bytes"] += size_b
                continue
            # If none of the pre-resolved ids exist (trace mismatch),
            # fall through to legacy resolution to recover.

        # Resolve per-node: pick the ANCHOR lid's first GPU instance and
        # ask which cgsim_tid IT reads for this ctid. Across iterations
        # the same ctid maps to different per-iter cgsim_tids; each
        # such cgsim_tid is its own physical tensor and must be kept
        # cuda-resident. So we collect from EVERY occurrence of the
        # ctid's launches.
        per_lid_nodes: dict[int, list[int]] = defaultdict(list)
        for lid_raw in meta.get("used_by_launch_ids") or []:
            try:
                lid_int = int(lid_raw)
            except (TypeError, ValueError):
                continue
            for nid in _lid_to_gpu_nodes.get((gid, lid_int), ()):
                per_lid_nodes[lid_int].append(nid)
        any_resolved = False
        for lid_int, nids in per_lid_nodes.items():
            for nid in nids:
                cg = _resolve_tid_for_node(
                    trace, nid, gid, lid_int, ctid,
                    tensor_metas, lid_ctid_order,
                )
                if cg is not None:
                    coldstart_cgsim.add(int(cg))
                    any_resolved = True
        if any_resolved:
            cs_diag["per_node_entries"] += 1
            cs_diag["per_node_bytes"] += size_b
            continue
        # Fallback to the legacy bundled resolver — better to
        # over-coldstart (extra layout VRAM) than to silently
        # retarget a tensor the schedule meant to keep cuda.
        bundled = tid_to_cgsim.get((gid, ctid), [])
        if bundled:
            for cg in bundled:
                coldstart_cgsim.add(int(cg))
            cs_diag["bundled_fallback_entries"] += 1
            cs_diag["bundled_fallback_bytes"] += size_b
            cs_diag["bundled_fallback_extra_tids"] += max(0, len(bundled) - 1)
            if len(cs_diag["samples_bundled_fallback"]) < 5:
                cs_diag["samples_bundled_fallback"].append(
                    (gid, ctid, size_b, len(bundled))
                )
        else:
            cs_diag["unresolved_entries"] += 1
            cs_diag["unresolved_bytes"] += size_b
            if len(cs_diag["samples_unresolved"]) < 5:
                cs_diag["samples_unresolved"].append(
                    (gid, ctid, size_b, uid)
                )

    arrivals: list[dict[str, Any]] = []
    prefetch_covered_cgsim: set[int] = set()
    # Sync prefetches (issuer == consumer) are kept in a separate set
    # so they trigger the cuda → cpu retarget (peak control via
    # residency-miss + stall) but bypass the async coverage_repair
    # logic — they have no arrivals to be checked for un-gated
    # consumers, by design. The retarget pass unions the two sets.
    sync_prefetch_cgsim: set[int] = set()
    n_pf_exact = 0
    n_pf_launch_fallback = 0
    n_pf_async = 0
    n_pf_sync = 0   # sync prefetches: not added; DAV residency-miss path handles them
    n_pf_skipped = 0
    n_pf_unresolved_tid = 0
    n_pf_pre_resolved = 0   # used cgsim_tid embedded in schedule (no disambiguation)
    # Prefetches whose schedule-emitted ``issue_node_id`` lands AFTER
    # the consumer in trace time (ts inversion). Registering the gate
    # would cause a deadlock — the issuer's retire-event firing the
    # arrival happens past the consumer's dispatch deadline. Treat as
    # sync prefetches: skip the arrival, leave the tid uncovered, let
    # the coverage-repair walk decide whether the tid stays cuda or
    # needs to be demoted.
    n_pf_infeasible = 0
    for pf in doc.get("prefetches", []):
        uid = int(pf.get("tensor_uid", -1))
        meta = tensors_by_uid.get(uid)
        if meta is None:
            n_pf_skipped += 1
            continue
        gid = int(meta.get("graph_id", -1))
        ctid = int(meta.get("compiled_tensor_id", -1))
        issue_gid = int(pf.get("issue_graph_id", -1))
        issue_lid = int(pf.get("issue_launch_id", -1))
        wait_lid = int(pf.get("wait_launch_id", -1))
        if wait_lid < 0:
            n_pf_skipped += 1
            continue
        if issue_lid < 0:
            issue_lid = wait_lid
            issue_gid = gid
        elif issue_gid < 0:
            issue_gid = gid
        issue_node_exact = _valid_gpu_node_id(
            trace, int(pf.get("issue_node_id", -1))
        )
        wait_node_exact = _valid_gpu_node_id(
            trace, int(pf.get("wait_node_id", -1))
        )
        if issue_node_exact is not None and wait_node_exact is not None:
            issuer_id = issue_node_exact
            consumer_id = wait_node_exact
            n_pf_exact += 1
        else:
            issuer_id = lid_to_node_id.get((issue_gid, issue_lid))
            consumer_id = lid_to_node_id.get((gid, wait_lid))
            n_pf_launch_fallback += 1
        if issuer_id is None or consumer_id is None:
            n_pf_skipped += 1
            continue
        # Fast path: scheduler already resolved at emit time.
        pre_resolved_pf = int(pf.get("cgsim_tid", -1))
        if pre_resolved_pf >= 0 and pre_resolved_pf in trace.tensor_map:
            cgsim_tid = pre_resolved_pf
            n_pf_pre_resolved += 1
        else:
            # Per-consumer cgsim_tid resolution: each prefetch refers to
            # ONE specific iter-instance of the ctid — the one that
            # consumer_id reads. The legacy `tid_to_cgsim` lookup returned
            # ALL same-(gid, ctid) cgsim_tids (across iters AND across
            # shape-twin ctids) and bundled them into every prefetch,
            # which inflated transfer sizes and left consumers of the
            # bundled-in tids un-gated. Resolve per-node instead.
            cgsim_tid = _resolve_tid_for_node(
                trace, consumer_id, gid, wait_lid, ctid,
                tensor_metas, lid_ctid_order,
            )
        if cgsim_tid is None:
            n_pf_unresolved_tid += 1
            n_pf_skipped += 1
            continue
        # Async iff the resolved issuer node differs from the consumer.
        # The issue placement is the single source of truth for the
        # schedule's intent: if it resolves to the consumer, it's a sync
        # prefetch and DAV's residency-miss path handles it. Otherwise,
        # respect the placement and gate the consumer.
        # ``trusted_async`` is intentionally ignored: it's a safety
        # flag from ct_milp_oracle that older schedulers don't set,
        # and treating its absence as "sync" would force every
        # multi-graph / belady / mincost-flow prefetch through the
        # sync path even when the schedule placed them earlier.
        if issuer_id == consumer_id:
            # Sync prefetch: no async arrival/gate registered — DAV's
            # residency-miss path fetches the tensor when the consumer
            # dispatches. We track it in a separate set that the
            # retarget pass unions in (so the tensor's home flips
            # cuda → cpu and the residency-miss path actually does
            # work) but the async coverage_repair loop skips, since
            # there's no arrival to validate against un-gated consumers.
            sync_prefetch_cgsim.add(int(cgsim_tid))
            n_pf_sync += 1
            continue
        # Time-order check: a gate registered on this consumer is only
        # useful if its issuer fires BEFORE the consumer dispatches.
        # Schedulers (esp. jit_sim_prune's ALAP backward pass) sometimes
        # pick a same-launch_id node from a later iter as the issuer,
        # which lands past the consumer in trace ts. Without this guard,
        # the runtime stalls forever waiting for the issuer's retire.
        # Demote to sync (skip the arrival) — coverage-repair below will
        # then keep the tid cuda-resident if no other arrival covers
        # this consumer.
        issuer_node = trace.node_map.get(issuer_id)
        consumer_node = trace.node_map.get(consumer_id)
        issuer_ts = int(issuer_node.args.get("start_ns") or 0) if issuer_node else 0
        consumer_ts = int(consumer_node.args.get("start_ns") or 0) if consumer_node else 0
        if issuer_ts >= consumer_ts:
            n_pf_infeasible += 1
            continue
        start_ns = int(pf.get("transfer_start_ns", 0))
        end_ns = int(pf.get("transfer_end_ns", 0))
        duration_ns = max(0, end_ns - start_ns)
        arrivals.append({
            "issuer_node_id": issuer_id,
            "consumer_node_id": consumer_id,
            "cgsim_tids": [int(cgsim_tid)],
            "duration_ns": duration_ns,
            "trusted_async": bool(pf.get("trusted_async", False)),
        })
        prefetch_covered_cgsim.add(int(cgsim_tid))
        n_pf_async += 1

    # ------------------------------------------------------------------
    # Coverage repair (was "coverage demote" — too coarse, kept ~85% of
    # weights cuda-resident).
    #
    # Background: the cg-sim loader merges cgsim Tensors that share
    # ``(device, storage_id)`` — a real PyTorch invariant from
    # ``aten::view`` / ``aten::reshape`` that creates alias rows
    # pointing at one storage. So two compiled_tensor_map ctids with
    # *different* compiled-graph view shapes can map to the same
    # post-merge cgsim_tid. The schedule plans only against ctids it
    # knows about; the un-counted view ctids' GPU consumers read the
    # same physical cgsim_tid but aren't gated.
    #
    # Old behavior (coverage_demote): if any consumer of a retargeted
    # cgsim_tid was un-gated, demote the tid to cuda-resident.
    # Correct, but pessimistic: the un-gated consumer's window often
    # falls *inside* the schedule's planned residency anyway — only
    # the few that fall in an evicted window are actually broken.
    #
    # New behavior:
    #   1. Per cgsim_tid, walk consumers + arrivals + evicts in trace
    #      time order.
    #   2. If an un-gated consumer falls in a window the schedule's
    #      evict had freed → drop just that evict (the tid stays
    #      loaded across the consumer) and synthesize a gate-only
    #      arrival so the consumer correctly waits for the
    #      previously-issued prefetch's transfer to complete.
    #   3. If an un-gated consumer runs *before* any arrival of this
    #      tid → demote (the tid must be cuda-resident from layout).
    #   4. The schedule's bytes / fire-times are preserved bit-for-
    #      bit; we only *defer* evictions that would have stranded a
    #      consumer the scheduler didn't account for.
    gated_by_tid_check: dict[int, set[int]] = defaultdict(set)
    for a in arrivals:
        for t in a["cgsim_tids"]:
            gated_by_tid_check[int(t)].add(int(a["consumer_node_id"]))
    consumers_by_tid: dict[int, set[int]] = defaultdict(set)
    for nid, n in trace.node_map.items():
        rk = str(n.args.get("resource_kind") or "")
        if rk not in ("gpu_stream", "gpu", "gpu_runtime"):
            continue
        for tid in n.input_tensors or []:
            consumers_by_tid[int(tid)].add(int(nid))

    # Sketch the schedule's per-tid timeline so we can locate each
    # un-gated consumer relative to the arrival/evict events the
    # schedule already plans.
    arrivals_by_tid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in arrivals:
        for t in a["cgsim_tids"]:
            arrivals_by_tid[int(t)].append(a)
    # Evicts at this point haven't been built yet (they come after this
    # block). We pre-resolve them from the schedule doc using the same
    # per-node logic so we know which evicts to keep / drop *before* we
    # apply them to ``evict_after_node``.
    planned_evicts_by_tid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for ev in doc.get("evicts", []):
        uid_e = int(ev.get("tensor_uid", -1))
        meta_e = tensors_by_uid.get(uid_e)
        if meta_e is None:
            continue
        gid_e = int(meta_e.get("graph_id", -1))
        ctid_e = int(meta_e.get("compiled_tensor_id", -1))
        evict_lid_e = int(ev.get("issue_launch_id", -1))
        ev_node = _valid_gpu_node_id(trace, int(ev.get("issue_node_id", -1)))
        if ev_node is None:
            ev_node = lid_to_node_id.get((gid_e, evict_lid_e))
        if ev_node is None:
            continue
        cg_e = _resolve_tid_for_node(
            trace, ev_node, gid_e, evict_lid_e, ctid_e,
            tensor_metas, lid_ctid_order,
        )
        if cg_e is None:
            continue
        ts_e = int(trace.node_map[ev_node].args.get("end_ns") or 0)
        planned_evicts_by_tid[int(cg_e)].append((ts_e, int(ev_node)))

    def _trace_ts(node_id: int) -> int:
        n = trace.node_map.get(int(node_id))
        if n is None:
            return 0
        return int(n.args.get("start_ns") or 0)

    coverage_demoted: set[int] = set()
    evicts_to_drop: set[tuple[int, int]] = set()  # (evict_node_id, cgsim_tid)
    synthesized_gates: list[dict[str, Any]] = []
    repaired_tids = 0
    repaired_consumers = 0
    # Diagnostics on coverage-repair paths. The injector silently
    # patches the schedule when its prefetch coverage doesn't include
    # every gpu trace consumer of a tid:
    #   - "demoted_no_arrivals_for_tid": tid was in prefetch_covered_cgsim
    #     but somehow has no scheduled arrival. Schedule emit bug.
    #   - "demoted_consumer_before_first_arrival": some gpu trace
    #     consumer of the tid (often an aten/aux op the schedule
    #     didn't account for) runs BEFORE the schedule's first arrival
    #     fires. The injector can't synthesize a gate (nothing prior
    #     to anchor it to), so it gives up on retargeting this tid →
    #     stays cuda-resident at layout. The schedule's prefetch
    #     becomes a no-op since the tid was already kept resident.
    #   - "repaired_with_synth_gates": some un-gated consumer existed
    #     but a prior arrival was available, so the injector
    #     synthesized a fake gate. Schedule should have emitted that
    #     gate itself.
    cov_diag = {
        "demoted_no_arrivals_tids": 0,
        "demoted_no_arrivals_bytes": 0,
        "demoted_consumer_before_arrival_tids": 0,
        "demoted_consumer_before_arrival_bytes": 0,
        "repaired_tids": 0,
        "repaired_bytes": 0,
        "synth_gates_count": 0,
        "samples_demoted": [],
        "samples_repaired": [],
    }

    def _tid_size(tid: int) -> int:
        tt = trace.tensor_map.get(int(tid))
        if tt is None:
            return 0
        return int(getattr(tt, "size_bytes", 0) or 0)

    for tid in list(prefetch_covered_cgsim):
        all_cons = consumers_by_tid.get(tid, set())
        gated = gated_by_tid_check.get(tid, set())
        un_gated = all_cons - gated
        if not un_gated:
            continue
        arrivals_t = arrivals_by_tid.get(tid, [])
        if not arrivals_t:
            # No arrival fires for this tid in this run (shouldn't be in
            # prefetch_covered_cgsim). Demote defensively.
            coverage_demoted.add(tid)
            cov_diag["demoted_no_arrivals_tids"] += 1
            cov_diag["demoted_no_arrivals_bytes"] += _tid_size(tid)
            continue
        # Order events on the tid's residency timeline by trace ts.
        # `key=` keeps the tuple comparison from falling through to dict
        # compare when two arrivals share a trace timestamp.
        arrival_ts = sorted(
            (
                (_trace_ts(a["issuer_node_id"]), idx, a)
                for idx, a in enumerate(arrivals_t)
            ),
            key=lambda x: (x[0], x[1]),
        )
        evict_ts = sorted(planned_evicts_by_tid.get(tid, []))
        first_arrival_t = arrival_ts[0][0]

        # For each un-gated consumer, locate its position relative to
        # the schedule's last arrival/evict before it.
        ok = True
        per_tid_drops: list[tuple[int, int]] = []
        per_tid_synth: list[dict[str, Any]] = []
        for c in un_gated:
            ct = _trace_ts(c)
            if ct < first_arrival_t:
                # Un-gated consumer runs before the first arrival fires:
                # the tid would be ABSENT at C's dispatch. Must demote.
                ok = False
                break
            # Most-recent arrival before C.
            last_arrival_t = max(
                (at for at, _idx, _a in arrival_ts if at <= ct),
                default=None,
            )
            if last_arrival_t is None:
                ok = False
                break
            # Find evicts in (last_arrival_t, ct]: each one would have
            # freed the tid before C runs. Drop them all.
            for et, ev_node in evict_ts:
                if last_arrival_t < et <= ct:
                    per_tid_drops.append((ev_node, tid))
            # Pick an issuer for the synthesized gate-only arrival —
            # reuse the most-recent gating arrival's issuer so the gate
            # is registered on a node we know fires.
            anchor_arrival = next(
                a for at, _idx, a in reversed(arrival_ts) if at <= ct
            )
            per_tid_synth.append({
                "issuer_node_id": anchor_arrival["issuer_node_id"],
                "consumer_node_id": int(c),
                "cgsim_tids": [int(tid)],
                "duration_ns": 0,
                "trusted_async": True,
            })

        if not ok:
            coverage_demoted.add(tid)
            cov_diag["demoted_consumer_before_arrival_tids"] += 1
            cov_diag["demoted_consumer_before_arrival_bytes"] += _tid_size(tid)
            if len(cov_diag["samples_demoted"]) < 5:
                tt = trace.tensor_map.get(int(tid))
                tname = (tt.args.get("name", "?") if tt else "?")[:40]
                cov_diag["samples_demoted"].append(
                    (int(tid), _tid_size(tid), tname, len(un_gated), len(arrivals_t))
                )
            continue
        evicts_to_drop.update(per_tid_drops)
        synthesized_gates.extend(per_tid_synth)
        repaired_tids += 1
        repaired_consumers += len(per_tid_synth)
        cov_diag["repaired_tids"] += 1
        cov_diag["repaired_bytes"] += _tid_size(tid)
        cov_diag["synth_gates_count"] += len(per_tid_synth)
        if len(cov_diag["samples_repaired"]) < 5:
            tt = trace.tensor_map.get(int(tid))
            tname = (tt.args.get("name", "?") if tt else "?")[:40]
            cov_diag["samples_repaired"].append(
                (int(tid), _tid_size(tid), tname, len(per_tid_synth), len(un_gated))
            )

    if coverage_demoted:
        prefetch_covered_cgsim -= coverage_demoted
        # The arrivals targeting these tids stay in `arrivals` so the
        # schedule's plan is preserved bit-for-bit. At runtime they
        # find the tensor RESIDENT and no-op (RESIDENT counts as the
        # gate's "ready" state, so dependent compute still proceeds).
    if synthesized_gates:
        arrivals.extend(synthesized_gates)

    # Retarget AFTER the prefetch loop so we know which tensors are
    # actually covered by a prefetch op AND have full consumer coverage.
    # Tensors with no prefetch coverage — or with un-gated consumers
    # exposed by storage aliasing — stay cuda-resident (= implicit
    # cold-start in DAV's layout) instead of being silently retargeted
    # to RAM and sync-loaded by DAV's residency-miss path.
    # Scheduler-opt-in: ``retarget_all_non_coldstart`` flag in schedule
    # meta tells the injector to retarget EVERY non-coldstart WEIGHT/
    # LEAF, not just those in ``prefetch_covered_cgsim``. Used by the
    # peak_runtime_planner — it emits sync prefetches against
    # tl.tensors, but the trace contains cgsim_tids outside the
    # sidecar (Pass-A storage merge artifacts, alias rows) that would
    # otherwise stay cuda-homed. With this flag set, the scheduler is
    # asserting "anything not explicitly cold-started should be
    # RAM-resident; DAV's residency-miss path will sync-fetch on
    # demand". Falls back to the default coverage-gated mode when
    # the flag is absent/false.
    retarget_all = bool(doc.get("meta", {}).get("retarget_all_non_coldstart"))
    covered_for_retarget = (
        None if retarget_all
        else (prefetch_covered_cgsim | sync_prefetch_cgsim)
    )
    n_retarget = _retarget_non_coldstart_weights_to_ram(
        trace, coldstart_cgsim,
        prefetch_covered_cgsim_tids=covered_for_retarget,
    )
    print(
        f"[inject_schedule] retargeted {n_retarget} WEIGHT tensors → RAM "
        f"(prefetch_covered={len(prefetch_covered_cgsim)}, "
        f"coverage_demoted={len(coverage_demoted)}, "
        f"coverage_repaired_tids={repaired_tids} "
        f"deferred_evicts={len(evicts_to_drop)} "
        f"synth_gates={len(synthesized_gates)})",
        flush=True,
    )
    # ---- Diagnostic breakdown of silent injector patches ----
    # Each section below is a place where the injector PATCHED the
    # schedule's intent without the scheduler's knowledge. Every
    # non-zero count here is a scheduler-side bug we should fix at
    # the root, not let the injector silently absorb.
    print(
        "[inject_schedule:diag] cold_start_resolution: "
        f"pre_resolved={cs_diag['pre_resolved_entries']} "
        f"({cs_diag['pre_resolved_bytes']/1e6:.1f}MB) | "
        f"per_node_recovered={cs_diag['per_node_entries']} "
        f"({cs_diag['per_node_bytes']/1e6:.1f}MB) | "
        f"bundled_fallback={cs_diag['bundled_fallback_entries']} "
        f"({cs_diag['bundled_fallback_bytes']/1e6:.1f}MB, "
        f"extra_tids_added={cs_diag['bundled_fallback_extra_tids']}) | "
        f"unresolved={cs_diag['unresolved_entries']} "
        f"({cs_diag['unresolved_bytes']/1e6:.1f}MB)",
        flush=True,
    )
    if cs_diag["samples_bundled_fallback"]:
        print(
            "[inject_schedule:diag]   bundled_fallback samples (gid, ctid, "
            "size_mb, bundle_size): "
            + ", ".join(
                f"({g},{c},{sz/1e6:.1f}MB,n={n})"
                for g, c, sz, n in cs_diag["samples_bundled_fallback"]
            ),
            flush=True,
        )
    if cs_diag["samples_unresolved"]:
        print(
            "[inject_schedule:diag]   unresolved samples (gid, ctid, "
            "size_mb, uid): "
            + ", ".join(
                f"({g},{c},{sz/1e6:.1f}MB,uid={u})"
                for g, c, sz, u in cs_diag["samples_unresolved"]
            ),
            flush=True,
        )
    print(
        "[inject_schedule:diag] coverage_repair: "
        f"demoted_no_arrivals={cov_diag['demoted_no_arrivals_tids']} "
        f"({cov_diag['demoted_no_arrivals_bytes']/1e6:.1f}MB) | "
        f"demoted_consumer_before_arrival="
        f"{cov_diag['demoted_consumer_before_arrival_tids']} "
        f"({cov_diag['demoted_consumer_before_arrival_bytes']/1e6:.1f}MB) | "
        f"repaired={cov_diag['repaired_tids']} "
        f"({cov_diag['repaired_bytes']/1e6:.1f}MB, "
        f"synth_gates={cov_diag['synth_gates_count']})",
        flush=True,
    )
    if cov_diag["samples_demoted"]:
        print(
            "[inject_schedule:diag]   demoted samples (cgsim_tid, "
            "size_mb, name, n_un_gated, n_arrivals): "
            + ", ".join(
                f"({t},{sz/1e6:.1f}MB,{nm!r},ng={ng},ar={ar})"
                for t, sz, nm, ng, ar in cov_diag["samples_demoted"]
            ),
            flush=True,
        )
    if cov_diag["samples_repaired"]:
        print(
            "[inject_schedule:diag]   repaired samples (cgsim_tid, "
            "size_mb, name, n_synth, n_un_gated): "
            + ", ".join(
                f"({t},{sz/1e6:.1f}MB,{nm!r},sg={ns},ng={ng})"
                for t, sz, nm, ns, ng in cov_diag["samples_repaired"]
            ),
            flush=True,
        )
    # Total VRAM impact: every demoted tid stays cuda from layout
    # despite the schedule saying "stream it." The bundled_fallback
    # extra tids are additional cgsim_tids the injector cold-starts
    # beyond what the schedule named.
    silent_overhead_bytes = (
        cs_diag["bundled_fallback_bytes"]
        + cs_diag["unresolved_bytes"]
        + cov_diag["demoted_no_arrivals_bytes"]
        + cov_diag["demoted_consumer_before_arrival_bytes"]
    )
    print(
        f"[inject_schedule:diag] TOTAL silent-patch VRAM overhead "
        f"(over what the schedule planned): "
        f"{silent_overhead_bytes/1e6:.1f}MB",
        flush=True,
    )

    evict_after_node: dict[int, set[int]] = {}
    evictable_cgsim: set[int] = set()
    n_ev_exact = 0
    n_ev_launch_fallback = 0
    n_ev_applied = 0
    n_ev_skipped = 0
    n_ev_unresolved_tid = 0
    for ev in doc.get("evicts", []):
        uid = int(ev.get("tensor_uid", -1))
        meta = tensors_by_uid.get(uid)
        if meta is None:
            n_ev_skipped += 1
            continue
        gid = int(meta.get("graph_id", -1))
        ctid = int(meta.get("compiled_tensor_id", -1))
        evict_lid = int(ev.get("issue_launch_id", -1))  # neutral evict uses issue_launch_id (after-last-use)
        evict_node_id = _valid_gpu_node_id(
            trace, int(ev.get("issue_node_id", -1))
        )
        if evict_node_id is not None:
            n_ev_exact += 1
        else:
            evict_node_id = lid_to_node_id.get((gid, evict_lid))
            n_ev_launch_fallback += 1
        if evict_node_id is None:
            n_ev_skipped += 1
            continue
        # Fast path: scheduler pre-resolved cgsim_tid at emit time
        # (resolve_neutral_cgsim_tids uses tl.tensors[uid].trace_tids
        # for an exact storage lookup). The prefetch path above uses
        # this fast path; mirror it here so per-use emits whose
        # consumer-k node has many same-shape inputs don't fall back
        # to shape-tiebreaker and silently mis-resolve. Without this
        # the per-node resolver returns None on small-shape tensors
        # with many same-shape candidates and the evict gets dropped.
        pre_resolved_ev = int(ev.get("cgsim_tid", -1))
        if pre_resolved_ev >= 0 and pre_resolved_ev in trace.tensor_map:
            cgsim_tid = pre_resolved_ev
        else:
            # Per-node evict tid resolution: jit_sim_prune anchors each
            # evict to the kernel that just read this tensor as its last
            # use. That kernel's input_tensors carry the exact iter-
            # instance cgsim_tid we want to free here — same per-node
            # logic as the prefetch path, no bundling.
            cgsim_tid = _resolve_tid_for_node(
                trace, evict_node_id, gid, evict_lid, ctid,
                tensor_metas, lid_ctid_order,
            )
        if cgsim_tid is None:
            n_ev_unresolved_tid += 1
            n_ev_skipped += 1
            continue
        # Skip evicts for cgsim_tids that we kept cuda-resident due to
        # incomplete schedule coverage. Letting the evict fire would
        # release VRAM mid-run for a tensor whose later un-gated
        # consumers expect it to still be resident.
        if int(cgsim_tid) in coverage_demoted:
            n_ev_skipped += 1
            continue
        # Also skip evicts the coverage-repair pass identified as
        # needing to be deferred — dropping them keeps the tid LOADED
        # across a downstream un-gated consumer.
        if (int(evict_node_id), int(cgsim_tid)) in evicts_to_drop:
            n_ev_skipped += 1
            continue
        evict_after_node.setdefault(evict_node_id, set()).add(int(cgsim_tid))
        evictable_cgsim.add(int(cgsim_tid))
        n_ev_applied += 1

    # ------------------------------------------------------------------
    # Cold-start last-use evict pass.
    #
    # Every WEIGHT/LEAF cgsim_tid that the schedule did NOT manage —
    # i.e. it's not in ``prefetch_covered_cgsim`` and not already in
    # ``evict_after_node`` — sits on VRAM from layout to iter end
    # because DAV's ``_PERMANENT_TYPES`` lock plus the silent
    # schedule = "never evict."  For tensors whose true last GPU
    # consumer happens before the iter's peak moment, releasing the
    # VRAM region at that last consumer is a free peak-VRAM win:
    # zero PCIe bytes (release is instant), zero stall (no later
    # consumer needs the tensor), and the layout-phase RAM staging
    # copy survives as a fallback if a missed consumer DOES exist.
    #
    # Crucially, we must use the TRACE's actual last consumer per
    # cgsim_tid, not the schedule's ``tl.uses`` view: Pass-A storage
    # merging can give a single cgsim_tid more consumers than any
    # one (gid, ctid) is aware of (cross-graph reuse, view aliases).
    # Evicting at the schedule's view of last-use would leave those
    # extra consumers reading an absent tensor and trigger DAV's
    # residency-miss sync transfer — wiping out the win.
    n_cold_start_evicts = 0
    n_cold_start_skipped_already_evicted = 0
    n_cold_start_skipped_no_consumers = 0

    # Build per-cgsim_tid last-consumer-node map from the trace.
    last_consumer_by_tid: dict[int, int] = {}
    last_consumer_ts_by_tid: dict[int, int] = {}
    for nid, n in trace.node_map.items():
        rk = str(n.args.get("resource_kind") or "")
        if rk not in ("gpu_stream", "gpu", "gpu_runtime"):
            continue
        ts = int(n.args.get("end_ns") or 0)
        for tid in n.input_tensors or []:
            tid_i = int(tid)
            if ts > last_consumer_ts_by_tid.get(tid_i, -1):
                last_consumer_ts_by_tid[tid_i] = ts
                last_consumer_by_tid[tid_i] = int(nid)

    # Set of cgsim_tids the schedule already plans to evict at some
    # point — we don't want to override its plan, just supplement it
    # for cgsim_tids it left silent.
    already_scheduled_evicted: set[int] = set()
    for nid, tids in evict_after_node.items():
        already_scheduled_evicted.update(tids)

    for tid, t in trace.tensor_map.items():
        ttype = t.args.get("tensor_type")
        if ttype not in ("WEIGHT", "LEAF"):
            continue
        if int(tid) in already_scheduled_evicted:
            n_cold_start_skipped_already_evicted += 1
            continue
        last_node = last_consumer_by_tid.get(int(tid))
        if last_node is None:
            n_cold_start_skipped_no_consumers += 1
            continue
        evict_after_node.setdefault(int(last_node), set()).add(int(tid))
        evictable_cgsim.add(int(tid))
        n_cold_start_evicts += 1

    if arrivals:
        existing = trace.args.setdefault("xfer_arrivals", [])
        existing.extend(arrivals)
    if evict_after_node:
        existing_ev = trace.args.setdefault("evict_after_node", {})
        for nid, tids in evict_after_node.items():
            existing_ev.setdefault(nid, set()).update(tids)
    if evictable_cgsim:
        trace.args.setdefault("evictable_tensor_ids", set()).update(evictable_cgsim)

    print(
        f"[inject_schedule:cold_start_evicts] "
        f"emitted={n_cold_start_evicts} "
        f"already_in_schedule={n_cold_start_skipped_already_evicted} "
        f"no_consumers={n_cold_start_skipped_no_consumers}",
        flush=True,
    )

    print(
        f"[inject_schedule:neutral] "
        f"prefetch async={n_pf_async} sync_fallback={n_pf_sync} "
        f"infeasible_ts={n_pf_infeasible} "
        f"exact={n_pf_exact} launch_fallback={n_pf_launch_fallback} "
        f"pre_resolved={n_pf_pre_resolved} "
        f"(skipped {n_pf_skipped}, unresolved_tid={n_pf_unresolved_tid}); "
        f"evict={n_ev_applied} exact={n_ev_exact} "
        f"launch_fallback={n_ev_launch_fallback} "
        f"(skipped {n_ev_skipped}, unresolved_tid={n_ev_unresolved_tid}); "
        f"evictable_tensors={len(evictable_cgsim)}",
        flush=True,
    )

    # Verification: every consumer of a retargeted-to-RAM cgsim_tid
    # must be in `gate_by_consumer` for the runtime; otherwise
    # `_ensure_inputs_resident` will issue a sync transfer that
    # bypasses xfer_h2d_streams=1 and the schedule's plan. Count and
    # report. With per-node tid resolution this should be 0; any
    # non-zero remainder is a real bug to fix before the simulator
    # can play the schedule "exactly as intended".
    gated_by_tid: dict[int, set[int]] = defaultdict(set)
    for a in arrivals:
        for t in a["cgsim_tids"]:
            gated_by_tid[int(t)].add(int(a["consumer_node_id"]))
    un_gated = 0
    total_consumers = 0
    for tid, t in trace.tensor_map.items():
        if t.args.get("tensor_type") not in ("WEIGHT", "LEAF"):
            continue
        if str(t.args.get("device", "")).lower() != "cpu":
            continue
        gated = gated_by_tid.get(int(tid), set())
        for nid, n in trace.node_map.items():
            if int(tid) not in (n.input_tensors or []):
                continue
            rk = str(n.args.get("resource_kind") or "")
            if rk not in ("gpu_stream", "gpu", "gpu_runtime"):
                continue
            total_consumers += 1
            if int(nid) not in gated:
                un_gated += 1
    print(
        f"[inject_schedule:gate-check] retargeted-tensor GPU consumers="
        f"{total_consumers} un_gated={un_gated}",
        flush=True,
    )




# --------------------------------------------------------------- entry point


def inject_schedule_into_trace(
    trace: Trace,
    schedule: dict[str, Any] | str | Path,
    *,
    bundle_dir: Path | str | None = None,
    disable_evict: bool = False,
    **_legacy_kwargs: Any,  # accept and ignore stale flags like ``model_evicts``
) -> Trace:
    """Annotate ``trace`` with prefetch/evict ops from a neutral schedule.

    Modifies the trace in place by populating
    ``trace.args["xfer_arrivals"]`` and ``trace.args["evict_after_node"]``,
    plus retargeting non-cold-start WEIGHT/LEAF tensors to RAM. The
    ``DeviceAwareVanillaAsync`` scheduler reads these annotations at
    runtime to fire async ``sys.transfer`` calls and gate consumer
    nodes.

    Parameters
    ----------
    trace
        Loaded by ``PytorchProfile``. Modified in place.
    schedule
        Either a parsed dict or a path to a neutral ``schedule.json``.
        The legacy pytorch-format ``jit_sim_prune_schedule.json`` is
        no longer accepted — convert via ``neutral_to_pytorch``'s
        inverse path or re-emit through the scheduler.
    bundle_dir
        Unused; kept for API compatibility.
    """
    del bundle_dir  # legacy positional arg; only the neutral path is supported

    if isinstance(schedule, (str, Path)):
        with open(schedule) as f:
            doc = json.load(f)
    else:
        doc = schedule

    _require_neutral_format(doc)
    _inject_neutral(trace, doc)

    if disable_evict:
        # Positive-control mode: keep prefetches + retargets, but drop
        # the eviction map and evictable-id set. Tensors stay resident
        # in VRAM after their first transfer; subsequent prefetches
        # become no-ops (residency check passes immediately). Useful
        # to test whether sim wall reduces to ~no-WS levels when
        # nothing actually streams — confirms async machinery is
        # correct (transfers don't fire when not needed).
        trace.args.pop("evict_after_node", None)
        trace.args.pop("evictable_tensor_ids", None)
        print("[inject_schedule] disable_evict=True — dropped evict map "
              "and evictable_tensor_ids", flush=True)

    return trace
