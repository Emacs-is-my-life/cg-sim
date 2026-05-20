"""Shared cgsim_tid resolver for the neutral schedule format.

Schedule entries reference tensors by ``(graph_id, compiled_tensor_id)`` —
a compile-space identifier. The simulator references tensors by
``cgsim_tid`` — the integer key into ``trace.tensor_map`` assigned at load
time. The two spaces are not 1:1: a single ``(gid, ctid)`` may map to
many ``cgsim_tid`` instances across runtime invocations of the same
compiled graph.

This module's resolver picks the *specific* ``cgsim_tid`` that a given
runtime node reads for a given ``(gid, ctid)``. Used both at schedule
emit time (in ``write_neutral_schedule``, to embed resolved ids into the
JSON) and as a back-compat fallback inside the injector when an older
schedule omits the resolved id.

Originally lived in ``graph_modifiers/inject_schedule/injector.py``;
factored out so the scheduler emit path can resolve before writing,
making the schedule format unambiguous by construction.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from sim.core.trace import Trace


# ---------------------------------------------------------------------------
# Cheap utilities
# ---------------------------------------------------------------------------


def shape_product(shape: Any) -> int | None:
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


def bytes_for(numel: int | None, dtype: Any) -> int | None:
    if numel is None:
        return None
    nbytes = _DTYPE_BYTES.get(str(dtype) if dtype is not None else None)
    if nbytes is None:
        return None
    return numel * nbytes


# ---------------------------------------------------------------------------
# Per-node resolver
# ---------------------------------------------------------------------------


def valid_gpu_node_id(trace: Trace, node_id: int) -> int | None:
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


def build_lid_to_node_id(trace: Trace) -> dict[tuple[int, int], int]:
    """(graph_id, launch_id) -> first GPU node id matching that key."""
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


def build_lid_to_gpu_node_ids(
    trace: Trace,
) -> dict[tuple[int, int], list[int]]:
    """(graph_id, launch_id) -> list of ALL GPU node ids (across iterations)."""
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


def build_lid_ctid_order(
    tensor_metas: dict[tuple[int, int], dict[str, Any]],
) -> dict[tuple[int, int], list[int]]:
    """Per-(gid, lid), sort ctids ascending by ``graph_input_idx``.

    Compiled kernels read their args from a flat tuple of function inputs;
    the ``graph_input_idx`` of each ctid is its slot. Within a launch, the
    kernel reads in slot order, giving a canonical position used as a
    tiebreaker when shape alone can't pick among same-shape candidates.
    """
    pairs: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for (gid, ctid), meta in tensor_metas.items():
        gidx = meta.get("graph_input_idx")
        if gidx is None:
            gidx = ctid
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


def resolve_tid_for_node(
    trace: Trace,
    node_id: int,
    gid: int,
    lid: int,
    ctid: int,
    tensor_metas: dict[tuple[int, int], dict[str, Any]],
    lid_ctid_order: dict[tuple[int, int], list[int]],
) -> int | None:
    """Pick the cgsim_tid in ``node_id``'s WEIGHT/LEAF inputs that
    corresponds to ``(gid, ctid)``.

    Two-stage:
      1. **Shape-unique**: filter the node's WEIGHT/LEAF inputs to those
         whose numel (or bytes when dtype is known) matches the ctid's
         shape. If exactly one survives, return it. (~95% of weights.)
      2. **Position tiebreaker**: when multiple inputs share the
         ctid's shape, break the tie by paired sort: rank the same-shape
         ctids of this lid by ``graph_input_idx``, rank the same-shape
         candidate cgsim_tids by ``id``, take the kth.

    Returns ``None`` if nothing matches.
    """
    node = trace.node_map.get(int(node_id))
    if node is None:
        return None
    meta = tensor_metas.get((gid, ctid))
    if meta is None:
        return None
    target_numel = shape_product(meta.get("shape"))
    target_bytes = meta.get("size_bytes")
    target_dtype = meta.get("dtype")

    def _byte_match(t_id: int) -> bool:
        t = trace.tensor_map.get(t_id)
        if t is None:
            return False
        cand_numel = shape_product(t.args.get("shape"))
        cand_dtype = t.args.get("dtype") or target_dtype
        if target_numel is not None and cand_numel == target_numel:
            return True
        if target_bytes is not None:
            cand_bytes = bytes_for(cand_numel, cand_dtype)
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

    lid_ctids = lid_ctid_order.get((gid, lid), [])

    def _meta_same_shape(other_meta: dict[str, Any]) -> bool:
        other_numel = shape_product(other_meta.get("shape"))
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
