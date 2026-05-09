"""Coalesce ``UnifiedTimeline.tensors`` by ``storage_group_id``.

Inductor's compiled tensor map can carry a per-ctid ``storage_group_id``
that identifies the underlying storage (see
``torch/_inductor/codegen/wrapper.py:_build_tensor_id_map``).  Two
compile-graph inputs that point at the same storage receive the same
group id; downstream schedulers should treat them as a single
schedulable unit because residency, prefetch, and eviction all key on
the storage, not on the compile-graph-input identity.

This module exposes a single helper, :func:`coalesce_by_storage`, that
turns a ``UnifiedTimeline.tensors`` list into a list of
``(representative, members)`` pairs:

  - ``representative`` is one ``GlobalTensor`` per group; its
    ``uses`` list is the *union* of all members' uses, sorted in
    timeline order, and its ``size_bytes`` is the max across members
    (a tight upper bound on the storage's true byte count).
  - ``members`` is the full list of ``GlobalTensor`` objects in the
    group, including the representative.

Tensors whose ``storage_group_id is None`` (older bundles, or compile
paths where the storage couldn't be determined upstream) are treated
as singletons — each becomes its own group keyed by
``(graph_id, compiled_tensor_id)`` — preserving today's per-ctid
scheduling behaviour for un-annotated bundles.

Schedulers call this once at the top of ``solve_neutral`` and iterate
over the representatives instead of the raw ``tl.tensors`` list.  The
emission step still emits prefetch/evict ops keyed by the
representative's ``(graph_id, compiled_tensor_id)`` — the runtime
injector resolves it to one cgsim_tid via the loader's storage merge,
which is the very same merge the upstream group id describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .multigraph_timeline import GlobalTensor


@dataclass
class _Group:
    representative: GlobalTensor
    members: list[GlobalTensor]


def _group_key(t: GlobalTensor) -> tuple:
    """Stable per-storage key.

    If the upstream emitted a ``storage_group_id``, key by
    ``(graph_id, storage_group_id)``.  Otherwise fall back to
    ``(graph_id, compiled_tensor_id)`` — singleton group.

    Note: storage_group_id is graph-local in the upstream emitter
    (it's just ``id(untyped_storage())`` from the compile process),
    but two different graphs that happened to see the same Python id
    must NOT be coalesced — they're different compiles.  The
    ``graph_id`` prefix prevents that.
    """
    sgid = t.storage_group_id
    if sgid is None:
        return ("singleton", t.graph_id, t.compiled_tensor_id)
    return ("storage", t.graph_id, sgid)


def coalesce_by_storage(
    tensors: list[GlobalTensor],
) -> list[tuple[GlobalTensor, list[GlobalTensor]]]:
    """Group ``tensors`` by storage and synthesize one representative per group.

    The representative is a *new* GlobalTensor object that shares
    identity (``uid``, ``graph_id``, ``compiled_tensor_id``,
    ``graph_input_name``) with the first member of the group, but
    carries the unioned ``uses`` and the max ``size_bytes`` across
    members.  Mutating the representative does not affect the
    underlying ``tensors`` list (callers can iterate the
    representatives without losing the original layout).

    Returns
    -------
    list of ``(representative, members)`` pairs in original-tensor
    order (the representative is the first member of each group).
    """
    by_key: dict[tuple, list[GlobalTensor]] = {}
    order: list[tuple] = []
    for t in tensors:
        k = _group_key(t)
        if k not in by_key:
            by_key[k] = []
            order.append(k)
        by_key[k].append(t)

    out: list[tuple[GlobalTensor, list[GlobalTensor]]] = []
    for k in order:
        members = by_key[k]
        first = members[0]
        if len(members) == 1:
            # Trivially a singleton — pass through with no copy needed.
            out.append((first, members))
            continue
        # Union the uses across the group; max the size; keep the
        # first member as representative for stable identity.
        unioned_uses = sorted({pos for m in members for pos in m.uses})
        max_size = max(m.size_bytes for m in members)
        rep = GlobalTensor(
            uid=first.uid,
            graph_id=first.graph_id,
            compiled_tensor_id=first.compiled_tensor_id,
            graph_input_name=first.graph_input_name,
            size_bytes=max_size,
            dtype=first.dtype,
            entry=first.entry,
            uses=unioned_uses,
            storage_group_id=first.storage_group_id,
        )
        out.append((rep, members))
    return out


def coalesced_size_total(
    tensors: list[GlobalTensor],
) -> int:
    """Sum of ``representative.size_bytes`` across coalesced groups.

    Equals ``sum(t.size_bytes for t in tensors)`` minus the duplicate
    bytes that aliased ctids would have contributed.  Useful for
    bandwidth budgets: this is how many *unique* bytes the scheduler
    is actually committing to move.
    """
    return sum(rep.size_bytes for rep, _ in coalesce_by_storage(tensors))
