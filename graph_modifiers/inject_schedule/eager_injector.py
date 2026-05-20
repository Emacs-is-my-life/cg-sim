"""Eager-mode schedule injector.

The standard ``inject_schedule_into_trace`` consumes a NeutralSchedule
keyed by ``(compiled_graph_id, compiled_tensor_id)`` — fields populated
only when the bundle was built via torch.compile / Inductor. Eager-mode
bundles (``runtime_role`` = ``cpu_leaf`` + ``gpu_runtime``, with no
``compiled_graph_id`` set) carry their schedule via cgsim ``node_id`` /
``tid`` directly. This injector consumes that format.

Schema (``schema = "hf_accelerate_eager_v1"`` or any future eager
variant):

    {
      "schema": "hf_accelerate_eager_v1",
      "xfer_arrivals": [
        {"issuer_node_id": int, "consumer_node_id": int,
         "cgsim_tids": [int, ...], "size_bytes": int}, ...
      ],
      "evict_after_node": {"<node_id>": [tid, ...], ...},
      "evictable_tensor_ids": [int, ...],
      "cold_start_tids": [int, ...],   # stay cuda-resident
      "streamed_tids": [int, ...],     # retarget device cuda → cpu
      "meta": {...}
    }
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sim.core.trace import Trace


_GPU_ROLES = ("gpu_runtime", "gpu_stream", "gpu")


def _require_eager_format(doc: dict[str, Any]) -> None:
    schema = str(doc.get("schema") or "")
    if not schema.endswith("_eager_v1"):
        raise Exception(
            f"[inject_eager_schedule] schema {schema!r} unsupported; "
            f"expected an *_eager_v1 schedule."
        )


def _validate_node(trace: Trace, node_id: int, role: str) -> None:
    node = trace.node_map.get(int(node_id))
    if node is None:
        raise Exception(
            f"[inject_eager_schedule] {role} node_id={node_id} not in trace."
        )
    rk = str((node.args or {}).get("resource_kind") or "")
    # Consumers must be GPU (they consume cuda-resident weights).
    # Issuers may be GPU OR CPU dispatchers (cpu_thread cuLaunchKernel
    # nodes) — DAV fires prefetches on any ComputeJob retire.
    if role == "consumer" and rk not in _GPU_ROLES:
        raise Exception(
            f"[inject_eager_schedule] {role} node_id={node_id} is not "
            f"a GPU node (resource_kind={rk!r})."
        )
    if role == "issuer" and rk not in _GPU_ROLES and rk not in ("cpu_thread", "cpu_leaf"):
        raise Exception(
            f"[inject_eager_schedule] {role} node_id={node_id} has "
            f"unsupported resource_kind={rk!r}."
        )


def inject_eager_schedule_into_trace(
    trace: Trace,
    schedule: dict[str, Any] | str | Path,
    *,
    disable_evict: bool = False,
) -> Trace:
    """Annotate ``trace`` with prefetch/evict ops from an eager-mode
    schedule. Modifies the trace in place.

    Effects:
      - Populates ``trace.args["xfer_arrivals"]`` (consumed by
        ``DeviceAwareVanillaAsync._build_arrival_index``).
      - Populates ``trace.args["evict_after_node"]`` (consumed by
        the retire path).
      - Populates ``trace.args["evictable_tensor_ids"]``.
      - Retargets every tid in ``streamed_tids`` from cuda → cpu so
        the async scheduler treats it as absent on boot.
    """
    if isinstance(schedule, (str, Path)):
        with open(schedule) as f:
            doc = json.load(f)
    else:
        doc = schedule

    _require_eager_format(doc)

    arrivals_in = doc.get("xfer_arrivals") or []
    arrivals_out: list[dict[str, Any]] = []
    for a in arrivals_in:
        issuer = int(a["issuer_node_id"])
        consumer = int(a["consumer_node_id"])
        _validate_node(trace, issuer, "issuer")
        _validate_node(trace, consumer, "consumer")
        tids = [int(t) for t in (a.get("cgsim_tids") or [])]
        if not tids:
            continue
        arrivals_out.append({
            "issuer_node_id": issuer,
            "consumer_node_id": consumer,
            "cgsim_tids": tids,
            "size_bytes": int(a.get("size_bytes") or 0),
        })

    evict_in = doc.get("evict_after_node") or {}
    evict_out: dict[int, set[int]] = defaultdict(set)
    for nid_key, tids in evict_in.items():
        nid = int(nid_key)
        if nid not in trace.node_map:
            raise Exception(
                f"[inject_eager_schedule] evict node_id={nid} not in trace."
            )
        evict_out[nid].update(int(t) for t in tids)

    evictable = set(int(t) for t in (doc.get("evictable_tensor_ids") or []))

    streamed = set(int(t) for t in (doc.get("streamed_tids") or []))
    retargeted = 0
    skipped_missing = 0
    for tid in streamed:
        t = trace.tensor_map.get(int(tid))
        if t is None:
            skipped_missing += 1
            continue
        ttype = (t.args or {}).get("tensor_type")
        if ttype not in ("WEIGHT", "LEAF"):
            # Stay safe: only retarget permanents. INTERMEDIATE tensors
            # shouldn't be in streamed_tids anyway.
            continue
        cur = str((t.args or {}).get("device") or "").lower()
        if cur.startswith("cuda"):
            t.args["device"] = "cpu"
            retargeted += 1

    if arrivals_out:
        existing = trace.args.setdefault("xfer_arrivals", [])
        existing.extend(arrivals_out)
    if evict_out and not disable_evict:
        existing_ev = trace.args.setdefault("evict_after_node", {})
        for nid, tids in evict_out.items():
            existing_ev.setdefault(nid, set()).update(tids)
    if evictable and not disable_evict:
        trace.args.setdefault("evictable_tensor_ids", set()).update(evictable)

    meta = doc.get("meta") or {}
    print(
        f"[inject_eager_schedule:{doc.get('schema')}] "
        f"arrivals={len(arrivals_out)} "
        f"evict_anchors={len(evict_out)} "
        f"evictable_tids={len(evictable)} "
        f"retargeted_cpu={retargeted} "
        f"(missing_tid_skipped={skipped_missing}) "
        f"knobs={meta.get('knobs')}",
        flush=True,
    )
    if disable_evict:
        print("[inject_eager_schedule] disable_evict=True — dropped evict map",
              flush=True)

    return trace
