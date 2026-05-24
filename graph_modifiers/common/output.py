"""Write ``jit_sim_prune_schedule.json`` compatible with PyTorch runtime.

The JSON schema expected by ``torch/_inductor/weight_streaming/plan.py``
``load_io_schedule`` has four top-level sections:

* ``summary``: strategy metadata (free-form dict).
* ``nodes``: list of ``{idx, name, resource_kind, start_ns, end_ns}``
  mirroring the profile's node timeline.
* ``io_operations``: list of dicts with ``type`` ∈ {``prefetch``,
  ``vram_prefetch_h2d``, ``vram_evict_d2h``}, plus node/launch anchors.
* ``cold_start_prefetches``: list of dicts loaded eagerly at startup.
* ``spill_decisions``: DRAM eviction records (we leave empty).

We also emit ``compilation_hash`` so Inductor's plan-loader can warn on
mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sim.core.trace import Trace


def _node_section(trace: Trace, node_starts: list[int], node_ends: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (nid, node) in enumerate(trace.node_map.items()):
        out.append({
            "idx": int(nid),
            "name": str(node.name),
            "resource_kind": str(node.args.get("resource_kind") or ""),
            "start_ns": int(node_starts[i]),
            "end_ns": int(node_ends[i]),
        })
    return out


def write_schedule_json(
    path: str | Path,
    *,
    trace: Trace,
    node_starts: list[int],
    node_ends: list[int],
    io_operations: list[dict[str, Any]],
    cold_start_prefetches: list[dict[str, Any]] | None = None,
    spill_decisions: list[dict[str, Any]] | None = None,
    summary: dict[str, Any],
    compilation_hash: str = "",
) -> None:
    doc = {
        "summary": dict(summary),
        "nodes": _node_section(trace, node_starts, node_ends),
        "io_operations": list(io_operations),
        "spill_decisions": list(spill_decisions or []),
        "cold_start_prefetches": list(cold_start_prefetches or []),
        "steady_state_resident": [],
    }
    if compilation_hash:
        doc["compilation_hash"] = compilation_hash

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
