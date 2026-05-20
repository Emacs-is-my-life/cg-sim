"""Unified timeline across multiple compiled graphs.

Builds a single execution sequence (``tasks``) over all graphs in their
profile-observed order, with a per-tensor identity that is *scoped to a
graph* (``(graph_id, compiled_tensor_id)``) — SDXL-style pipelines do not
share weights across graphs, and collisions on ``compiled_launch_id``
(each graph counts from 0) make global aliasing unsafe.

The legacy ``build_compiled_tensor_problem`` silently uses a globally-shared
``launch_to_node`` map, which lets two graphs' launches collide. This
module avoids the collision by filtering nodes by ``compiled_graph_id``
first.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sim.core.trace import Node, Trace

from .loader import MultiGraphSidecars
from .problem import (
    _DTYPE_BYTES,
    _GPU_RESOURCE_KINDS,
    _compute_wall_timeline,
    _is_gpu_node,
    _size_bytes_from_entry,
)


@dataclass
class GlobalTask:
    """One compute kernel on the unified pipeline timeline."""
    global_pos: int
    graph_id: int
    launch_id: int            # per-graph local launch_id (for op emission)
    node_id: int
    duration_ns: int
    start_ns: int             # synthetic: cumulative kernel duration (no idle gaps)
    end_ns: int               # synthetic: start_ns + duration_ns
    # Trace's actual wall-clock start_ns / end_ns (from the profile).
    # Includes aux/aten ops that aren't in tl.tasks but DO advance the
    # sim's clock — using this for arc-window placement keeps the LP's
    # peak prediction aligned with sim's actual moment-by-moment vram
    # use. Without it, the LP samples at synthetic-time iter-0 launches,
    # which can be 2× off from sim's wall time on workloads where aux
    # ops dominate (e.g. llama8b: 357 ms synthetic vs 693 ms sim wall).
    trace_start_ns: int = 0
    trace_end_ns: int = 0
    used_tensors: list[int] = field(default_factory=list)  # indices into tensors[]


@dataclass
class GlobalTensor:
    """A unique (graph_id, compiled_tensor_id) tensor on the unified timeline.

    ``storage_group_id`` (when present in the sidecar) identifies an
    alias group: two ``(graph_id, compiled_tensor_id)`` entries with the
    same group id refer to the same physical storage at compile time
    and should be coalesced into a single schedulable unit by
    ``coalesce_by_storage`` before scheduling.  When the bundle producer
    didn't emit it (older bundles, or compile paths where storage isn't
    determinable), the value is ``None`` and the tensor is treated as a
    singleton group keyed by ``(graph_id, compiled_tensor_id)``.
    """
    uid: int                  # index in UnifiedTimeline.tensors
    graph_id: int
    compiled_tensor_id: int
    graph_input_name: str
    size_bytes: int
    dtype: str
    entry: dict[str, Any]     # original sidecar entry
    uses: list[int] = field(default_factory=list)  # task global_pos values
    storage_group_id: Any = None   # opaque, compared by equality
    # Trace-runtime tids that this unified-timeline tensor backs. For
    # sidecar tensors, populated by storage_id pairing in
    # ``build_unified_timeline`` to point at the trace tid(s) sharing
    # this storage_group_id. For synth tensors, populated to the
    # specific trace tid(s) that motivated the synthesis. Empty list
    # means "unresolved at timeline build" — the scheduler should not
    # reference this tensor (see RC2 audit). Carrying these here lets
    # ``resolve_neutral_cgsim_tids`` skip per-node shape-disambiguation
    # entirely (RC1) and the scheduler emit pre-resolved cgsim_tids
    # without any injector heuristic (RC4).
    trace_tids: list[int] = field(default_factory=list)
    # RC3: earliest trace gpu consumer start_ns across this tensor's
    # trace_tids. Includes aten/aux ops, not just compiled kernels.
    # The scheduler uses this for ``c_feasibility`` (can the initial
    # prefetch fit before *any* runtime consumer, not just before the
    # first compiled-kernel consumer). When the value is 0 or smaller
    # than τ_h2d, streaming is infeasible — force cold-start, don't
    # let the injector demote later.
    earliest_consumer_ns: int = 0


@dataclass
class UnifiedTimeline:
    """A concatenated view of all graphs for cross-graph scheduling."""
    tasks: list[GlobalTask]
    tensors: list[GlobalTensor]
    graph_order: list[int]
    per_graph_tasks: dict[int, list[int]]   # gid -> list of global_pos (in order)
    per_graph_launch_to_task: dict[int, dict[int, int]]  # gid -> (launch_id -> global_pos)
    per_graph_hash: dict[int, str]
    total_duration_ns: int
    cpu_per_launch_ns: int = 0
    # Per-pipeline-call iteration count for each graph. Defaults to 1
    # (single invocation per pipeline call). For SDXL diffusion: UNet
    # has multiplicity = num_inference_steps × cfg_factor; TEs and VAE
    # are 1. Required for sound issuer placement in multi-iter
    # consumers — a graph A can only issue cross-graph H2D for graph
    # B's tensors if multiplicity[A] >= multiplicity[B] (otherwise A's
    # one fire starves B's later iterations).
    graph_multiplicity: dict[int, int] = field(default_factory=dict)
    # RC3: per-trace-tid list of *all* gpu trace consumer node_ids
    # (compiled kernels AND aten/aux ops). The injector's
    # coverage-repair pass walks the same set and demotes prefetched
    # tids whose earliest consumer runs before the schedule's first
    # arrival. Exposing it on the timeline lets schedulers shift
    # issue points / force cold-start for tensors whose earliest aux
    # consumer is too close to t=0 — i.e. fix the demote at the
    # source instead of letting the injector silently patch it.
    trace_tid_consumers: dict[int, list[int]] = field(default_factory=dict)
    # Per-trace-tid earliest gpu consumer start_ns. Same map flattened
    # to a single timestamp for fast LP feasibility checks.
    trace_tid_earliest_consumer_ns: dict[int, int] = field(
        default_factory=dict
    )
    # Per-graph earliest gpu trace ns (= min start_ns of any gpu trace
    # node tagged with this compiled_graph_id). Required to convert
    # absolute trace_tid_earliest_consumer_ns values into within-graph
    # relative offsets the LP can compare against τ_h2d.
    graph_first_trace_ns: dict[int, int] = field(default_factory=dict)


def _safe_numel(shape: Any) -> int | None:
    """Product of shape's int-coercible entries, or None if shape unusable."""
    if not shape:
        return None
    n = 1
    for x in shape:
        try:
            n *= int(x)
        except (TypeError, ValueError):
            return None
    return n


def _graph_order_by_first_kernel(
    trace: Trace, node_to_graph_id: dict[int, int],
) -> list[int]:
    """Execution order = order of first GPU kernel per graph (by profile ``start_ns``)."""
    first_time: dict[int, int] = {}
    synth = 0
    synth_starts: dict[int, int] = {}
    for nid, node in trace.node_map.items():
        synth_starts[nid] = synth
        synth += int(node.compute_time_micros * 1_000)
    for nid, node in trace.node_map.items():
        gid = node_to_graph_id.get(nid, -1)
        if gid < 0 or not _is_gpu_node(node):
            continue
        raw = node.args.get("start_ns") or 0
        start = int(raw) if raw else synth_starts[nid]
        if gid not in first_time or start < first_time[gid]:
            first_time[gid] = start
    return sorted(first_time.keys(), key=lambda g: first_time[g])


def build_unified_timeline(
    trace: Trace,
    sidecars: MultiGraphSidecars,
    *,
    cpu_per_launch_ns: int = 0,
    graph_multiplicity: dict[int, int] | None = None,
    replicate_uses: bool = False,
) -> UnifiedTimeline:
    """Build a global timeline by concatenating per-graph kernel sequences.

    Graphs run sequentially in profile order. Each graph's kernels retain
    their per-graph ``launch_id`` so the wrapper filter can route emitted
    ops correctly. Tensors are scoped to the graph they live in (no
    cross-graph tensor sharing is assumed or attempted).

    When ``replicate_uses=False`` (default), each ``(graph_id, launch_id)``
    is recorded once — duplicates in the trace are dropped. This collapses
    multi-iter consumers (e.g., SDXL UNet running N times per pipeline)
    to a single pass; downstream LP schedulers see the compressed view
    and may underestimate H2D byte requirements.

    When ``replicate_uses=True``, every trace event is preserved. Tasks
    are sorted by ``start_ns`` so they appear in execution order, and a
    tensor's ``uses`` list contains every position the tensor is read
    (one entry per real GPU event, not per unique launch_id). The
    ``per_graph_launch_to_task`` map becomes ``dict[gid, dict[lid,
    list[int]]]`` because a single lid may map to multiple positions.
    Use this when a scheduler needs the true per-iter pattern.
    """
    # Per-node graph membership.
    node_to_graph_id: dict[int, int] = {}
    for nid, node in trace.node_map.items():
        gid = node.args.get("compiled_graph_id")
        if gid is None:
            continue
        node_to_graph_id[nid] = int(gid)

    graph_order = _graph_order_by_first_kernel(trace, node_to_graph_id)

    # Per-graph bucketing: only GPU nodes that carry a valid compiled_launch_id.
    # In replicate_uses=True mode, per_graph_lid maps lid → list[nid] (one
    # per occurrence). In dedup mode, per_graph_lid maps lid → first nid.
    per_graph_nodes: dict[int, list[int]] = {gid: [] for gid in graph_order}
    nid_to_lid: dict[int, int] = {}
    seen_lids: dict[int, set[int]] = {gid: set() for gid in graph_order}
    for nid, node in trace.node_map.items():
        if not _is_gpu_node(node):
            continue
        gid = node_to_graph_id.get(nid, -1)
        if gid not in per_graph_nodes:
            continue
        lid_raw = node.args.get("compiled_launch_id")
        if lid_raw is None:
            continue
        lid = int(lid_raw)
        if lid < 0:
            continue
        if not replicate_uses and lid in seen_lids[gid]:
            # Dedup mode: drop subsequent occurrences of the same lid.
            continue
        seen_lids[gid].add(lid)
        nid_to_lid[nid] = lid
        per_graph_nodes[gid].append(nid)

    # Sort each graph's nodes for deterministic execution order.
    # In dedup mode, sort by lid (one entry per lid). In replicate mode,
    # sort by (start_ns, lid) — primary by trace timestamp so multi-iter
    # copies appear in actual execution order; lid breaks ties when
    # timestamps are missing or equal.
    def _node_sort_key(nid: int) -> tuple:
        lid = nid_to_lid.get(nid, 0)
        if replicate_uses:
            raw = trace.node_map[nid].args.get("start_ns") or 0
            return (int(raw) if raw else 0, lid)
        return (lid,)

    for gid in graph_order:
        per_graph_nodes[gid].sort(key=_node_sort_key)

    # Compute per-kernel durations from compute_time_micros.
    def _dur_ns(nid: int) -> int:
        return int(trace.node_map[nid].compute_time_micros * 1_000)

    # Build global task list.
    tasks: list[GlobalTask] = []
    per_graph_tasks: dict[int, list[int]] = {gid: [] for gid in graph_order}
    # In replicate_uses=True mode, this maps lid → list[global_pos] (one
    # per occurrence). Dedup mode keeps the historical lid → int shape.
    per_graph_launch_to_task: dict[int, dict[int, Any]] = {
        gid: {} for gid in graph_order
    }
    per_graph_hash: dict[int, str] = {}

    # Compact GPU events back-to-back, removing the inter-event gaps
    # the profiler injects between record_function calls. Each event's
    # duration_ns from the trace is profiler-clean (the gaps are where
    # bookkeeping lives, not inside the events). Laying them out with
    # no gaps at their durations gives an iter wall ≈ max(GPU sum,
    # CPU sum) — close to real unprofiled wall — without needing any
    # calibration data.
    #
    # No CPU dispatch is added on top: cpu_per_launch_ns inflated wall
    # by ~70% on SDXL (586 ms vs ~350 ms reality) by treating CPU and
    # GPU as serial when in fact they overlap. This compaction pass
    # only sees GPU events anyway; CPU dispatch is implicit in the
    # gaps we just removed.
    global_pos = 0
    t_cursor = 0
    for gid in graph_order:
        lm = sidecars.launch_maps.get(gid, {})
        per_graph_hash[gid] = str(lm.get("compilation_hash", ""))
        for nid in per_graph_nodes[gid]:
            dur = _dur_ns(nid)
            start = t_cursor
            end = t_cursor + dur
            t_cursor = end
            lid = nid_to_lid.get(nid, 0)
            trace_s = int(trace.node_map[nid].args.get("start_ns") or 0)
            trace_e = int(trace.node_map[nid].args.get("end_ns") or trace_s)
            tasks.append(GlobalTask(
                global_pos=global_pos,
                graph_id=gid,
                launch_id=lid,
                node_id=nid,
                duration_ns=dur,
                start_ns=start,
                end_ns=end,
                trace_start_ns=trace_s,
                trace_end_ns=trace_e,
            ))
            per_graph_tasks[gid].append(global_pos)
            if replicate_uses:
                per_graph_launch_to_task[gid].setdefault(lid, []).append(global_pos)
            else:
                per_graph_launch_to_task[gid][lid] = global_pos
            global_pos += 1

    # Tensor table. Iterate sidecars in graph_order so per-graph tids stay
    # adjacent and tensors used earlier get smaller uids.
    tensors: list[GlobalTensor] = []
    for gid in graph_order:
        tmap = sidecars.tensor_maps.get(gid)
        if tmap is None:
            continue
        for entry in tmap.get("tensors", []):
            tid = int(entry["compiled_tensor_id"])
            size = _size_bytes_from_entry(entry)
            gname = str(entry.get("graph_input_name", f"g{gid}_tid{tid}"))
            uses: list[int] = []
            for ulid in entry.get("used_by_launch_ids", []):
                lid_int = int(ulid)
                hit = per_graph_launch_to_task[gid].get(lid_int)
                if hit is None:
                    continue
                if replicate_uses:
                    uses.extend(hit)
                else:
                    uses.append(hit)
            uses.sort()
            if not uses:
                continue
            uid = len(tensors)
            # Storage group: read straight from the sidecar.  Bundles
            # produced by the upstream-patched Inductor carry
            # ``storage_group_id`` per ctid (from compile-time
            # ``id(untyped_storage())``).  Bundles without it fall to
            # singleton groups keyed by ``(gid, ctid)`` in the
            # coalescer — same behaviour as no-coalescing.
            sgid = entry.get("storage_group_id", None)
            tensors.append(GlobalTensor(
                uid=uid,
                graph_id=gid,
                compiled_tensor_id=tid,
                graph_input_name=gname,
                size_bytes=size,
                dtype=str(entry.get("dtype", "")),
                entry=entry,
                uses=uses,
                storage_group_id=sgid,
            ))
            for pos in uses:
                tasks[pos].used_tensors.append(uid)

    # === Fix A: synthesize one tensor per trace WEIGHT/LEAF tid ===
    #
    # The compile pipeline can produce under-specified tensor maps:
    # ``compiled_tensor_map_graph<gid>.json`` may record only N entries
    # whose ``used_by_launch_ids`` cover a launch even though the runtime
    # trace shows that launch's GPU nodes reading M > N WEIGHT/LEAF
    # tensors of the same (shape, dtype). Without intervention these
    # "shadow" weights are invisible to the scheduler — and the
    # injector's resolver can't bind them either, leaving them un-claimed
    # at consumer time and triggering ``SCHEDULER_DEADLOCK`` under
    # aggressive streaming.
    #
    # **Synthesis policy (post-2026-05 dedup-off, "option 2"):** every
    # trace WEIGHT/LEAF tid that's an input to one of our compute tasks
    # gets its OWN ``GlobalTensor`` entry. The synth's
    # ``storage_group_id`` is the trace tid's ``storage_id``, and a
    # per-graph ``(graph_id, storage_id) -> uid`` registry collapses
    # duplicate synth attempts WITHIN the synthesis loop (so a tid
    # consumed by 12 tasks resolves to one uid, not 12).
    #
    # We DO NOT pair surplus trace tids back into the sidecar uids that
    # already cover the same (numel, size) shape group. That pairing
    # (used through 2026-04) collapsed each trace tid into a single
    # sidecar uid — the injector retargets per-trace-tid, so after the
    # collapse the injector could only retarget ONE trace tid per shared
    # storage, leaving sibling trace tids cuda-resident at runtime. The
    # downstream effect was a 1148-uid → 893-uid drop on llama3b that
    # made jit_sim_prune lose its ability to fit the 5 GiB cap.
    #
    # Synthetic entries get a high ``compiled_tensor_id`` (≥
    # ``SYNTH_CTID_BASE``) and a high ``graph_input_idx`` so the
    # injector's per-node resolver places them at the *end* of the
    # same-shape position list. ``storage_group_id`` is set to the
    # trace tensor's ``storage_id`` so ``coalesce_by_storage`` unifies
    # cross-graph references to the same physical storage exactly the
    # way Inductor's storage_group_id field would have. Sidecar uids
    # that don't get any trace_tids appended (their compile-side entry
    # never matched a runtime input) are left as "ghost" uids —
    # downstream schedulers detect them via ``not tensor.trace_tids``
    # and must skip them since the injector can't retarget anything on
    # their behalf.
    SYNTH_CTID_BASE = 1_000_000_000
    SYNTH_INPUT_IDX_BASE = 999_000_000
    syn_count = 0
    # (graph_id, trace_storage_id) -> uid in tensors[]
    syn_by_graph_storage: dict[tuple[int, int], int] = {}
    sidecar_storage_paired = 0
    for task_idx, task in enumerate(tasks):
        node = trace.node_map.get(task.node_id)
        if node is None:
            continue
        # Group trace WEIGHT/LEAF inputs of this node by (numel, size_bytes).
        # We don't pre-count sidecar matches for the same shape — the
        # dedup-off policy below synthesizes one uid per trace tid
        # regardless of whether the sidecar already had a same-shape
        # entry, so the existing_count bookkeeping the legacy version
        # carried here would be dead weight.
        trace_tids_per_key: dict[tuple[int, int], list[int]] = {}
        for tid in node.input_tensors or []:
            tt = trace.tensor_map.get(tid)
            if tt is None:
                continue
            ttype = (tt.args or {}).get("tensor_type")
            if ttype not in ("WEIGHT", "LEAF"):
                continue
            shp = (tt.args or {}).get("shape")
            numel = _safe_numel(shp)
            if numel is None:
                continue
            size_bytes = int(getattr(tt, "size_bytes", 0) or 0)
            if size_bytes <= 0:
                # Fall back to numel × dtype-bytes if size missing
                dtype_bytes = _DTYPE_BYTES.get(
                    str((tt.args or {}).get("dtype") or ""), 0
                )
                size_bytes = int(numel * dtype_bytes)
            if size_bytes <= 0:
                continue
            trace_tids_per_key.setdefault((numel, size_bytes), []).append(int(tid))
        # Dedup-off policy: synthesize a uid for EVERY trace WEIGHT/
        # LEAF input of this node (subject to the per-graph storage_id
        # dedup in syn_by_graph_storage below). Setting ``surplus =
        # trace_tids_sorted`` skips the legacy "pair the first
        # existing_count trace tids with sidecar uids" step that used
        # to collapse trace tids into sidecar uids and broke per-tid
        # retargeting. The synth registry below still prevents
        # per-task surplus loops from inflating the same physical
        # storage into multiple uids.
        for key, trace_tids in trace_tids_per_key.items():
            trace_tids_sorted = sorted(trace_tids)
            surplus = trace_tids_sorted
            for trace_tid in surplus:
                tt = trace.tensor_map[trace_tid]
                storage_id = int((tt.args or {}).get("storage_id") or trace_tid)
                key_g = (task.graph_id, storage_id)
                if key_g in syn_by_graph_storage:
                    # Already synthesized in this graph for this storage —
                    # just record this task as a use. (We still dedup
                    # synth-to-synth so per-task surplus loops don't
                    # multiply uids for the same physical storage.)
                    uid = syn_by_graph_storage[key_g]
                    if task_idx not in tensors[uid].uses:
                        tensors[uid].uses.append(task_idx)
                        tensors[uid].uses.sort()
                    if task.launch_id not in tensors[uid].entry["used_by_launch_ids"]:
                        tensors[uid].entry["used_by_launch_ids"].append(int(task.launch_id))
                    if uid not in tasks[task_idx].used_tensors:
                        tasks[task_idx].used_tensors.append(uid)
                    if int(trace_tid) not in tensors[uid].trace_tids:
                        tensors[uid].trace_tids.append(int(trace_tid))
                    continue
                # New synthesis.
                synth_ctid = SYNTH_CTID_BASE + syn_count
                synth_input_idx = SYNTH_INPUT_IDX_BASE + syn_count
                syn_count += 1
                size_bytes = int(getattr(tt, "size_bytes", 0) or 0)
                if not size_bytes:
                    # Compute from shape × dtype if size_bytes missing.
                    numel = _safe_numel((tt.args or {}).get("shape")) or 0
                    dtype_bytes = _DTYPE_BYTES.get(
                        str((tt.args or {}).get("dtype") or ""), 0
                    )
                    size_bytes = int(numel * dtype_bytes)
                synth_entry = {
                    "compiled_tensor_id": int(synth_ctid),
                    "graph_input_name": f"synth_storage{storage_id}",
                    "graph_input_idx": int(synth_input_idx),
                    "dtype": str((tt.args or {}).get("dtype") or ""),
                    "shape": list((tt.args or {}).get("shape") or []),
                    "storage_group_id": int(storage_id),
                    "used_by_launch_ids": [int(task.launch_id)],
                    "size_bytes": int(size_bytes),
                }
                uid = len(tensors)
                tensors.append(GlobalTensor(
                    uid=uid,
                    graph_id=int(task.graph_id),
                    compiled_tensor_id=int(synth_ctid),
                    graph_input_name=synth_entry["graph_input_name"],
                    size_bytes=int(size_bytes),
                    dtype=synth_entry["dtype"],
                    entry=synth_entry,
                    uses=[task_idx],
                    storage_group_id=int(storage_id),
                    trace_tids=[int(trace_tid)],
                ))
                tasks[task_idx].used_tensors.append(uid)
                syn_by_graph_storage[key_g] = uid
    if syn_count or sidecar_storage_paired:
        print(
            f"[multigraph_timeline] synthesized {syn_count} tensor entries "
            f"for trace inputs missing from the compile-side tensor map "
            f"(WEIGHT/LEAF tensors with same-shape duplicates); "
            f"paired {sidecar_storage_paired} sidecar uids with trace storage_id "
            f"(dedup against surplus synth).",
            flush=True,
        )

    # ---- RC2: backfill trace_tids for sidecar tensors that didn't get
    # paired through the surplus loop (they had a storage_group_id from
    # the bundle but were never matched against trace.tensor_map). Index
    # trace by storage_id once and look up. Any tensor still left with
    # empty trace_tids after this is genuinely unschedulable — log loud.
    storage_to_trace_tids: dict[int, list[int]] = {}
    for trace_tid, tt in trace.tensor_map.items():
        sid = (tt.args or {}).get("storage_id")
        if sid is None:
            continue
        storage_to_trace_tids.setdefault(int(sid), []).append(int(trace_tid))

    backfilled = 0
    unschedulable_count = 0
    unschedulable_bytes = 0
    samples_unsch: list[tuple[int, int, int, int]] = []
    for t in tensors:
        if t.trace_tids:
            continue
        if t.storage_group_id is not None:
            sid = int(t.storage_group_id)
            hits = storage_to_trace_tids.get(sid)
            if hits:
                t.trace_tids = list(hits)
                backfilled += 1
                continue
        # No storage_group_id OR no trace tid backs this storage. The
        # tensor is unschedulable: the scheduler should not reference
        # it, because the injector won't be able to find a runtime
        # tid to gate / retarget.
        unschedulable_count += 1
        unschedulable_bytes += int(t.size_bytes)
        if len(samples_unsch) < 5:
            samples_unsch.append(
                (int(t.graph_id), int(t.compiled_tensor_id),
                 int(t.size_bytes), int(t.uid))
            )
    if backfilled or unschedulable_count:
        print(
            f"[multigraph_timeline:diag] trace_tid coverage: "
            f"backfilled_by_storage_id={backfilled} "
            f"unschedulable={unschedulable_count} "
            f"({unschedulable_bytes/1e6:.1f}MB)",
            flush=True,
        )
        if samples_unsch:
            print(
                f"[multigraph_timeline:diag]   unschedulable samples "
                f"(gid, ctid, size_b, uid): {samples_unsch}",
                flush=True,
            )

    # ---- RC3: per-trace-tid all-gpu-consumer index ----
    #
    # Build the full ``trace_tid -> [gpu consumer node_id]`` map from
    # trace.node_map ONCE, plus the flattened ``trace_tid ->
    # earliest_consumer_ns`` lookup. The injector computes the same
    # information (its ``consumers_by_tid`` map) at inject time; by
    # exposing it on the timeline we let the scheduler shift issue
    # times to cover aux ops BEFORE handing the schedule to the
    # injector. Without this, the injector silently demotes tids
    # whose earliest aux consumer runs before the schedule's first
    # arrival — see RC3 in the residual-gap plan.
    #
    # TIME-SPACE NOTE: trace.node_map uses absolute profile-time ns,
    # NOT the synthetic cumulative time tl.tasks uses (each
    # GlobalTask.start_ns is built by summing kernel durations from
    # zero). So values from this map must be compared against other
    # *trace-time* values. The scheduler converts at use time by
    # taking deltas against ``graph_first_trace_ns`` below.
    trace_tid_consumers: dict[int, list[int]] = defaultdict(list)
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in ("gpu_stream", "gpu", "gpu_runtime"):
            continue
        for tid in node.input_tensors or []:
            trace_tid_consumers[int(tid)].append(int(nid))
    trace_tid_earliest_consumer_ns: dict[int, int] = {}
    for tid, nids in trace_tid_consumers.items():
        best_ns = None
        for nid in nids:
            n = trace.node_map.get(nid)
            if n is None:
                continue
            try:
                s_ns = int((n.args or {}).get("start_ns") or 0)
            except (TypeError, ValueError):
                continue
            if best_ns is None or s_ns < best_ns:
                best_ns = s_ns
        if best_ns is not None:
            trace_tid_earliest_consumer_ns[int(tid)] = int(best_ns)

    # Per-graph earliest trace-time start_ns over all gpu nodes. Used
    # to convert absolute earliest_consumer_ns into a within-graph
    # relative time the scheduler can compare against τ_h2d.
    graph_first_trace_ns: dict[int, int] = {}
    for nid, node in trace.node_map.items():
        rk = str((node.args or {}).get("resource_kind") or "")
        if rk not in ("gpu_stream", "gpu", "gpu_runtime"):
            continue
        gid_raw_v = node.args.get("compiled_graph_id")
        if gid_raw_v is None:
            continue
        try:
            gid_raw = int(gid_raw_v)
        except (TypeError, ValueError):
            continue
        if gid_raw < 0:
            continue
        try:
            s_ns = int((node.args or {}).get("start_ns") or 0)
        except (TypeError, ValueError):
            continue
        prev = graph_first_trace_ns.get(gid_raw)
        if prev is None or s_ns < prev:
            graph_first_trace_ns[gid_raw] = s_ns

    # Fill GlobalTensor.earliest_consumer_ns: min across all its
    # trace_tids' earliest gpu consumer ns. A tensor with no
    # trace_tids gets 0 (treated as "earliest at t=0", forces
    # cold-start at LP since no prefetch can fit before).
    for t in tensors:
        if not t.trace_tids:
            t.earliest_consumer_ns = 0
            continue
        candidates = [
            trace_tid_earliest_consumer_ns[int(tt)]
            for tt in t.trace_tids
            if int(tt) in trace_tid_earliest_consumer_ns
        ]
        t.earliest_consumer_ns = min(candidates) if candidates else 0

    # Default multiplicity = 1 for any graph not explicitly provided.
    mult: dict[int, int] = {gid: 1 for gid in graph_order}
    if graph_multiplicity:
        for gid, n in graph_multiplicity.items():
            if gid in mult:
                mult[gid] = max(1, int(n))

    total_duration_ns = max((t.end_ns for t in tasks), default=0)

    return UnifiedTimeline(
        tasks=tasks,
        tensors=tensors,
        graph_order=graph_order,
        per_graph_tasks=per_graph_tasks,
        per_graph_launch_to_task=per_graph_launch_to_task,
        per_graph_hash=per_graph_hash,
        total_duration_ns=total_duration_ns,
        cpu_per_launch_ns=cpu_per_launch_ns,
        graph_multiplicity=mult,
        trace_tid_consumers=dict(trace_tid_consumers),
        trace_tid_earliest_consumer_ns=trace_tid_earliest_consumer_ns,
        graph_first_trace_ns=graph_first_trace_ns,
    )


def emit_h2d_op(
    tensor: GlobalTensor,
    task: GlobalTask,
    *,
    duration_ns: int,
    reason: str,
    after_launch_id: int = -1,
) -> dict[str, Any]:
    """Build a vram_prefetch_h2d op dict routed to the tensor's graph."""
    return {
        "type": "vram_prefetch_h2d",
        "tensor_name": tensor.graph_input_name,
        "tensor_kind": "WEIGHT",
        "before_node": int(task.node_id),
        "after_node": -1,
        "duration_ns": int(duration_ns),
        "size_bytes": int(tensor.size_bytes),
        "reason": reason,
        "before_launch_id": int(task.launch_id),
        "after_launch_id": int(after_launch_id),
        "compiled_tensor_id": int(tensor.compiled_tensor_id),
        "compiled_graph_id": int(tensor.graph_id),
        "compilation_hash": "",
        "compiled_graph_input_name": tensor.graph_input_name,
    }


def emit_d2h_op(
    tensor: GlobalTensor,
    task: GlobalTask,
    *,
    duration_ns: int,
    reason: str,
) -> dict[str, Any]:
    """Build a vram_evict_d2h op dict anchored in the tensor's graph."""
    return {
        "type": "vram_evict_d2h",
        "tensor_name": tensor.graph_input_name,
        "tensor_kind": "WEIGHT",
        "after_node": int(task.node_id),
        "duration_ns": int(duration_ns),
        "size_bytes": int(tensor.size_bytes),
        "reason": reason,
        "after_launch_id": int(task.launch_id),
        "compiled_tensor_id": int(tensor.compiled_tensor_id),
        "compiled_graph_id": int(tensor.graph_id),
        "compilation_hash": "",
        "compiled_graph_input_name": tensor.graph_input_name,
    }


def emit_cold_start(
    tensor: GlobalTensor,
    first_task: GlobalTask,
    *,
    reason: str,
) -> dict[str, Any]:
    """Build a cold_start_prefetch entry (fires once at module init)."""
    return {
        "tensor_name": tensor.graph_input_name,
        "tensor_kind": "WEIGHT",
        "reason": reason,
        "miss_ns": 0,
        "attach_before_node": int(first_task.node_id),
        "eager_start": True,
        "before_launch_id": int(first_task.launch_id),
        "compiled_tensor_id": int(tensor.compiled_tensor_id),
        "compiled_graph_id": int(tensor.graph_id),
        "compilation_hash": "",
        "compiled_graph_input_name": tensor.graph_input_name,
    }


def build_node_timeline(tl: UnifiedTimeline, trace: Trace) -> tuple[list[int], list[int]]:
    """Return (node_starts, node_ends) in trace.node_map iteration order.

    Compute nodes covered by the timeline use their assigned start/end; all
    other nodes are filled with cumulative compute_time_micros so the output
    JSON's ``nodes`` section stays consistent with the runtime schema.
    """
    task_by_node: dict[int, GlobalTask] = {
        t.node_id: t for t in tl.tasks
    }
    starts: list[int] = []
    ends: list[int] = []
    t = 0
    for nid, node in trace.node_map.items():
        if nid in task_by_node:
            tk = task_by_node[nid]
            starts.append(tk.start_ns)
            ends.append(tk.end_ns)
            t = tk.end_ns
            continue
        starts.append(t)
        dur = int(node.compute_time_micros * 1_000)
        t += dur
        ends.append(t)
    return starts, ends
