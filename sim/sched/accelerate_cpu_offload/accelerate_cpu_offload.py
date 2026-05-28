"""Scheduler that simulates ``accelerate.cpu_offload(..., offload_buffers=False)``
in cg-sim.

Strategy
--------
The offload trace's profiler records each ``Memcpy HtoD`` event as an
alias-style pair: the cuda dst tid sits in both ``input_tensors`` and
``output_tensors``, and the same cuda dst tid is reused across all 15
generated tokens. The loader's ``_mark_implicit_inputs`` heuristic
mis-classifies these (and many other alias-chained activations) as
``INPUT``, which fills VRAM at layout phase 0 and aborts the run.

This compile stage rewrites the trace so cg-sim's natural CONTEXT
lifecycle can handle it:

1. **Dealias every Memcpy HtoD** — remove the cuda dst tid from
   ``input_tensors`` so the Memcpy looks like a real producer.

2. **Per-invocation tid splitting** — each leaf invocation contains
   one Memcpy per paged tensor; mint a fresh tid for that Memcpy's
   output and rewire the invocation-local consumers. Each fresh tid
   now has a single-invocation lifetime, so cg-sim's natural release
   fires at the last consumer per token (matching accelerate's
   ``post_forward`` drop-to-meta).

3. **Aggressive cuda-INPUT → CONTEXT demotion** — accelerate's
   per-leaf hook stack creates so many alias chains (aten::to /
   detach_ / set_module_tensor_to_device / view / unsqueeze /
   transpose / in-place ops) that the loader's heuristic over-marks
   transient activations as INPUT. Legitimate persistent cuda state
   in this workload is classified as WEIGHT (e.g. rotary_emb's
   ``inv_freq``) — anything else cuda-homed is genuinely transient.

4. **Explicit evict_after_node hints** for the freshly-minted tids
   at each invocation's last GPU node, as belt-and-suspenders to
   bound VRAM if the natural release misses a tid.

5. **Rebuild DAV's lifetime tracking** — ``__init__`` ran against
   the pre-rewrite graph; refresh ``_remaining_consumers``,
   ``_remaining_cpu_consumers``, and ``_consumers_by_tid`` on the
   rewritten graph.

The trace's Memcpy HtoD nodes run as gpu_runtime compute jobs with
their recorded durations, contributing the pageable-with-driver-staged-
bounce H2D wall-time (~14 GB/s on RTX 4090, per
``docs/offload-schemes/accelerate_cpu-offload_buffers-false.md:239-246``)
directly to the simulated wall-clock.

Known limitations
-----------------
The peak-VRAM estimate is ≈ 2× the reference: cg-sim's runtime keeps
both the previous leaf's cuda mirror and the next leaf's cuda mirror
resident momentarily when the next leaf's Memcpy starts before the
previous mirror's last consumer retires. This is a modeling artifact
of cg-sim's natural CONTEXT release vs. the explicit evict_after_node
timing, not the rewrite itself.

The 8B trace deadlocks partway through (cg-sim cannot make progress
because some downstream aten::view / aten::_unsafe_view alias op
cannot find a region for its input). 3B completes end-to-end. The
exact root cause is in cg-sim's per-tid region tracking when a
shared activation tid is touched by both a real producer and an
alias-style op in the same time window.

Reference: ``docs/offload-schemes/accelerate_cpu-offload_buffers-false.md``.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, TYPE_CHECKING

from sim.core.trace import Trace
from sim.sched.device_aware_vanilla_async import DeviceAwareVanillaAsync

if TYPE_CHECKING:
    from sim.core.log import Log
    from sim.core.system import System


_MAX_PARENT_WALK_DEPTH = 8


class _LeafEpoch:
    __slots__ = ("first_node", "first_gpu_nid", "last_gpu_nid")

    def __init__(self, first_node: int) -> None:
        self.first_node = first_node
        self.first_gpu_nid: int | None = None
        self.last_gpu_nid: int | None = None


class AccelerateCpuOffload(DeviceAwareVanillaAsync):
    """Plans accelerate-cpu_offload paging via trace rewrite at compile time."""

    def __init__(
        self,
        obj_id: int,
        name: str,
        log: "Log",
        sys: "System",
        args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(obj_id, name, log, sys, args)

    def compile(self, trace: Trace) -> None:
        # 1. Dealias every Memcpy HtoD node.
        memcpy_h2d_dst_tids = self._dealias_memcpy_h2d(trace)

        # 2. Resolve module paths and enumerate leaf invocations.
        node_module = self._resolve_node_modules(trace)
        leaf_paths = self._discover_leaf_paths(trace, node_module)
        leaf_invocations = self._enumerate_leaf_invocations(
            trace, node_module, leaf_paths,
        )

        # 3. Per-invocation tid splitting.
        split_stats = self._split_memcpy_outputs_per_invocation(
            trace, leaf_invocations, memcpy_h2d_dst_tids,
        )

        # 4. Aggressive cuda-INPUT → CONTEXT demotion.
        n_demoted = 0
        for tid, tensor in trace.tensor_map.items():
            dev = str(tensor.args.get("device", "")).lower()
            if not dev.startswith("cuda"):
                continue
            if tensor.args.get("tensor_type") == "INPUT":
                tensor.args["tensor_type"] = "CONTEXT"
                tensor.args.pop("implicit_input", None)
                n_demoted += 1

        # 5. Explicit evict_after_node for the freshly-minted tids.
        evict_after_node: dict[int, list[int]] = defaultdict(list)
        for last_gpu_nid, tids in split_stats.get("new_tids_by_invocation", {}).items():
            if tids:
                evict_after_node[last_gpu_nid].extend(tids)

        trace.args["xfer_arrivals"] = []
        trace.args["d2h_xfer_arrivals"] = []
        trace.args["evict_after_node"] = dict(evict_after_node)

        # 6. Refresh DAV's hint and lifetime state.
        self._arrivals_by_issuer.clear()
        self._gate_by_consumer.clear()
        self._pending_consumers_by_tid.clear()
        self._d2h_arrivals_by_issuer.clear()
        self._build_arrival_index()
        self._xfer_state.clear()
        self._init_xfer_states()
        self._rebuild_lifetime_tracking(trace)

        print(
            f"[{type(self).__name__}] compile: "
            f"memcpy_h2d_nodes={len(memcpy_h2d_dst_tids)} "
            f"leaf_paths={len(leaf_paths)} "
            f"invocations={sum(len(v) for v in leaf_invocations.values())} "
            f"splits={split_stats['splits']} "
            f"new_tids={split_stats['new_tids']} "
            f"rewired_consumers={split_stats['rewired_consumers']} "
            f"demoted_cuda_INPUT_to_CONTEXT={n_demoted} "
            f"evict_after_node_keys={len(evict_after_node)}",
            flush=True,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _dealias_memcpy_h2d(trace: Trace) -> dict[int, int]:
        result: dict[int, int] = {}
        for nid, node in trace.node_map.items():
            name = ""
            if isinstance(node.args, dict):
                name = node.args.get("op_name") or node.args.get("node_name") or ""
            if "Memcpy HtoD" not in name:
                continue
            ins = list(node.input_tensors)
            outs = list(node.output_tensors)
            for tid in set(ins) & set(outs):
                tensor = trace.tensor_map.get(tid)
                if tensor is None:
                    continue
                if not str(tensor.args.get("device", "")).startswith("cuda"):
                    continue
                node.input_tensors = [t for t in ins if t != tid]
                result[nid] = tid
                break
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _split_memcpy_outputs_per_invocation(
        trace: Trace,
        leaf_invocations: dict[str, list[_LeafEpoch]],
        memcpy_h2d_dst_tids: dict[int, int],
    ) -> dict[str, Any]:
        memcpy_node_ids = set(memcpy_h2d_dst_tids.keys())
        next_tid_id = (max(trace.tensor_map.keys()) + 1) if trace.tensor_map else 1
        splits = 0
        new_tids_created = 0
        rewired_consumers = 0
        new_tids_by_invocation: dict[int, list[int]] = defaultdict(list)

        consumers_of_tid: dict[int, list[int]] = defaultdict(list)
        for nid, node in trace.node_map.items():
            for tid in node.input_tensors:
                consumers_of_tid[tid].append(nid)

        for leaf_path, invs in leaf_invocations.items():
            for inv in invs:
                if inv.last_gpu_nid is None:
                    continue
                first_nid = inv.first_node
                last_nid = inv.last_gpu_nid
                inv_memcpys = [
                    nid for nid in memcpy_node_ids
                    if first_nid <= nid <= last_nid
                ]
                if not inv_memcpys:
                    continue
                for memcpy_nid in inv_memcpys:
                    memcpy_node = trace.node_map[memcpy_nid]
                    orig_dst_tid = memcpy_h2d_dst_tids[memcpy_nid]
                    new_tid_id = next_tid_id
                    next_tid_id += 1
                    orig_tensor = trace.tensor_map[orig_dst_tid]
                    new_tensor = copy.copy(orig_tensor)
                    new_tensor.id = new_tid_id
                    new_tensor.args = dict(orig_tensor.args)
                    new_tensor.args["tensor_type"] = "CONTEXT"
                    new_tensor.args.pop("implicit_input", None)
                    trace.tensor_map[new_tid_id] = new_tensor
                    new_tids_created += 1

                    memcpy_node.output_tensors = [
                        new_tid_id if t == orig_dst_tid else t
                        for t in memcpy_node.output_tensors
                    ]

                    for consumer_nid in consumers_of_tid.get(orig_dst_tid, []):
                        if not (first_nid <= consumer_nid <= last_nid):
                            continue
                        cnode = trace.node_map.get(consumer_nid)
                        if cnode is None or cnode is memcpy_node:
                            continue
                        if orig_dst_tid in cnode.input_tensors:
                            cnode.input_tensors = [
                                new_tid_id if t == orig_dst_tid else t
                                for t in cnode.input_tensors
                            ]
                            rewired_consumers += 1
                        if orig_dst_tid in cnode.output_tensors:
                            cnode.output_tensors = [
                                new_tid_id if t == orig_dst_tid else t
                                for t in cnode.output_tensors
                            ]
                    splits += 1
                    new_tids_by_invocation[inv.last_gpu_nid].append(new_tid_id)

        return {
            "splits": splits,
            "new_tids": new_tids_created,
            "rewired_consumers": rewired_consumers,
            "new_tids_by_invocation": dict(new_tids_by_invocation),
        }

    # ------------------------------------------------------------------
    def _rebuild_lifetime_tracking(self, trace: Trace) -> None:
        self._remaining_consumers = {}
        self._remaining_cpu_consumers = {}
        self._consumers_by_tid = {}
        for nid, n in trace.node_map.items():
            ins = set(n.input_tensors)
            outs = set(n.output_tensors)
            for tid in ins:
                self._remaining_consumers[tid] = self._remaining_consumers.get(tid, 0) + 1
            role = n.args.get("runtime_role") if isinstance(n.args, dict) else None
            if role == "cpu_leaf":
                for tid in ins:
                    self._remaining_cpu_consumers[tid] = (
                        self._remaining_cpu_consumers.get(tid, 0) + 1
                    )
            for tid in ins - outs:
                self._consumers_by_tid.setdefault(tid, []).append(nid)

    # ------------------------------------------------------------------
    def _resolve_node_modules(self, trace: Trace) -> dict[int, str]:
        node_map = trace.node_map
        node_module: dict[int, str] = {}

        for nid, node in node_map.items():
            mod = node.args.get("module") if isinstance(node.args, dict) else None
            mp = mod.get("module_path") if isinstance(mod, dict) else None
            if mp:
                node_module[nid] = mp

        gpu_launcher: dict[int, int] = {}
        for parent_id, child_id in trace.args.get("start_gated_edges", []) or []:
            gpu_launcher[int(child_id)] = int(parent_id)
        for nid, node in node_map.items():
            if nid in node_module:
                continue
            role = node.args.get("runtime_role") if isinstance(node.args, dict) else None
            if role != "gpu_runtime":
                continue
            cpu_id = gpu_launcher.get(nid)
            if cpu_id is not None:
                mp = node_module.get(cpu_id)
                if mp:
                    node_module[nid] = mp

        for nid in list(node_map.keys()):
            if nid in node_module:
                continue
            seen: set[int] = set()
            frontier: list[tuple[int, int]] = [(nid, 0)]
            found: str | None = None
            while frontier and found is None:
                nxt: list[tuple[int, int]] = []
                for x, d in frontier:
                    if x in seen or d > _MAX_PARENT_WALK_DEPTH:
                        continue
                    seen.add(x)
                    if x in node_module:
                        found = node_module[x]
                        break
                    xn = node_map.get(x)
                    if xn is None:
                        continue
                    for pid in xn.parent_nodes:
                        nxt.append((int(pid), d + 1))
                frontier = nxt
            if found:
                node_module[nid] = found
        return node_module

    @staticmethod
    def _discover_leaf_paths(
        trace: Trace, node_module: dict[int, str],
    ) -> set[str]:
        leaf_paths: set[str] = set()
        for nid, node in trace.node_map.items():
            name = ""
            if isinstance(node.args, dict):
                name = node.args.get("op_name") or node.args.get("node_name") or ""
            if "cudaMemcpyAsync" not in name:
                continue
            mp = node_module.get(nid)
            if mp:
                leaf_paths.add(mp)
        return leaf_paths

    @staticmethod
    def _enumerate_leaf_invocations(
        trace: Trace,
        node_module: dict[int, str],
        leaf_paths: set[str],
    ) -> dict[str, list[_LeafEpoch]]:
        sorted_leaves = sorted(leaf_paths, key=len, reverse=True)

        order: list[tuple[int, int]] = []
        for nid, node in trace.node_map.items():
            sn = node.args.get("start_ns") if isinstance(node.args, dict) else None
            order.append((sn if sn is not None else 1 << 62, nid))
        order.sort()

        invocations: dict[str, list[_LeafEpoch]] = defaultdict(list)
        cur_leaf: str | None = None
        cur_epoch: _LeafEpoch | None = None

        def _classify(mp: str | None) -> str | None:
            if not mp:
                return None
            for leaf in sorted_leaves:
                if mp == leaf or mp.startswith(leaf + "."):
                    return leaf
            return None

        for _, nid in order:
            node = trace.node_map[nid]
            mp = node_module.get(nid)
            leaf = _classify(mp)
            if leaf != cur_leaf:
                if cur_epoch is not None and cur_leaf is not None:
                    invocations[cur_leaf].append(cur_epoch)
                cur_leaf = leaf
                cur_epoch = _LeafEpoch(nid) if leaf is not None else None
            if cur_epoch is None:
                continue
            role = node.args.get("runtime_role") if isinstance(node.args, dict) else None
            if role == "gpu_runtime":
                if cur_epoch.first_gpu_nid is None:
                    cur_epoch.first_gpu_nid = nid
                cur_epoch.last_gpu_nid = nid

        if cur_epoch is not None and cur_leaf is not None:
            invocations[cur_leaf].append(cur_epoch)
        return invocations
