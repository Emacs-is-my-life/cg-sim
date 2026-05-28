"""Scheduler that simulates ``accelerate.cpu_offload(..., offload_buffers=False)``
in cg-sim.

Mirrors the ``DiffusersGroupOffload`` pattern: input is the **eager**
trace (everything resident on cuda from the start, no per-leaf
Memcpy events in the trace), and ``compile()`` injects per-leaf paging
hints via the four DAV ``trace.args`` channels.

What accelerate cpu_offload models
----------------------------------
Per
``docs/offload-schemes/accelerate_cpu-offload_buffers-false.md``:

* Every module that owns direct params (Linear / Embedding /
  LlamaRMSNorm in Llama-3) gets an ``AlignDevicesHook``. Its
  ``pre_forward`` does a synchronous ``cudaMemcpyAsync(Pageable →
  Device)`` of every param to the execution device; its
  ``post_forward`` rebinds the param to ``meta`` (no DtoH bytes).
* Buffers (persistent and non-persistent) stay resident on the
  execution device for the lifetime of the model.
* The H2D is on the compute stream with a hard
  ``cudaStreamSynchronize`` after each copy — no prefetch overlap,
  end-to-end serialization per leaf.

What this compile stage does
----------------------------
1. **Discover paged leaves** — every cuda WEIGHT tid in the eager
   trace is a parameter. Resolve its first consumer's
   ``module_path``; that path is the leaf module (Linear,
   Embedding, LlamaRMSNorm, …). Group WEIGHT tids by leaf.
2. **Patch each paged WEIGHT tid's device to "cpu"** — DAV's layout
   phase 0 places ``device == "cpu"`` tensors in RAM (the
   ``weights_map`` master copy in accelerate, populated at
   ``big_modeling.py:209``). Buffers (non-WEIGHT cuda tensors or
   WEIGHT tensors not under any leaf) stay resident.
3. **Enumerate per-leaf invocations.** Walk the trace in temporal
   order; a leaf's "epoch" is a contiguous run of nodes whose
   resolved ``module_path`` equals (or starts with) the leaf path.
   Llama's eager trace has 15 generated tokens → 15 invocations
   per leaf.
4. **Emit ``xfer_arrival`` + ``evict_after_node`` per invocation.**
   Issuer = first node of the epoch (typically a CPU op at the
   start of the leaf's forward). Consumer = first GPU node of the
   epoch (gated on the H2D landing). Evict at the last GPU node
   (releases the VRAM mirror; RAM master copy preserved by
   ``_release_vram_only``). No ``d2h_xfer_arrivals`` — accelerate's
   post_forward drops to meta (descriptor swap, no D2H bytes).
5. **Refresh DAV's arrival / xfer-state indexes.** DAV.__init__
   already ran against an empty hint set.

Reference: ``docs/offload-schemes/accelerate_cpu-offload_buffers-false.md``.
"""

from __future__ import annotations

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
    """Plans accelerate-cpu_offload paging as DAV hints at compile time."""

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
        # 1. Resolve module_path for every node (gpu_runtime nodes
        #    inherit from their CPU launcher via start-gated submit edges).
        node_module = self._resolve_node_modules(trace)

        # 2. WEIGHT tids on cuda → group by owning leaf module_path.
        leaf_tids = self._resolve_paged_leaves(trace, node_module)

        # 3. Patch each paged WEIGHT's device to "cpu" so layout puts
        #    it in RAM. DAV's runtime then issues RAM→VRAM transfers
        #    via the xfer_arrival hints we emit below; evict_after_node
        #    uses _release_vram_only so the RAM master copy survives.
        all_offload_tids: set[int] = set()
        for tids in leaf_tids.values():
            all_offload_tids.update(tids)
        for tid in all_offload_tids:
            tensor = trace.tensor_map.get(tid)
            if tensor is not None:
                tensor.args["device"] = "cpu"

        # 4. Enumerate per-leaf invocations.
        leaf_invocations = self._enumerate_leaf_invocations(
            trace, node_module, set(leaf_tids.keys()),
        )

        # 5. Emit hints. Chain ALL invocations (across every leaf and
        #    every token) in temporal order, so each prefetch's issuer
        #    is the *previous* invocation's last_gpu_nid — i.e. the
        #    previous leaf's eviction key. This serializes per-leaf
        #    paging exactly the way accelerate's synchronous
        #    cudaStreamSynchronize does in real life: only one leaf's
        #    weights are in VRAM at a time.
        #
        #    Without this chain, the trace's weak inter-leaf control
        #    graph lets multiple leaves' CPU preambles retire in
        #    parallel in sim-time, so multiple prefetches queue up
        #    concurrently and either (a) inflate VRAM peak (multiple
        #    leaves resident simultaneously) or (b) collapse into
        #    no-ops because the relevant tid is already LOADING from a
        #    previous arrival. The chain serializes arrivals into the
        #    intended one-leaf-at-a-time pattern.
        ordered_invocations: list[tuple[int, str, _LeafEpoch]] = []
        for leaf_path, invocations in leaf_invocations.items():
            for inv in invocations:
                if inv.first_gpu_nid is None or inv.last_gpu_nid is None:
                    continue
                ordered_invocations.append((inv.first_gpu_nid, leaf_path, inv))
        ordered_invocations.sort(key=lambda x: x[0])

        xfer_arrivals: list[dict[str, Any]] = []
        evict_after_node: dict[int, list[int]] = defaultdict(list)
        prev_last_gpu_nid: int | None = None
        for _, leaf_path, inv in ordered_invocations:
            tids = leaf_tids[leaf_path]
            if not tids:
                continue
            issuer = (
                prev_last_gpu_nid
                if prev_last_gpu_nid is not None
                else inv.first_node
            )
            xfer_arrivals.append({
                "issuer_node_id": issuer,
                "consumer_node_id": inv.first_gpu_nid,
                "cgsim_tids": tids,
            })
            evict_after_node[inv.last_gpu_nid].extend(tids)
            prev_last_gpu_nid = inv.last_gpu_nid

        trace.args["xfer_arrivals"] = xfer_arrivals
        trace.args["d2h_xfer_arrivals"] = []
        trace.args["evict_after_node"] = dict(evict_after_node)

        # 6. Refresh DAV's hint and xfer-state indexes.
        self._arrivals_by_issuer.clear()
        self._gate_by_consumer.clear()
        self._pending_consumers_by_tid.clear()
        self._d2h_arrivals_by_issuer.clear()
        self._build_arrival_index()
        self._xfer_state.clear()
        self._init_xfer_states()

        total_invocations = sum(len(v) for v in leaf_invocations.values())
        print(
            f"[{type(self).__name__}] compile: "
            f"paged_leaves={len(leaf_tids)} "
            f"paged_tids={len(all_offload_tids)} "
            f"invocations={total_invocations} "
            f"xfer_arrivals={len(xfer_arrivals)} "
            f"evict_after_node_keys={len(evict_after_node)}",
            flush=True,
        )

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
    def _resolve_paged_leaves(
        trace: Trace, node_module: dict[int, str],
    ) -> dict[str, list[int]]:
        """WEIGHT tid → owning module_path(s), grouped by leaf.

        Each tid is registered with EVERY leaf module whose forward
        consumes it. Tied weights (e.g. Llama's
        ``model.embed_tokens.weight`` ↔ ``lm_head.weight``, same
        storage, two ``nn.Parameter`` aliases) appear once in the
        eager trace's WEIGHT tensors but accelerate's
        ``AlignDevicesHook.pre_forward`` issues a separate
        ``cudaMemcpyAsync`` for each owning module — once when
        ``embed_tokens`` runs at the top of the model and again when
        ``lm_head`` runs at the bottom. Assigning to all consumers
        reproduces the two-transfer behavior; the eviction after
        ``embed_tokens.last_gpu_nid`` then releases the VRAM region
        so ``lm_head``'s subsequent prefetch is a real transfer.
        """
        consumers_by_tid: dict[int, list[int]] = defaultdict(list)
        for nid, node in trace.node_map.items():
            for tid in node.input_tensors:
                consumers_by_tid[tid].append(nid)

        leaf_tids: dict[str, list[int]] = defaultdict(list)
        for tid, tensor in trace.tensor_map.items():
            if tensor.args.get("tensor_type") != "WEIGHT":
                continue
            device = str(tensor.args.get("device", "")).lower()
            if not device.startswith("cuda"):
                continue
            seen_mps: set[str] = set()
            for cnid in sorted(consumers_by_tid.get(tid, [])):
                mp = node_module.get(cnid)
                if mp and mp not in seen_mps:
                    seen_mps.add(mp)
                    leaf_tids[mp].append(tid)
        return leaf_tids

    @staticmethod
    def _enumerate_leaf_invocations(
        trace: Trace,
        node_module: dict[int, str],
        leaf_paths: set[str],
    ) -> dict[str, list[_LeafEpoch]]:
        """Enumerate per-leaf invocations.

        Epochs are bracketed by the leaf's CPU activity (sorted by
        start_ns). The CPU thread runs each module's forward
        sequentially, so the CPU-side trace gives an unambiguous
        per-token leaf bracketing. The GPU kernels each CPU forward
        launches run asynchronously and may not be contiguous with
        their CPU preamble in start_ns order (other modules' CPU
        activity can land in between) — so we don't use them to
        bracket the epoch. Instead, for each closed CPU epoch we
        collect the gpu_runtime children launched by any submit node
        in the epoch (via ``trace.args["start_gated_edges"]``) and
        take the temporal min/max for first_gpu_nid / last_gpu_nid.
        """
        sorted_leaves = sorted(leaf_paths, key=len, reverse=True)

        # Map: cpu_launcher_nid -> list of launched gpu_runtime nids.
        gpu_children_by_launcher: dict[int, list[int]] = defaultdict(list)
        for parent_id, child_id in trace.args.get("start_gated_edges", []) or []:
            gpu_children_by_launcher[int(parent_id)].append(int(child_id))

        def _classify(mp: str | None) -> str | None:
            if not mp:
                return None
            for leaf in sorted_leaves:
                if mp == leaf or mp.startswith(leaf + "."):
                    return leaf
            return None

        # Iterate CPU nodes (cpu_leaf + submit) by start_ns. gpu_runtime
        # nodes are not part of the bracketing — their epoch membership
        # is derived from their CPU launcher.
        cpu_order: list[tuple[int, int]] = []
        for nid, node in trace.node_map.items():
            if not isinstance(node.args, dict):
                continue
            role = node.args.get("runtime_role")
            if role not in ("cpu_leaf", "submit"):
                continue
            sn = node.args.get("start_ns")
            cpu_order.append((sn if sn is not None else 1 << 62, nid))
        cpu_order.sort()

        invocations: dict[str, list[_LeafEpoch]] = defaultdict(list)

        def _close_epoch(leaf: str, first_nid: int, launchers: list[int]) -> None:
            if not launchers:
                # No GPU activity in this CPU epoch — nothing to gate
                # an arrival on. Skip.
                return
            gpu_nids: list[int] = []
            for L in launchers:
                gpu_nids.extend(gpu_children_by_launcher.get(L, []))
            if not gpu_nids:
                return
            node_map = trace.node_map
            def _sn(x: int) -> int:
                n = node_map.get(x)
                if n is None or not isinstance(n.args, dict):
                    return 1 << 62
                return n.args.get("start_ns") or (1 << 62)
            gpu_nids.sort(key=_sn)
            ep = _LeafEpoch(first_nid)
            ep.first_gpu_nid = gpu_nids[0]
            ep.last_gpu_nid = gpu_nids[-1]
            invocations[leaf].append(ep)

        cur_leaf: str | None = None
        cur_first: int | None = None
        cur_launchers: list[int] = []
        for _, nid in cpu_order:
            node = trace.node_map[nid]
            mp = node_module.get(nid)
            leaf = _classify(mp)
            if leaf != cur_leaf:
                if cur_leaf is not None and cur_first is not None:
                    _close_epoch(cur_leaf, cur_first, cur_launchers)
                cur_leaf = leaf
                cur_first = nid if leaf is not None else None
                cur_launchers = []
            if cur_leaf is None:
                continue
            role = node.args.get("runtime_role")
            if role == "submit":
                cur_launchers.append(nid)

        if cur_leaf is not None and cur_first is not None:
            _close_epoch(cur_leaf, cur_first, cur_launchers)
        return invocations
