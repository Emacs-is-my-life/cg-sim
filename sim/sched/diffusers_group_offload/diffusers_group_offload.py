"""Scheduler that simulates diffusers' ``enable_group_offload`` in cg-sim.

Extends ``DeviceAwareVanillaAsync`` with a ``compile`` stage that emits the
same four ``trace.args`` hints the DAV runtime already consumes:

* ``xfer_arrivals``     RAM→VRAM prefetch, gates a named consumer.
* ``d2h_xfer_arrivals`` VRAM→RAM evict via real DtoH transfer.
* ``evict_after_node``  pointer-swap eviction (VRAM region release only).
* ``evictable_tensor_ids`` (not used here — ``evict_after_node`` is
  sufficient and avoids the natural last-consumer release path).

Models ``offload_type="block_level"``, ``num_blocks_per_group=1``,
``use_stream=True``, ``record_stream=False`` — the variant whose runtime
is documented at ``docs/offload-schemes/diffusers_group-offload_use-stream-true.md``.

Direct-child ``ModuleList``/``Sequential`` grandchildren of each top-level
pipeline component become "matched" groups: pinned-source H2D on a side
stream, pointer-swap eviction (no D2H). Everything else under a component
forms the per-component "unmatched" lump: H2D and D2H both on the compute
stream, gated to serialize with kernels.

``block_modules: dict[str, list[str]]`` in ``args`` mirrors diffusers'
``block_modules=`` argument — names additional non-ModuleList direct
children of a component to recurse into (e.g. Flux's ``mid_block``).
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


class _Epoch:
    __slots__ = ("block_path", "first_gpu_nid", "last_gpu_nid")

    def __init__(self, block_path: str, first_gpu_nid: int) -> None:
        self.block_path = block_path
        self.first_gpu_nid = first_gpu_nid
        self.last_gpu_nid = first_gpu_nid


class _Span:
    __slots__ = (
        "component",
        "first_node",
        "first_gpu_nid",
        "last_gpu_nid",
        "matched_epochs",
    )

    def __init__(self, component: str, first_node: int) -> None:
        self.component = component
        self.first_node = first_node
        self.first_gpu_nid: int | None = None
        self.last_gpu_nid: int | None = None
        self.matched_epochs: list[_Epoch] = []


class DiffusersGroupOffload(DeviceAwareVanillaAsync):
    """Scheduler that plans diffusers group_offload as DAV hints at compile time."""

    def __init__(
        self,
        obj_id: int,
        name: str,
        log: "Log",
        sys: "System",
        args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(obj_id, name, log, sys, args)

    # ============================================================ compile
    def compile(self, trace: Trace) -> None:
        hierarchy = trace.args.get("module_hierarchy") or {}
        matched_paths, component_paths = self._classify_hierarchy(hierarchy)
        self._apply_block_modules_override(hierarchy, matched_paths)

        node_module = self._resolve_node_modules(trace)
        tid_module = self._resolve_weight_modules(trace, node_module)

        matched_tids_by_block: dict[str, list[int]] = defaultdict(list)
        unmatched_tids_by_comp: dict[str, list[int]] = defaultdict(list)
        all_offload_tids: set[int] = set()
        # Sort matched_paths longest-first so the prefix match picks the
        # deepest containing block (e.g. "unet.down_blocks.0" wins over
        # "unet" — the latter isn't actually in matched_paths anyway, but
        # the lookup must be deterministic when nesting could appear).
        matched_paths_sorted = sorted(matched_paths, key=len, reverse=True)
        for tid, mp in tid_module.items():
            block = self._longest_prefix(mp, matched_paths_sorted)
            if block:
                matched_tids_by_block[block].append(tid)
                all_offload_tids.add(tid)
                continue
            comp = self._owning_component(mp, component_paths)
            if comp:
                unmatched_tids_by_comp[comp].append(tid)
                all_offload_tids.add(tid)

        # Patch tensor.args["device"] = "cpu" for every offload weight so
        # the layout phase places them in RAM (not VRAM). The compile
        # stage runs before layout, so this is the right time to override.
        for tid in all_offload_tids:
            tensor = trace.tensor_map.get(tid)
            if tensor is not None:
                tensor.args["device"] = "cpu"

        spans = self._enumerate_spans(
            trace, node_module, component_paths, matched_paths_sorted,
        )

        xfer_arrivals: list[dict[str, Any]] = []
        d2h_xfer_arrivals: list[dict[str, Any]] = []
        evict_after_node: dict[int, list[int]] = defaultdict(list)

        for span in spans:
            unmatched_tids = unmatched_tids_by_comp.get(span.component, [])
            first_matched = span.matched_epochs[0] if span.matched_epochs else None
            consumer = (
                first_matched.first_gpu_nid if first_matched else span.first_gpu_nid
            )
            load_tids: list[int] = list(unmatched_tids)
            if first_matched:
                load_tids += matched_tids_by_block.get(first_matched.block_path, [])
            if load_tids and consumer is not None and span.first_node is not None:
                xfer_arrivals.append({
                    "issuer_node_id": span.first_node,
                    "consumer_node_id": consumer,
                    "cgsim_tids": load_tids,
                })

            for prev_ep, nxt_ep in zip(span.matched_epochs, span.matched_epochs[1:]):
                nxt_tids = matched_tids_by_block.get(nxt_ep.block_path, [])
                if nxt_tids:
                    xfer_arrivals.append({
                        "issuer_node_id": prev_ep.first_gpu_nid,
                        "consumer_node_id": nxt_ep.first_gpu_nid,
                        "cgsim_tids": nxt_tids,
                    })

            for ep in span.matched_epochs:
                tids = matched_tids_by_block.get(ep.block_path, [])
                if tids:
                    evict_after_node[ep.last_gpu_nid].extend(tids)

            if unmatched_tids and span.last_gpu_nid is not None:
                d2h_xfer_arrivals.append({
                    "issuer_node_id": span.last_gpu_nid,
                    "cgsim_tids": unmatched_tids,
                })

        trace.args["xfer_arrivals"] = xfer_arrivals
        trace.args["d2h_xfer_arrivals"] = d2h_xfer_arrivals
        trace.args["evict_after_node"] = dict(evict_after_node)

        # DAV's __init__ already populated its arrival indexes from the
        # empty hint set. Reset and re-build now that the hints are written.
        self._arrivals_by_issuer.clear()
        self._gate_by_consumer.clear()
        self._pending_consumers_by_tid.clear()
        self._d2h_arrivals_by_issuer.clear()
        self._build_arrival_index()

        # Re-init xfer_state to reflect the patched devices (every
        # offload weight now starts as ABSENT in VRAM).
        self._xfer_state.clear()
        self._init_xfer_states()

        n_matched_blocks = sum(1 for _ in matched_tids_by_block)
        n_unmatched_lumps = sum(1 for _ in unmatched_tids_by_comp)
        print(
            f"[{type(self).__name__}] compile: "
            f"components={len(component_paths)} "
            f"matched_blocks={n_matched_blocks} "
            f"unmatched_lumps={n_unmatched_lumps} "
            f"offload_tids={len(all_offload_tids)} "
            f"spans={len(spans)} "
            f"xfer_arrivals={len(xfer_arrivals)} "
            f"d2h_xfer_arrivals={len(d2h_xfer_arrivals)} "
            f"evict_after_node={len(evict_after_node)}",
            flush=True,
        )

    # ============================================================ hierarchy
    @staticmethod
    def _classify_hierarchy(
        hierarchy: dict[str, Any],
    ) -> tuple[set[str], list[str]]:
        matched_paths: set[str] = set()
        component_paths: list[str] = []
        for comp_name, comp_info in hierarchy.get("children", {}).items():
            component_paths.append(comp_name)
            for child_name, child_info in comp_info.get("children", {}).items():
                cls = child_info.get("class")
                child_path = f"{comp_name}.{child_name}"
                if cls in ("ModuleList", "Sequential"):
                    for gc_name in child_info.get("children", {}):
                        matched_paths.add(f"{child_path}.{gc_name}")
        return matched_paths, component_paths

    def _apply_block_modules_override(
        self,
        hierarchy: dict[str, Any],
        matched_paths: set[str],
    ) -> None:
        block_modules: dict[str, list[str]] = self.args.get("block_modules") or {}
        if not block_modules:
            return
        for comp_name, forced_children in block_modules.items():
            comp_info = hierarchy.get("children", {}).get(comp_name)
            if comp_info is None:
                continue
            for child_name in forced_children:
                child_info = comp_info.get("children", {}).get(child_name)
                if child_info is None:
                    continue
                child_path = f"{comp_name}.{child_name}"
                for gc_name, gc_info in child_info.get("children", {}).items():
                    if gc_info.get("class") in ("ModuleList", "Sequential"):
                        for gg_name in gc_info.get("children", {}):
                            matched_paths.add(f"{child_path}.{gc_name}.{gg_name}")

    # ============================================================ resolution
    def _resolve_node_modules(self, trace: Trace) -> dict[int, str]:
        node_map = trace.node_map
        node_module: dict[int, str] = {}

        # Pass 1: direct annotations.
        for nid, node in node_map.items():
            mod = node.args.get("module") if isinstance(node.args, dict) else None
            mp = mod.get("module_path") if isinstance(mod, dict) else None
            if mp:
                node_module[nid] = mp

        # Pass 2: gpu_runtime nodes inherit from their CPU launcher via
        # the start-gated `submit` edge (the loader stashes these on
        # trace.args["start_gated_edges"]).
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

        # Pass 3: walk control parents for nodes still missing a module.
        # Bounded depth so the worst-case scan stays linear.
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
    def _resolve_weight_modules(
        trace: Trace, node_module: dict[int, str],
    ) -> dict[int, str]:
        # consumers_by_tid: tid -> list of consumer node_ids in trace order.
        consumers_by_tid: dict[int, list[int]] = defaultdict(list)
        for nid, node in trace.node_map.items():
            for tid in node.input_tensors:
                consumers_by_tid[tid].append(nid)

        tid_module: dict[int, str] = {}
        for tid, tensor in trace.tensor_map.items():
            if tensor.args.get("tensor_type") != "WEIGHT":
                continue
            for cnid in consumers_by_tid.get(tid, []):
                mp = node_module.get(cnid)
                if mp:
                    tid_module[tid] = mp
                    break
        return tid_module

    @staticmethod
    def _longest_prefix(mp: str, sorted_paths: list[str]) -> str | None:
        for p in sorted_paths:
            if mp == p or mp.startswith(p + "."):
                return p
        return None

    @staticmethod
    def _owning_component(mp: str, components: list[str]) -> str | None:
        for c in components:
            if mp == c or mp.startswith(c + "."):
                return c
        return None

    # ============================================================ spans
    def _enumerate_spans(
        self,
        trace: Trace,
        node_module: dict[int, str],
        component_paths: list[str],
        matched_paths_sorted: list[str],
    ) -> list[_Span]:
        components_set = set(component_paths)

        # Iterate nodes in temporal order (start_ns). Nodes without
        # start_ns (e.g. TerminalNode) sort last via a sentinel key.
        sortable: list[tuple[int, int]] = []  # (start_ns_or_huge, node_id)
        for nid, node in trace.node_map.items():
            sn = node.args.get("start_ns") if isinstance(node.args, dict) else None
            sortable.append((sn if sn is not None else 1 << 62, nid))
        sortable.sort()

        spans: list[_Span] = []
        cur_span: _Span | None = None
        cur_epoch: _Epoch | None = None

        for _, nid in sortable:
            node = trace.node_map[nid]
            mp = node_module.get(nid)
            comp: str | None = None
            block: str | None = None
            if mp:
                comp = self._owning_component(mp, component_paths) if mp else None
                if comp is not None:
                    block = self._longest_prefix(mp, matched_paths_sorted)
            role = node.args.get("runtime_role") if isinstance(node.args, dict) else None
            is_gpu = role == "gpu_runtime"

            # Span transition.
            if comp != (cur_span.component if cur_span else None):
                if cur_epoch is not None:
                    assert cur_span is not None
                    cur_span.matched_epochs.append(cur_epoch)
                    cur_epoch = None
                if cur_span is not None:
                    spans.append(cur_span)
                cur_span = _Span(comp, nid) if comp is not None else None

            if cur_span is None:
                continue

            if is_gpu:
                if cur_span.first_gpu_nid is None:
                    cur_span.first_gpu_nid = nid
                cur_span.last_gpu_nid = nid

            # Epoch transition (matched block).
            if block != (cur_epoch.block_path if cur_epoch else None):
                if cur_epoch is not None:
                    cur_span.matched_epochs.append(cur_epoch)
                    cur_epoch = None
                if block is not None and is_gpu:
                    cur_epoch = _Epoch(block, nid)
            else:
                if cur_epoch is not None and is_gpu:
                    cur_epoch.last_gpu_nid = nid

        # Close any open span / epoch at end.
        if cur_epoch is not None and cur_span is not None:
            cur_span.matched_epochs.append(cur_epoch)
        if cur_span is not None:
            spans.append(cur_span)

        # Apply block_modules override before returning. (It needs to be
        # applied to matched_paths, but we don't take matched_paths here —
        # the caller already passed the overridden set in matched_paths_sorted.)
        del components_set  # not used after iteration
        return spans
