"""Async-prefetch variant of DeviceAwareVanilla.

Reads ``trace.args["xfer_arrivals"]`` populated by
``graph_modifiers.inject_schedule`` and fires RAM->VRAM transfers in
the BACKGROUND when each issuer node retires, gating the consumer
node's compute submission on those transfers' completion.

Mechanics mirror ``sim.sched.flexinfer``:

  - Each prefetch tensor carries a per-tensor ``xfer_state`` flag
    (RESIDENT / ABSENT / LOADING / LOADED) on the scheduler.
  - At issuer-retire, ``sys.transfer`` is invoked for the planned
    RAM->VRAM moves; the dst VRAM region is claimed at that moment, so
    peak VRAM accounting tracks "VRAM occupied during transfer" — the
    real PyTorch behavior. Tensor flips ABSENT -> LOADING.
  - ``sys.transfer`` runs the byte movement in parallel with GPU
    compute on the existing memory subsystem (PCIe bandwidth enforced
    by SimpleRAM/SimpleVRAM bandwidths). No new resource needed.
  - On TransferJob retire, tensor flips LOADING -> LOADED.
  - The base scheduler's ``_submit_ready_nodes`` is wrapped: a node
    that has any required tensor with state != LOADED/RESIDENT is
    re-queued, retried next tick.
  - Eviction is unchanged from the base — driven by
    ``trace.args["evict_after_node"]`` populated by the injector.

This keeps the simulator's wall-clock asymmetry between async and
sync prefetch realistic without inventing a phantom copy resource.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, TYPE_CHECKING

from sim.core.job import BaseJob, ComputeJob, TransferJob
from sim.core.log import Log
from sim.core.trace import NodeStatus
from sim.hw.common import DataRegion, DataRegionAccess
from sim.sched.device_aware_vanilla import DeviceAwareVanilla

if TYPE_CHECKING:
    from sim.core.system import System


# Tensor xfer states.
_RESIDENT = "RESIDENT"  # placed in VRAM at layout, never moved
_ABSENT = "ABSENT"      # lives in RAM only (or already evicted)
_LOADING = "LOADING"    # transfer in flight to VRAM
_LOADED = "LOADED"      # transfer finished, region resident


class DeviceAwareVanillaAsync(DeviceAwareVanilla):
    """DAV + flexinfer-style background prefetch driven by injector."""

    def __init__(
        self,
        obj_id: int,
        name: str,
        log: Log,
        sys: System,
        args: dict[str, Any] | None = None,
    ):
        super().__init__(obj_id, name, log, sys, args)

        # Per cgsim-tensor-id residency state.
        self._xfer_state: dict[int, str] = {}

        # TransferJob.id -> [cgsim_tid, ...] for retire callback.
        self._inflight_jobs: dict[Any, list[int]] = {}
        self._active_prefetch_jobs = 0

        # Optional H2D FIFO width. 0 means legacy/unlimited behavior.
        stream_arg = (args or {}).get(
            "h2d_streams",
            self.sys.trace.args.get("xfer_h2d_streams", 0),
        )
        try:
            self._h2d_streams = max(0, int(stream_arg))
        except (TypeError, ValueError):
            self._h2d_streams = 0
        self._prefetch_queue: deque[list[int]] = deque()

        # issuer_node_id -> list of arrival dicts (fire on retire).
        self._arrivals_by_issuer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        # consumer_node_id -> set of cgsim_tids that must be LOADED before dispatch.
        self._gate_by_consumer: dict[int, set[int]] = defaultdict(set)

        self._build_arrival_index()
        self._init_xfer_states()

    # ------------------------------------------------------------------ init

    def _build_arrival_index(self) -> None:
        arrivals = self.sys.trace.args.get("xfer_arrivals") or []
        for a in arrivals:
            issuer = int(a["issuer_node_id"])
            consumer = int(a["consumer_node_id"])
            tids = [int(t) for t in a["cgsim_tids"]]
            self._arrivals_by_issuer[issuer].append({
                "consumer_node_id": consumer,
                "cgsim_tids": tids,
            })
            self._gate_by_consumer[consumer].update(tids)

    def _init_xfer_states(self) -> None:
        for tid, tensor in self.sys.trace.tensor_map.items():
            ttype = tensor.args.get("tensor_type")
            if ttype not in ("WEIGHT", "LEAF"):
                continue
            device = str(tensor.args.get("device", "")).lower()
            if device.startswith("cuda"):
                self._xfer_state[tid] = _RESIDENT
            else:
                self._xfer_state[tid] = _ABSENT

    # -------------------------------------------------------------- runtime

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        # Phase 1: walk retired jobs.
        # - TransferJob retire flips LOADING -> LOADED.
        # - ComputeJob retire fires any prefetches scheduled on this node
        #   AFTER the standard _retire_completed_nodes runs (which honors
        #   evict_after_node, releasing VRAM regions before we try to
        #   claim them for prefetch destinations).
        self._handle_transfer_retires(retired_jobs)
        self._retire_completed_nodes(retired_jobs)
        self._fire_prefetches_for_retired(retired_jobs)
        self._drain_prefetch_queue()
        self._retry_pending_releases()

        while self._submit_ready_nodes():
            pass

        if not self.sys.engine.job_running and not self.sys.engine.job_waiting:
            todo = [
                n for n in self.sys.trace.node_map.values()
                if n.status == NodeStatus.TODO
            ]
            if todo:
                blocked = todo[0]
                gate = self._gate_by_consumer.get(blocked.id, set())
                gate_status = [
                    {"tid": t, "state": self._xfer_state.get(t, "?")}
                    for t in gate
                ]
                self.sys.abort({
                    "from": self.name,
                    "error": "SCHEDULER_DEADLOCK",
                    "msg": "No runnable node is available.",
                    "node": {
                        "id": blocked.id,
                        "name": blocked.name,
                        "parent_nodes": blocked.parent_nodes,
                        "input_tensors": blocked.input_tensors,
                        "output_tensors": blocked.output_tensors,
                        "xfer_gate": gate_status,
                    },
                })

    def _handle_transfer_retires(self, retired_jobs: list[BaseJob]) -> None:
        for job in retired_jobs:
            if not isinstance(job, TransferJob):
                continue
            tids = self._inflight_jobs.pop(job.id, None)
            if not tids:
                continue
            self._active_prefetch_jobs = max(0, self._active_prefetch_jobs - 1)
            for tid in tids:
                if self._xfer_state.get(tid) == _LOADING:
                    self._xfer_state[tid] = _LOADED

    def _fire_prefetches_for_retired(
        self, retired_jobs: list[BaseJob]
    ) -> None:
        for job in retired_jobs:
            if not isinstance(job, ComputeJob):
                continue
            arrivals = self._arrivals_by_issuer.pop(job.node.id, None)
            if not arrivals:
                continue
            for a in arrivals:
                if self._h2d_streams > 0:
                    self._prefetch_queue.append(a["cgsim_tids"])
                else:
                    self._issue_prefetch(a["cgsim_tids"])

    # ------------------------------------------------------------- prefetch

    def _drain_prefetch_queue(self) -> None:
        if self._h2d_streams <= 0:
            return
        while (
            self._prefetch_queue
            and self._active_prefetch_jobs < self._h2d_streams
        ):
            cgsim_tids = self._prefetch_queue.popleft()
            if self._issue_prefetch(cgsim_tids):
                self._active_prefetch_jobs += 1

    def _issue_prefetch(self, cgsim_tids: list[int]) -> bool:
        """Fire async RAM->VRAM transfer for the given tensors."""
        batch: list[tuple[DataRegion, DataRegion]] = []
        loaded_tids: list[int] = []
        for tid in cgsim_tids:
            st = self._xfer_state.get(tid, _ABSENT)
            if st in (_RESIDENT, _LOADED, _LOADING):
                continue
            tensor = self.sys.trace.tensor_map.get(tid)
            if tensor is None:
                continue
            src_regions = self._ram.space.get_by_tensor_id(tid)
            if not src_regions:
                continue
            src = src_regions[0]
            dst = self._claim_region(self._vram, tensor)
            if dst is None:
                # VRAM full — don't change state; consumer will re-gate
                # next tick. The base scheduler may retire other nodes
                # that free VRAM in the meantime.
                continue
            batch.append((src, dst))
            self._xfer_state[tid] = _LOADING
            loaded_tids.append(tid)
        if not batch:
            return False
        job_id = self.sys.transfer(batch)
        self._inflight_jobs[job_id] = loaded_tids
        return True

    # ----------------------------------------------------------- node gate

    def _node_xfer_gate_satisfied(self, node_id: int) -> bool:
        needed = self._gate_by_consumer.get(node_id)
        if not needed:
            return True
        for tid in needed:
            st = self._xfer_state.get(tid, _ABSENT)
            if st not in (_RESIDENT, _LOADED):
                return False
        return True

    def _submit_ready_nodes(self) -> bool:
        """Wrap base submit loop with the prefetch gate.

        We pre-filter ``ready_node_ids``: any node whose gate is
        unsatisfied is moved to a parked queue and re-injected at the
        end of the tick. The base submit loop runs over the remaining
        gate-satisfied nodes.
        """
        if not self._gate_by_consumer:
            return super()._submit_ready_nodes()

        parked: deque[int] = deque()
        passable: deque[int] = deque()
        while self.ready_node_ids:
            nid = self.ready_node_ids.popleft()
            if self._node_xfer_gate_satisfied(nid):
                passable.append(nid)
            else:
                parked.append(nid)

        self.ready_node_ids = passable
        any_submitted = super()._submit_ready_nodes()

        # Re-merge: parked nodes go back; whatever the parent left in
        # ready_node_ids stays ahead so it's tried first next tick.
        parked.extend(self.ready_node_ids)
        self.ready_node_ids = parked
        return any_submitted

    # ----------------------------------------------------------------- evict

    # Eviction is handled by the base scheduler's _retire_completed_nodes
    # via trace.args["evict_after_node"]. Tensor xfer_state flips back to
    # ABSENT when the VRAM region is released — the base path doesn't know
    # about xfer_state, so we hook _release_tensor_regions to mirror it.

    def _release_tensor_regions(self, tensor_id: int) -> None:
        super()._release_tensor_regions(tensor_id)
        # If we still have NO vram region for this tensor, mark ABSENT
        # so the next prefetch can re-LOAD it.
        if not self._vram.space.get_by_tensor_id(tensor_id):
            if self._xfer_state.get(tensor_id) in (_LOADED, _RESIDENT):
                self._xfer_state[tensor_id] = _ABSENT
