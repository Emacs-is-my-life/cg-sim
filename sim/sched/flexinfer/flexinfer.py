from __future__ import annotations

import uuid
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

from sim.sched.common import BaseScheduler
from sim.core.log import Log, TrackID
from sim.core.trace import Trace, Tensor, NodeStatus, Node
from sim.core.job import BaseJob, ComputeJob, ClaimJob, ReleaseJob, TransferJob
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory, MemoryRegion
from sim.hw.storage.common import BaseStorage, StorageRegion

if TYPE_CHECKING:
    from sim.core.system import System

from .utils import categorize_tensors, categorize_nodes


class FlexInferMode(Enum):
    MEMORY_SUFFICIENT = auto()
    MEMORY_INTERMEDIATE_2 = auto()
    MEMORY_INTERMEDIATE_1 = auto()
    MEMROY_LIMITED = auto()


class FlexInferStatus(Enum):
    MEMORY_LOADED = auto()
    MEMORY_LOADING = auto()
    MEMORY_ABSENT = auto()


class FlexInfer(BaseScheduler):
    """
    Implementation of, FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference
    Published in EuroMLSys '25
    """

    def __init__(self, obj_id: int, name: str, log: Log, sys: System, args: dict[str, Any] | None = None):
        super().__init__(obj_id, name, log, sys, args)
        self.prefetch_window = 3  # 3 Layers by default, in paper
        if "prefetch_window" in args:
            self.prefetch_window = int(args["prefetch_window"])
            if self.prefetch_window < 1:
                raise Exception(f"[FlexInfer] Prefetch window cannot be: {self.prefetch_window}")

        """
        Assumption: Expect one of each hardware:

        - One compute unit
        - One memory unit
        - One storage unit
        """
        for hw in sys.hw.values():
            if isinstance(hw, BaseCompute):
                self.compute = hw
            elif isinstance(hw, BaseMemory):
                self.memory = hw
            elif isinstance(hw, BaseStorage):
                self.storage = hw

        # Tensor information gather
        tensor_map = sys.trace.tensor_map
        self.flex_layers, self.other_tensors = categorize_tensors(tensor_map)
        attn_big_num_pages = self.flex_layers[0].attn_big[0].num_pages
        attn_small_num_pages = self.flex_layers[0].attn_small[0].num_pages
        ffn_num_pages = self.flex_layers[0].ffn[0].num_pages
        num_layers = len(self.flex_layers)

        # Get total size of "other tensors"
        others_size_num_pages = 0
        for tensor in self.other_tensors:
            others_size_num_pages += tensor.num_pages

        # Check operation mode
        memory_size = self.memory.space.num_total_pages

        # Pessimistic dyn-heap reserve: prefetch_window layers' worth of every
        # flex-layer tensor, matching the per-layer slot size used at runtime.
        reserve_pessimistic = self.prefetch_window * (
            3 * ffn_num_pages + 2 * attn_big_num_pages + 2 * attn_small_num_pages
        )

        cost_sufficient = (
            others_size_num_pages
            + num_layers * (3 * ffn_num_pages + attn_big_num_pages)
            + reserve_pessimistic
        )
        cost_intermediate_2 = (
            others_size_num_pages
            + num_layers * (2 * ffn_num_pages)
            + reserve_pessimistic
        )
        cost_intermediate_1 = (
            others_size_num_pages
            + num_layers * (1 * ffn_num_pages)
            + reserve_pessimistic
        )
        minimum_size = others_size_num_pages + reserve_pessimistic

        if memory_size < minimum_size:
            print("[FlexInfer] Failed due to memory shortage.")
            args = {
                "from": self.name,
                "msg": f"Memory size is too small to run FlexInfer policy. {4 * minimum_size} KB required, have {4 * memory_size} KB."
            }
            sys.abort(args)

        args = {"from": self.name, "msg": ""}
        if memory_size >= cost_sufficient:
            self.mode = FlexInferMode.MEMORY_SUFFICIENT
            args["msg"] = "FlexInfer Scheduler operating in MEMORY_SUFFICIENT mode."
        elif memory_size >= cost_intermediate_2:
            self.mode = FlexInferMode.MEMORY_INTERMEDIATE_2
            args["msg"] = "FlexInfer Scheduler operating in MEMORY_INTERMEDIATE_2 mode."
        elif memory_size >= cost_intermediate_1:
            self.mode = FlexInferMode.MEMORY_INTERMEDIATE_1
            args["msg"] = "FlexInfer Scheduler operating in MEMORY_INTERMEDIATE_1 mode."
        else:
            self.mode = FlexInferMode.MEMROY_LIMITED
            args["msg"] = "FlexInfer Scheduler operating in MEMORY_LIMITED mode."

        self.log.record(Log.engine(self.id, "SCHEDULER_MESSAGE", 0, args))

        self.attn_big_num_pages = attn_big_num_pages
        self.attn_small_num_pages = attn_small_num_pages
        self.ffn_num_pages = ffn_num_pages
        self.free_region_page_idx = 0   # Keep track of available memory pages
        return

    def compile(self, trace: Trace) -> None:
        """No compilation"""
        return

    def _build_payload(self, init_storage: BaseStorage, tensor: Tensor) -> tuple[MemoryRegion, StorageRegion]:
        mem_region = self.sys.claim(self.memory, tensor, self.free_region_page_idx)
        self.free_region_page_idx += tensor.num_pages
        stor_region = self.sys.find(init_storage, tensor)[0]
        return (stor_region, mem_region)

    def _assign_in_mem(self, tensor: Tensor) -> None:
        self.sys.claim(self.memory, tensor, self.free_region_page_idx)
        self.free_region_page_idx += tensor.num_pages
        return

    def _acquire_slot(self, step_idx: int, layer_idx: int) -> int | None:
        """Claim the first free dynamic-heap slot for this (step, layer)."""
        for i in range(self.num_slots):
            if self.dyn_slots[i] is None:
                self.dyn_slots[i] = (step_idx, layer_idx)
                return i
        return None

    def _release_slot(self, step_idx: int, layer_idx: int) -> None:
        """Release whichever slot this (step, layer) occupies (no-op if none)."""
        for i in range(self.num_slots):
            if self.dyn_slots[i] == (step_idx, layer_idx):
                self.dyn_slots[i] = None
                return

    def _tensor_page_idx(self, layer_idx: int, slot_idx: int, tensor: Tensor) -> int:
        """Absolute memory page index for tensor when layer sits in slot.
        Placement is a function of layer shape only, so step doesn't enter here."""
        slot_base = self.dyn_start_page + slot_idx * self.slot_size_pages
        return slot_base + self.layer_tensor_offsets[layer_idx][tensor.id]

    def _unload_layer(self, step_idx: int, layer_idx: int) -> None:
        """Release a (step, layer)'s unpinned tensors and free its slot."""
        for tensor in self.unpinned_per_layer[layer_idx]:
            regions = self.sys.find(self.memory, tensor)
            if not regions:
                continue
            self.sys.release(regions[0])
            tensor.args["flexinfer"] = FlexInferStatus.MEMORY_ABSENT
        self._release_slot(step_idx, layer_idx)

    def _advance_prefetch_ptr(self) -> None:
        """Move next_prefetch pointer forward, wrapping layer → step+1."""
        self.next_prefetch_layer_idx += 1
        if self.next_prefetch_layer_idx >= len(self.flex_layers):
            self.next_prefetch_step_idx += 1
            self.next_prefetch_layer_idx = 0

    def layout(self, init_storage: BaseStorage) -> bool:
        """  FlexInfer Algorithm
        Input: Attention tensor size 𝑠𝑖𝑧𝑒𝑎𝑡𝑡𝑒 , FFN tensor size
        𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 , Layer number 𝑁 , Memory budget 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 ,
        Output: Tensor preservation plan 𝑃
        1: if 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 >= 𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 ∗ 𝑁 ∗ 3 + 𝑠𝑖𝑧𝑒𝑎𝑡𝑡𝑛 ∗ 𝑁 ∗ 1 then
        2:     Set all FFN tensors for all layers  (MEMORY_SUFFICIENT)
        3: else
        4:     if 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 >= 𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 ∗ 𝑁 ∗ 2 then
        5:         Set two FFN tensor for all layers (MEMORY_INTERMEDIATE2)
        6:     else
        7:         if 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 >= 𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 ∗ 𝑁 then
        8:             Set one FFN tensor for all layers (MEMORY_INTERMEDIATE1)
        9:         end if
        10:    end if
        11: end if
        12: Set as much as possible attention tensors one by one (MEMORY_LIMITED)
        13: return 𝑃
        """

        """
        How to load a tensor(type: Tensor) into a memory in layout state:

        Iterate over a tensor_map.

        tensor_map = self.sys.trace.tensor_map
        for tensor in tensor_map.values:

        - 1. Claim a memory region for a tensor in the memory
             (page_idx) is index of the start page in memory, which you must keep track of.
            mem_region = self.sys.claim(self.memory, tensor, page_idx)

        - 2. Find this tensor in init_storage (if not INTERMEDIATE tensor)
            stor_region = self.sys.find(init_storage, tensor)[0]

        - 3. Add this stor_region -> mem_region transfer, to a batch we will request
            load_batch.append((mem_region, stor_region))
        """
        load_batch = []


        # [CLAUDE_BEGIN]

        # Pin "other tensors"
        for tensor in self.other_tensors:
            tensor.args["pin"] = True                  # Mark as pinned tensor
            if tensor.args["tensor_type"] == "INTERMEDIATE":
                # Only assign memory region for it
                # because value for INTERMEDIATE tensors does not exist in init_storage
                self._assign_in_mem(tensor)
            else:
                # Load tensor from the storage, to the memory
                payload = self._build_payload(init_storage, tensor)
                load_batch.append(payload)
            tensor.args["flexinfer"] = FlexInferStatus.MEMORY_LOADED   # Mark as loaded

        # Default every flex-layer tensor to unpinned / absent.
        # The blocks below flip the ones chosen for pinning.
        for layer in self.flex_layers:
            for tensor in layer.attn_big + layer.attn_small + layer.ffn:
                tensor.args["pin"] = False
                tensor.args["flexinfer"] = FlexInferStatus.MEMORY_ABSENT

        # Pin FFN tensors per layer according to the selected mode.
        # FlexInfer paper: 3 / 2 / 1 / 0 FFN tensors per layer.
        ffn_pin_count_per_layer = {
            FlexInferMode.MEMORY_SUFFICIENT: 3,
            FlexInferMode.MEMORY_INTERMEDIATE_2: 2,
            FlexInferMode.MEMORY_INTERMEDIATE_1: 1,
            FlexInferMode.MEMROY_LIMITED: 0,
        }[self.mode]

        for layer in self.flex_layers:
            for i in range(min(ffn_pin_count_per_layer, len(layer.ffn))):
                ffn_tensor = layer.ffn[i]
                ffn_tensor.args["pin"] = True
                payload = self._build_payload(init_storage, ffn_tensor)
                load_batch.append(payload)
                ffn_tensor.args["flexinfer"] = FlexInferStatus.MEMORY_LOADED

        # Round-robin pin attn_big tensors across layers, stopping before we
        # eat into the dynamic-heap reserve. Reserve is a pessimistic
        # prefetch_window layers' worth of every flex-layer tensor, per user.
        per_layer_flex_pages = (
            2 * self.attn_big_num_pages
            + 2 * self.attn_small_num_pages
            + 3 * self.ffn_num_pages
        )
        reserve_pages = self.prefetch_window * per_layer_flex_pages
        num_total_pages = self.memory.space.num_total_pages

        done = False
        for attn_big_idx in range(2):  # attn_q first across layers, then attn_output
            if done:
                break
            for layer in self.flex_layers:
                if attn_big_idx >= len(layer.attn_big):
                    continue
                tensor = layer.attn_big[attn_big_idx]
                pages_free_after_pin = (
                    num_total_pages - self.free_region_page_idx - tensor.num_pages
                )
                if pages_free_after_pin < reserve_pages:
                    done = True
                    break
                tensor.args["pin"] = True
                payload = self._build_payload(init_storage, tensor)
                load_batch.append(payload)
                tensor.args["flexinfer"] = FlexInferStatus.MEMORY_LOADED

        # Build the runtime convenience index: per-layer list of tensors
        # that will be load/unload-cycled at runtime.
        self.unpinned_per_layer: list[list[Tensor]] = []
        for layer in self.flex_layers:
            unpinned = [
                t for t in (layer.attn_big + layer.attn_small + layer.ffn)
                if not t.args["pin"]
            ]
            self.unpinned_per_layer.append(unpinned)

        # Dynamic heap: divided into prefetch_window equal slots, each sized
        # to hold one layer's worst-case unpinned tensors. The minimum_size
        # check in __init__ guarantees the dynamic heap fits every slot.
        self.dyn_start_page: int = self.free_region_page_idx
        self.slot_size_pages: int = per_layer_flex_pages
        self.num_slots: int = self.prefetch_window
        self.dyn_slots: list[tuple[int, int] | None] = [None] * self.num_slots

        # Frozen intra-slot placement: each unpinned tensor of layer L lands
        # at slot_base + layer_tensor_offsets[L][tensor.id], regardless of
        # which slot L ends up occupying.
        self.layer_tensor_offsets: list[dict[int, int]] = []
        for unpinned in self.unpinned_per_layer:
            offsets: dict[int, int] = {}
            off = 0
            for t in unpinned:
                offsets[t.id] = off
                off += t.num_pages
            self.layer_tensor_offsets.append(offsets)

        # Runtime orchestration state. Node categorization is step-aware:
        # layer_node_ids[step][layer] holds the node ids for that invocation,
        # pre/post buckets are per-step. The same layer-shape info above is
        # reused across all steps.
        self.layer_node_ids, self.pre_node_ids_per_step, self.post_node_ids_per_step = \
            categorize_nodes(self.sys.trace.node_map)
        self.num_steps: int = len(self.layer_node_ids)

        # Reverse index for O(1) retire handling; only layer nodes populate it.
        # Pre- and post-layer computes retire too but are left out of the map,
        # so retire handling silently ignores them.
        self.node_to_step_layer: dict[int, tuple[int, int]] = {}
        for step in range(self.num_steps):
            for layer, nids in enumerate(self.layer_node_ids[step]):
                for nid in nids:
                    self.node_to_step_layer[nid] = (step, layer)

        # Per-(step, layer) outstanding compute-node counter.
        self.layer_nodes_remaining: list[list[int]] = [
            [len(nids) for nids in per_step] for per_step in self.layer_node_ids
        ]

        # Progress pointers and one-shot submission flags (filled in Phase 5).
        self.current_step_idx: int = 0
        self.current_layer_idx: int = 0
        self.next_prefetch_step_idx: int = 0
        self.next_prefetch_layer_idx: int = 0
        self.compute_submitted: set[tuple[int, int]] = set()
        self.pre_submitted: set[int] = set()
        self.post_submitted: set[int] = set()

        # Maps prefetch TransferJob id to the (step, layer) it's loading.
        self.prefetch_job_to_step_layer: dict[uuid.UUID, tuple[int, int]] = {}

        # [CLAUDE_END]


        # Submit transfer job
        if load_batch:
            self.sys.transfer(load_batch)

        return True

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        """ Help
        - Trace information to execute
          node_map = self.sys.trace.node_map (dict[node_id, Node])
          tensor_map = self.sys.trace.tensor_map (dict[tensor_id, Tensor])

        - How to release a memory region in memory for re-use
          mem_region = self.sys.find(self.memory, tensor)[0]
          self.sys.release(mem_region)

        - How to claim a memory region, for use or load (page_idx is a start address in memory for this tensor)
          mem_region = self.sys.claim(self.memory, tensor, page_idx)

        - How to load tensors from storage to memory (load them in batch for performance)
          load_batch = []
          mem_region = self.sys.find(self.memory, tensor)[0]
          stor_region = self.sys.find(self.storage, tensor)[0]
          load_batch.append((mem_region, stor_region))
          # ... (add more tuples to the load_batch)
          self.sys.transfer(load_batch)

        - How to request computation of a node
          self.sys.compute(self.compute, node)
        """


        # [CLAUDE_BEGIN]

        # Retire handling: update prefetch status and layer counters.
        for job in retired_jobs:
            if isinstance(job, TransferJob):
                sl = self.prefetch_job_to_step_layer.pop(job.id, None)
                if sl is not None:
                    _step, layer = sl
                    for tensor in self.unpinned_per_layer[layer]:
                        tensor.args["flexinfer"] = FlexInferStatus.MEMORY_LOADED
            elif isinstance(job, ComputeJob):
                sl = self.node_to_step_layer.get(job.node.id)
                if sl is not None:
                    step, layer = sl
                    self.layer_nodes_remaining[step][layer] -= 1

        # Advance current (step, layer) while its compute counter hits zero.
        # Wraps to the next step when we finish a step's last layer. Loops so
        # that a tick which retires the final node(s) of several consecutive
        # layers advances through all of them and frees their slots.
        while self.current_step_idx < self.num_steps:
            if self.current_layer_idx >= len(self.flex_layers):
                self.current_step_idx += 1
                self.current_layer_idx = 0
                continue
            if self.layer_nodes_remaining[self.current_step_idx][self.current_layer_idx] == 0:
                self._unload_layer(self.current_step_idx, self.current_layer_idx)
                self.current_layer_idx += 1
            else:
                break

        # Post-layer submission: a step's post nodes include the parent of the
        # next step's pre node, so they must be queued BEFORE the pre block.
        for step in range(self.num_steps):
            if step in self.post_submitted:
                continue
            if all(r == 0 for r in self.layer_nodes_remaining[step]):
                for nid in self.post_node_ids_per_step[step]:
                    self.sys.compute(self.compute, self.sys.trace.node_map[nid])
                self.post_submitted.add(step)

        # Pre-layer submission: one-shot per step, once current reaches it.
        if (self.current_step_idx < self.num_steps
                and self.current_step_idx not in self.pre_submitted):
            for nid in self.pre_node_ids_per_step[self.current_step_idx]:
                self.sys.compute(self.compute, self.sys.trace.node_map[nid])
            self.pre_submitted.add(self.current_step_idx)

        # Prefetch loop: start transfers for (step, layer) pairs within the
        # prefetch_window of current, until we run out of slots or window.
        num_layers = len(self.flex_layers)
        current_mega = self.current_step_idx * num_layers + self.current_layer_idx
        while self.next_prefetch_step_idx < self.num_steps:
            next_mega = (
                self.next_prefetch_step_idx * num_layers
                + self.next_prefetch_layer_idx
            )
            if next_mega >= current_mega + self.prefetch_window:
                break

            pref_step = self.next_prefetch_step_idx
            pref_layer = self.next_prefetch_layer_idx
            unpinned = self.unpinned_per_layer[pref_layer]

            if not unpinned:
                # No load needed; just advance the pointer.
                self._advance_prefetch_ptr()
                continue

            slot_idx = self._acquire_slot(pref_step, pref_layer)
            if slot_idx is None:
                # All slots busy; retry on a later tick once a layer frees one.
                break

            batch = []
            for tensor in unpinned:
                page_idx = self._tensor_page_idx(pref_layer, slot_idx, tensor)
                mem_region = self.sys.claim(self.memory, tensor, page_idx)
                if mem_region is None:
                    # sys.claim already signalled abort on the engine; stop
                    # here so we don't hand a None dest to sys.transfer.
                    return
                stor_region = self.sys.find(self.storage, tensor)[0]
                batch.append((stor_region, mem_region))
                tensor.args["flexinfer"] = FlexInferStatus.MEMORY_LOADING

            job_id = self.sys.transfer(batch)
            self.prefetch_job_to_step_layer[job_id] = (pref_step, pref_layer)
            self._advance_prefetch_ptr()

        # Compute submission for the current layer, once all its unpinned
        # tensors are loaded. Later layers wait for their turn as current
        # advances on future ticks.
        if self.current_step_idx < self.num_steps:
            cur_key = (self.current_step_idx, self.current_layer_idx)
            if cur_key not in self.compute_submitted:
                unpinned = self.unpinned_per_layer[self.current_layer_idx]
                if all(
                    t.args["flexinfer"] == FlexInferStatus.MEMORY_LOADED
                    for t in unpinned
                ):
                    for nid in self.layer_node_ids[self.current_step_idx][self.current_layer_idx]:
                        self.sys.compute(self.compute, self.sys.trace.node_map[nid])
                    self.compute_submitted.add(cur_key)

        # [CLAUDE_END]


        return
