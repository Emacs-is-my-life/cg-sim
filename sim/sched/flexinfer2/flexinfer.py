from __future__ import annotations

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

from .utils import categorize_tensors


class FlexInferMode(Enum):
    MEMORY_SUFFICIENT = auto()
    MEMORY_INTERMEDIATE_2 = auto()
    MEMORY_INTERMEDIATE_1 = auto()
    MEMROY_LIMITED = auto()


class TensorSlot:
    def __init__(self, base_page_idx: int, num_pages: int):
        self.page_idx = base_page_idx
        self.num_pages = num_pages
        self.used: bool = False
        self.tensor: Tensor = None
        return

class LayerSlot:
    def __init__(self, base_page_idx: int, attn_num_pages: int, ffn_num_pages: int):
        self.base_page_idx = base_page_idx
        self.ffn_slots = [TensorSlot(base_page_idx + i * ffn_num_pages, ffn_num_pages) for i in range(3)]
        self.attn_slots = [TensorSlot(base_page_idx + 3 * ffn_num_pages + i * attn_num_pages) for i in range(2)]
        self.layer_idx: int = -1
        self.used: bool = False
        return

    def mark_layer(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.used = True
        return

    def clear_layer(self):
        for slot in self.ffn_slots:
            slot.used = False
            slot.tensor = None

        for slot in self.attn_slots:
            slot.used = False
            slot.tensor = None

        self.layer_idx = -1
        self.used = False
        return

def _check_flex_layer_ready(layer, sys, mem) -> bool:
    for tensor in layer.attn_big + layer.attn_small + layer.ffn:
        if not tensor.args["pin"]:
            mem_region = sys.find(mem, tensor)
            if mem_region is None:
                return False

    return True

class FlexInfer2(BaseScheduler):
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

        node_map = self.sys.trace.node_map
        _layer_start_node_id = -1
        _layer_end_node_id = -1
        for node in node_map.values():
            if "-0" in node.name:
                _layer_start_node_id = node.id
                break

        for node in node_map.values():
            if "-1" in node.name:
                _layer_end_node_id = node.id
                break

        self.num_nodes_in_a_layer = _layer_end_node_id - _layer_start_node_id

        # Tensor information gather
        self.flex_layers, self.other_tensors = categorize_tensors(sys.trace.tensor_map)
        num_layers = len(self.flex_layers)
        ffn_tensor_size = self.flex_layers[0].ffn[0].num_pages
        attn_tensor_size = self.flex_layers[0].attn_big[0].num_pages

        # Get memory requirement for other_tensors pages
        others_size_num_pages = 0
        for tensor in self.other_tensors:
            others_size_num_pages += tensor.num_pages

        # Check operation mode
        memory_size = self.memory.space.num_total_pages

        cost_sufficient = others_size_num_pages + num_layers * (3 * ffn_tensor_size + attn_tensor_size)
        cost_intermediate_2 = (
            others_size_num_pages
            + num_layers * (2 * ffn_tensor_size)
            + self.prefetch_window * (1 * ffn_tensor_size + 2 * attn_tensor_size)
        )
        cost_intermediate_1 = (
            others_size_num_pages
            + num_layers * (1 * ffn_tensor_size)
            + self.prefetch_window * (2 * ffn_tensor_size + 2 * attn_tensor_size)
        )
        minimum_size = others_size_num_pages + self.prefetch_window * (3 * ffn_tensor_size + 2 * attn_tensor_size)

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

        self.attn_tensor_size = attn_tensor_size
        self.ffn_tensor_size = ffn_tensor_size

        self.heap_num_pages = self.prefetch_window * (3 * ffn_tensor_size + 2 * attn_tensor_size)
        self.page_idx_heap_start = 0
        self.page_idx_heap_end = self.memory.space.num_total_pages

        self.current_layer = 0
        self.waiting_job_ids: set = set()
        self.heap = None
        return

    def compile(self, trace: Trace) -> None:
        """No compilation"""
        return

    def _build_payload(self, init_storage: BaseStorage, tensor: Tensor) -> tuple[MemoryRegion, StorageRegion]:
        mem_region = self.sys.claim(self.memory, tensor, self.page_idx_heap_start)
        self.page_idx_heap_start += tensor.num_pages
        stor_region = self.sys.find(init_storage, tensor)[0]
        return (stor_region, mem_region)

    def _can_pin_tensor(self, tensor: Tensor) -> bool:
        avail_pages = self.page_idx_heap_end - self.page_idx_heap_start
        if avail_pages - tensor.num_pages > self.heap_num_pages:
            return True
        else:
            return False

    def layout(self, init_storage: BaseStorage) -> None:
        """
        Input: Attention tensor size 𝑠𝑖𝑧𝑒𝑎𝑡𝑡𝑒 , FFN tensor size
        𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 , Layer number 𝑁 , Memory budget 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 ,
        Output: Tensor preservation plan 𝑃
        1: if 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 >= 𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 ∗ 𝑁 ∗ 3 + 𝑠𝑖𝑧𝑒𝑎𝑡𝑡𝑛 ∗ 𝑁 ∗ 2 then
        2:     Set all FFN tensors for all layers
        3: else
        4:     if 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 >= 𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 ∗ 𝑁 ∗ 2 then
        5:         Set two FFN tensor for all layers
        6:     else
        7:         if 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 >= 𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 ∗ 𝑁 then
        8:             Set one FFN tensor for all layers
        9:         end if
        10:    end if
        11: end if
        12: Set as much as possible attention tensors one by one
        13: return 𝑃
        """

        # Mark pinning info metadata first
        tensor_map = self.sys.trace.tensor_map
        for tensor in tensor_map.values():
            tensor.args["pin"] = False

        others_batch = []
        # Load/Claim All 'other' tensors in memory
        for tensor in self.other_tensors:
            tensor.args["pin"] = True
            if tensor.args["tensor_type"] == "INTERMEDIATE":
                # Claiming memory region in Memory is enough
                self.sys.claim(self.memory, tensor, self.page_idx_heap_start)
                self.page_idx_heap_start += tensor.num_pages
            else:
                # Claim a memory region, and load this tensor from the storage
                mem_region = self.sys.claim(self.memory, tensor, self.page_idx_heap_start)
                self.page_idx_heap_start += tensor.num_pages
                stor_region = self.sys.find(init_storage, tensor)[0]
                others_batch.append((stor_region, mem_region))

        # Pin Attn/FFN tensors
        pin_batch = []
        if self.mode == FlexInferMode.MEMORY_SUFFICIENT:
            # For all layer, pin all FFN
            for layer in self.flex_layers:
                for ffn in layer.ffn:
                    ffn.args["pin"] = True
                    pin_batch.append(self._build_payload(init_storage, ffn))
        elif self.mode == FlexInferMode.MEMORY_INTERMEDIATE_2:
            # For all layer, pin two FFN
            for layer in self.flex_layers:
                for ffn_idx in range(2):
                    layer.ffn[ffn_idx].args["pin"] = True
                    pin_batch.append(self._build_payload(init_storage, layer.ffn[ffn_idx]))
        elif self.mode == FlexInferMode.MEMORY_INTERMEDIATE_1:
            # For all layer, pin one FFN
            for layer in self.flex_layers:
                layer.ffn[0].args["pin"] = True
                pin_batch.append(self._build_payload(init_storage, layer.ffn[0]))

        # Try to pin ATTN every layer as much as possible (up to two), with per-tensor dynamic feasibility check
        for attn_idx in range(2):
            for layer in self.flex_layers:
                if attn_idx >= len(layer.attn_big):
                    continue

                attn = layer.attn_big[attn_idx]
                if attn.args["pin"]:
                    continue

                if self._can_pin_tensor(attn):
                    attn.args["pin"] = True
                    pin_batch.append(self._build_payload(init_storage, attn))

        # Try to pin remaining ATTN weights as much as possible, with per-tensor dynamic feasibility check
        its_over = False
        for layer in self.flex_layers:
            for attn in layer.attn_big:
                if not attn.args["pin"]:
                    if self._can_pin_tensor(attn):
                        attn.args["pin"] = True
                        pin_batch.append(self._build_payload(init_storage, attn))
                    else:
                        its_over = True
                        break

            if its_over:
                break

        # Begin Storage -> Memory Load
        self.sys.transfer(others_batch + pin_batch)

        # Prepare Heap region for Dynamic loading
        layer_slot_size = (3 * self.ffn_tensor_size + 2 * self.attn_tensor_size)
        self.heap = [LayerSlot(self.page_idx_heap_start + i * layer_slot_size, self.attn_tensor_size, self.ffn_tensor_size) for i in range(self.prefetch_window)]

        # Statistics Report
        total_memory_pages = self.memory.space.num_total_pages
        pinned_memory_pages = self.page_idx_heap_start
        heap_space_pages = self.page_idx_heap_end - self.page_idx_heap_start
        unpinned_attn_tensor_count = sum(
            1
            for layer in self.flex_layers
            for tensor in layer.attn_big
            if not tensor.args["pin"]
        )

        unpinned_ffn_tensor_count = sum(
            1
            for layer in self.flex_layers
            for tensor in layer.ffn
            if not tensor.args["pin"]
        )
        unpinned_tensor_count = unpinned_attn_tensor_count + unpinned_ffn_tensor_count

        self.log.on = True
        self.log.record(Log.engine(self.id, "SCHEDULER_MESSAGE", 0, {
            "from": self.name,
            "msg": "FlexInfer Memory Layout Summary.",
            "total_memory_pages": total_memory_pages,
            "pinned_memory_pages": pinned_memory_pages,
            "heap_space_pages": heap_space_pages,
            "unpinned_tensor_count": unpinned_tensor_count,
            "unpinned_attn_tensor_count": unpinned_attn_tensor_count,
            "unpinned_ffn_tensor_count": unpinned_ffn_tensor_count,
        }))

        self.log.on = False
        return

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        Forward = (len(self.waiting_job_ids) == 0)

        # Try to update current_layer number
        used_tensor_ids = set()
        for job in retired_jobs:
            if job.id in self.waiting_job_ids:
                self.waiting_job_ids.remove(job.id)
                if not self.waiting_job_ids:
                    Forward = True

            if isinstance(job, ComputeJob):
                node = job.node
                used_tensor_ids.update(node.input_tensors)
                used_tensor_ids.update(node.output_tensors)

        used_tensor_ids = list(used_tensor_ids)
        if used_tensor_ids:
            current_layer = self._find_layer_from_tensor(used_tensor_ids)
            if current_layer is not None:
                self.current_layer = current_layer

        # Retire not-needed layers from heap
        # Based on self.current_layer
        if self.current_layer >= 1:
            for layer_slot in self.heap:
                if layer_slot.layer_idx < (self.current_layer - 1) or layer_slot.layer_idx > (self.current_layer + self.prefetch_window - 1):
                    if layer_slot.used:
                        # release all memory region claimed by them
                        for tensor_slot in layer_slot.ffn_slots:
                            mem_region = self.sys.find(self.memory, tensor_slot.tensor)
                            self.sys.release(mem_region)

                        for tensor_slot in layer_slot.attn_slots:
                            mem_region = self.sys.find(self.memory, tensor_slot.tensor)
                            self.sys.release(mem_region)

                        layer_slot.clear_layer()

        batch = []
        # Look ahead
        for look_ahead in range(1, self.prefetch_window - 1):
            # Check if layer is not ready
            layer = self.flex_layers[self.current_layer + look_ahead]
            if _check_flex_layer_ready(layer, self.sys, self.mem):
                # Prepare batch to load
                # Find an empty slot in heap
                empty_slot = None
                for layer_slot in self.heap:
                    if not layer_slot.used:
                        empty_slot = layer_slot
                        break

                if empty_slot is None:
                    args = {
                        "from": self.name,
                        "msg": "[FlexInfer] Faild to find an empty slot in heap for dynamic allocation."
                    }
                    self.sys.abort(args)

                empty_slot.mark_layer(layer.N)
                for attn in layer.attn_big:
                    if (not attn.args["pin"]) and (attn.)


        # Prefetch Tensors from storage
        if batch:
            self.sys.transfer(batch)

        # Submit compute jobs
        if Forward:
            node_map = self.sys.trace.node_map
            todo_start = -1
            for node in node_map.values():
                if node.status == NodeStatus.TODO:
                    todo_start = node.id
                    break

            self.waiting_job_ids.clear()
            for node_id in range(todo_start, todo_start + self.num_nodes_in_a_layer):
                if node_id not in node_map:
                    continue

                node = node_map[node_id]
                job_id = self.sys.compute(self.compute, node)
                self.waiting_job_ids.add(job_id)
        return
