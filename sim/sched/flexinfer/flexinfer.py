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


class Heap:
    def __init__(self, page_start: int, page_end: int):
        self.page_start = page_start
        self.page_end = page_end
        self.cap = self.page_end - self.page_start

        self.allocs = []
        return

    def free_spans(self) -> list[tuple[int, int]]:
        spans = []

        if not self.allocs:
            return [(0, self.cap)]

        first_start, _, _ = self.allocs[0]
        if first_start > 0:
            spans.append((0, first_start))

        for i in range(len(self.allocs) - 1):
            cur_start, cur_size, _ = self.allocs[i]
            next_start, _, _ = self.allocs[i + 1]
            gap_start = cur_start + cur_size
            gap_end = next_start
            if gap_start < gap_end:
                spans.append((gap_start, gap_end))

        last_start, last_size, _ = self.allocs[-1]
        end = last_start + last_size
        if end < self.cap:
            spans.append((end, self.cap))

        return spans

    def _idx(self, idx: int):
        return self.page_start + idx

    def alloc(self, tensor: Tensor) -> int | None:
        tensor_size = tensor.num_pages

        # First alloc
        if not self.allocs:
            if tensor_size <= self.cap:
                self.allocs.append((0, tensor_size, tensor))
                return self._idx(0)
            return None

        # Check before first block
        first_start, _, _ = self.allocs[0]
        if first_start >= tensor_size:
            self.allocs.insert(0, (0, tensor_size, tensor))
            return self._idx(0)

        # Check gaps between blocks
        for i in range(len(self.allocs) - 1):
            cur_start, cur_size, _ = self.allocs[i]
            next_start, _, _ = self.allocs[i + 1]

            gap_start = cur_start + cur_size
            gap_size = next_start - gap_start

            if gap_size >= tensor_size:
                self.allocs.insert(i + 1, (gap_start, tensor_size, tensor))
                return self._idx(gap_start)

        # Check after last block
        last_start, last_size, _ = self.allocs[-1]
        end = last_start + last_size

        if self.cap - end >= tensor_size:
            self.allocs.append((end, tensor_size, tensor))
            return self._idx(end)

        return None

    def free(self, free_tensor: Tensor) -> bool:
        for i, (start, tensor_size, tensor) in enumerate(self.allocs):
            if tensor.id == free_tensor.id:
                del self.allocs[i]
                return True

        return False


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
        self.layers, self.tensor_layer_num, others_size_num_pages = categorize_tensors(sys.trace.tensor_map)
        attn_tensor_size = self.layers[0].attn[0].num_pages
        ffn_tensor_size = self.layers[0].ffn[0].num_pages
        num_layers = len(self.layers)

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

        self.must_reserve_pages = 0

        self.page_idx_heap_start = 0
        self.page_idx_heap_end = self.memory.space.num_total_pages

        self.current_layer = 0
        self.waiting_job_ids: set = set()

        self.heap: Heap = None
        return

    def compile(self, trace: Trace) -> None:
        """No compilation"""
        return

    def _build_payload(self, init_storage: BaseStorage, tensor: Tensor) -> tuple[MemoryRegion, StorageRegion]:
        mem_region = self.sys.claim(self.memory, tensor, self.page_idx_heap_start)
        self.page_idx_heap_start += tensor.num_pages
        stor_region = self.sys.find(init_storage, tensor)[0]
        return (stor_region, mem_region)

    def _dynamic_pages_for_window(self, start_layer: int, window_size: int | None = None) -> int:
        window = self.prefetch_window if window_size is None else window_size
        total = 0
        N_layers = len(self.layers)

        for offset in range(window):
            layer = self.layers[(start_layer + offset) % N_layers]
            for tensor in layer.attn:
                if not tensor.args["pin"]:
                    total += tensor.num_pages
            for tensor in layer.ffn:
                if not tensor.args["pin"]:
                    total += tensor.num_pages

        return total

    def _max_dynamic_pages_over_all_windows(self, window_size: int | None = None) -> int:
        if not self.layers:
            return 0
        return max(self._dynamic_pages_for_window(i, window_size) for i in range(len(self.layers)))

    def _can_pin_tensor(self, tensor: Tensor) -> bool:
        tensor.args["pin"] = True
        try:
            required_dynamic_pages = self._max_dynamic_pages_over_all_windows()
        finally:
            tensor.args["pin"] = False

        remaining_pages = self.page_idx_heap_end - (self.page_idx_heap_start + tensor.num_pages)
        return remaining_pages >= required_dynamic_pages

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

        tensor_map = self.sys.trace.tensor_map
        for tensor in tensor_map.values():
            tensor.args["pin"] = False     # Initialize as False

        others_batch = []
        # Load/Claim All 'other' tensors in memory
        for tensor_id, layer_num in self.tensor_layer_num.items():
            if layer_num == -1:
                tensor = tensor_map[tensor_id]
                tensor.args["pin"] = True
                if tensor.args["tensor_type"] == "INTERMEDIATE":
                    # Claiming memory region is enough
                    self.sys.claim(self.memory, tensor, self.page_idx_heap_start)
                    self.page_idx_heap_start += tensor.num_pages
                else:
                    # Claim a memory region, and load tensor from the storage
                    mem_region = self.sys.claim(self.memory, tensor, self.page_idx_heap_start)
                    if mem_region is None:
                        print("[FlexInfer] mem_region is None!")

                    self.page_idx_heap_start += tensor.num_pages
                    stor_region = self.sys.find(init_storage, tensor)[0]
                    others_batch.append((stor_region, mem_region))

        # Pin Attn/FFN tensors
        pin_batch = []
        if self.mode == FlexInferMode.MEMORY_SUFFICIENT:
            # For all layer, pin all FFN
            for layer in self.layers:
                for ffn in layer.ffn:
                    ffn.args["pin"] = True
                    pin_batch.append(self._build_payload(init_storage, ffn))
        elif self.mode == FlexInferMode.MEMORY_INTERMEDIATE_2:
            # For all layer, pin two FFN
            for layer in self.layers:
                for ffn_idx in range(2):
                    layer.ffn[ffn_idx].args["pin"] = True
                    pin_batch.append(self._build_payload(init_storage, layer.ffn[ffn_idx]))
        elif self.mode == FlexInferMode.MEMORY_INTERMEDIATE_1:
            # For all layer, pin one FFN
            for layer in self.layers:
                layer.ffn[0].args["pin"] = True
                pin_batch.append(self._build_payload(init_storage, layer.ffn[0]))

        # Try to pin ATTN every layer as much as possible (up to two), with per-tensor dynamic feasibility check
        for attn_idx in range(2):
            for layer in self.layers:
                if attn_idx >= len(layer.attn):
                    continue

                attn = layer.attn[attn_idx]
                if attn.args["pin"]:
                    continue

                if self._can_pin_tensor(attn):
                    attn.args["pin"] = True
                    pin_batch.append(self._build_payload(init_storage, attn))

        # Try to pin remaining ATTN weights as much as possible, with per-tensor dynamic feasibility check
        its_over = False
        for layer in self.layers:
            for attn in layer.attn:
                if not attn.args["pin"]:
                    num_pages_avail = self.page_idx_heap_end - self.page_idx_heap_start
                    if num_pages_avail >= attn.num_pages and self._can_pin_tensor(attn):
                        attn.args["pin"] = True
                        pin_batch.append(self._build_payload(init_storage, attn))
                    else:
                        its_over = True
                        break

            if its_over:
                break

        self.must_reserve_pages = self._max_dynamic_pages_over_all_windows()

        required_dynamic_pages = self._max_dynamic_pages_over_all_windows()
        heap_cap_pages = self.page_idx_heap_end - self.page_idx_heap_start
        if required_dynamic_pages > heap_cap_pages:
            self.sys.abort({
                "from": self.name,
                "msg": "[FlexInfer] Layout produced insufficient heap for dynamic working set.",
                "required_dynamic_pages": required_dynamic_pages,
                "heap_cap_pages": heap_cap_pages,
                "page_idx_heap_start": self.page_idx_heap_start,
                "page_idx_heap_end": self.page_idx_heap_end,
            })
            return

        self.sys.transfer(others_batch + pin_batch)

        total_memory_pages = self.memory.space.num_total_pages
        pinned_memory_pages = self.page_idx_heap_start
        heap_space_pages = self.page_idx_heap_end - self.page_idx_heap_start
        unpinned_attn_tensor_count = sum(
            1
            for layer in self.layers
            for tensor in layer.attn
            if not tensor.args["pin"]
        )

        unpinned_ffn_tensor_count = sum(
            1
            for layer in self.layers
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

        self.heap: Heap = Heap(self.page_idx_heap_start, self.page_idx_heap_end)
        return

    def _find_layer_from_tensor(self, tensor_ids: list[int]) -> int | None:
        tensor_map = self.sys.trace.tensor_map
        layer = None
        for tensor_id in tensor_ids:
            tensor = tensor_map[tensor_id]
            if "layer" in tensor.args:
                _layer = tensor.args["layer"]
                if layer is None or _layer < layer:
                    layer = _layer

        return layer

    def _heap_used_pages(self) -> int:
        return sum(num_pages for _, num_pages, _ in self.heap.allocs)

    def _get_needed_tensor_ids(self) -> list[int]:
        N_layers = len(self.layers)
        ids = []
        seen = set()

        for offset in range(self.prefetch_window):
            layer = self.layers[(self.current_layer + offset) % N_layers]
            for tensor in layer.attn:
                if tensor.id not in seen:
                    seen.add(tensor.id)
                    ids.append(tensor.id)
            for tensor in layer.ffn:
                if tensor.id not in seen:
                    seen.add(tensor.id)
                    ids.append(tensor.id)

        return ids

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

        needed_tensor_ids = self._get_needed_tensor_ids()
        tensors_in_heap = []
        for _, _, tensor in self.heap.allocs:
            tensors_in_heap.append(tensor.id)

        needed_dynamic_tensor_ids = []
        needed_dynamic_pages = 0
        resident_needed_dynamic_tensor_ids = []
        missing_needed_dynamic_tensor_ids = []
        missing_needed_dynamic_pages = 0

        tensor_map = self.sys.trace.tensor_map
        for tensor_id in needed_tensor_ids:
            tensor = tensor_map[tensor_id]
            if not tensor.args["pin"]:
                needed_dynamic_tensor_ids.append(tensor_id)
                needed_dynamic_pages += tensor.num_pages
                if tensor_id in tensors_in_heap:
                    resident_needed_dynamic_tensor_ids.append(tensor_id)
                else:
                    missing_needed_dynamic_tensor_ids.append(tensor_id)
                    missing_needed_dynamic_pages += tensor.num_pages

        retire_tensor_ids = list(set(tensors_in_heap) - set(needed_tensor_ids))

        # Release won't be needed tensors
        for tensor_id in retire_tensor_ids:
            tensor = tensor_map[tensor_id]
            if not tensor.args["pin"]:
                if self.heap.free(tensor):
                    mem_region = self.sys.find(self.memory, tensor)[0]
                    self.sys.release(mem_region)

        # Load will-be-needed tensors
        batch = []
        for tensor_id in needed_tensor_ids:
            tensor = tensor_map[tensor_id]
            if not tensor.args["pin"]:       # Dynamic Tensor
                mem_region = self.sys.find(self.memory, tensor_id)
                if not mem_region:
                    alloc_page_idx = self.heap.alloc(tensor)
                    if alloc_page_idx is None:
                        args = {
                            "from": self.name,
                            "msg": "[FlexInfer] Failed to claim MemoryRegion for Dynamic Tensor",
                            "current_layer": self.current_layer,
                            "prefetch_window": self.prefetch_window,
                            "heap": {
                                "page_idx_heap_start": self.page_idx_heap_start,
                                "page_idx_heap_end": self.page_idx_heap_end,
                                "cap_pages": self.heap.cap,
                                "used_pages": self._heap_used_pages(),
                                "free_pages": self.heap.cap - self._heap_used_pages(),
                                "free_spans": self.heap.free_spans(),
                            },
                            "tensor": {
                                "id": tensor.id,
                                "name": tensor.name,
                                "requested_tensor_size": tensor.num_pages,
                                "pin": tensor.args["pin"],
                                "layer": tensor.args.get("layer"),
                            },
                            "needed_tensor_ids": needed_tensor_ids,
                            "needed_dynamic_tensor_ids": needed_dynamic_tensor_ids,
                            "needed_dynamic_pages": needed_dynamic_pages,
                            "missing_needed_dynamic_tensor_ids": missing_needed_dynamic_tensor_ids,
                            "missing_needed_dynamic_pages": missing_needed_dynamic_pages,
                            "resident_tensor_ids_in_heap": tensors_in_heap,
                        }
                        self.sys.abort(args)
                        return

                    mem_region = self.sys.claim(self.memory, tensor, alloc_page_idx)
                    if mem_region is None:
                        print("mem_region claim failed!")
                    stor_region = self.sys.find(self.storage, tensor)[0]
                    batch.append((stor_region, mem_region))

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
