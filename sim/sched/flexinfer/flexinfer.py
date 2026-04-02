from __future__ import annotations

from enum import Enum, auto
from typing import Any, TYPE_CHECKING

from sim.sched.common import BaseScheduler
from sim.core.log import Log
from sim.core.trace import Trace, Tensor
from sim.core.job import BaseJob, ComputeJob, ClaimJob, ReleaseJob, TransferJob
from sim.hw.compute.common import BaseCompute
from sim.hw.memory.common import BaseMemory, MemoryRegion
from sim.hw.storage.common import BaseStorage, StorageRegion

if TYPE_CHECKING:
    from sim.core.system import System

from .utils import categorize_tensors


class FlexInferMode(Enum):
    MEMORY_SUFFICIENT = auto()
    MEMORY_INTERMEDIATE = auto()
    MEMROY_LIMITED = auto()


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
        self.layers, self.tensor_layer_num, others_size_num_pages = categorize_tensors(sys.trace.tensor_map)
        attn_tensor_size = self.layers[0].attn[0].num_pages
        ffn_tensor_size = self.layers[0].ffn[0].num_pages
        num_layers = len(self.layers)

        # Check operation mode
        memory_size = self.memory.space.num_total_pages
        minimum_size = self.prefetch_window * (3 * ffn_tensor_size + 2 * attn_tensor_size) + others_size_num_pages
        if memory_size < minimum_size:
            args = {
                "from": self.name,
                "msg": f"Memory size is too small to run FlexInfer policy. {4 * minimum_size} KB required, have {4 * memory_size} KB."
            }
            sys.abort(args)

        args = {"from": self.name, "msg": ""}
        if (memory_size > (3 * num_layers * ffn_tensor_size + 2 * num_layers * attn_tensor_size) + others_size_num_pages):
            self.mode = FlexInferMode.MEMORY_SUFFICIENT
            args["msg"] = f"FlexInfer Scheduler operating in MEMORY_SUFFICIENT mode."
        elif (memory_size > num_layers * ffn_tensor_size + others_size_num_pages):
            self.mode = FlexInferMode.MEMORY_INTERMEDIATE
            args["msg"] = f"FlexInfer Scheduler operating in MEMORY_INTERMEDIATE mode."
        else:
            self.mode = FlexInferMode.MEMROY_LIMITED
            args["msg"] = f"FlexInfer Scheduler operating in MEMORY_LIMITED mode."

        self.log(Log.engine(self.id, "SCHEDULER_MESSAGE", 0, args))


        self.attn_tensor_size = attn_tensor_size
        self.ffn_tensor_size = ffn_tensor_size

        self.must_reserve_pages = self.prefetch_window * (3 * ffn_tensor_size + 2 * attn_tensor_size)

        self.page_idx_heap_start = 0
        self.page_idx_heap_end = self.memory.space.total_num_pages
        return

    def compile(self, trace: Trace) -> None:
        """No compilation"""
        return

    def _build_payload(self, init_storage: BaseStorage, tensor: Tensor) -> tuple[MemoryRegion, StorageRegion]:
        mem_region = self.sys.claim(self.memory, tensor, self.page_idx_heap_start)
        self.page_idx_heap_start += tensor.num_pages
        stor_region = self.sys.find(init_storage, tensor)[0]
        return (mem_region, stor_region)

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
        others_batch = []
        # Load/Claim All 'other' tensors in memory
        for tensor_id, layer_num in self.tensor_layer_num:
            if layer_num == -1:
                tensor = tensor_map[tensor_id]
                if tensor.args["tensor_type"] == "INTERMEDIATE":
                    # Claiming memory region is enough
                    self.sys.claim(self.memory, tensor, self.page_idx_heap_start)
                    self.page_idx_heap_start += tensor.num_pages
                else:
                    # Claim a memory region, and load tensor from the storage
                    mem_region = self.sys.claim(self.memory, tensor, self.page_idx_heap_start)
                    self.page_idx_heap_start += tensor.num_pages
                    stor_region = self.sys.find(init_storage, tensor)[0]
                    others_batch.append((mem_region, stor_region))

        # Pin Attn/FFN tensors
        pin_batch = []
        if self.mode == FlexInferMode.MEMORY_SUFFICIENT:
            for layer in self.layers:     # For every layer,
                for tensor in layer.ffn:  # Pin all FFN Tensors and one Attn Tensor
                    pin_batch.append(self._build_payload(init_storage, tensor))

                pin_batch.append(self._build_payload(init_storage, layer.attn[0]))

        if self.mode == FlexInferMode.MEMORY_INTERMEDIATE:
            num_pages_avail = (self.page_idx_heap_end - self.page_idx_heap_start) - self.must_reserve_pages

            # TODO: Implement
            # Consider self.attn_tensor_size, self.ffn_tensor_size and num_pages_avail,
            # 1. Try to pin FFN tensors in memory, in uniform manner (all layers have similar number of pinned FFN tensors)
            # 2. If space is still available, pin Attn tensors as much as possible, in uniform manner
            pass
        if self.mode == FlexInferMode.MEMROY_LIMITED:
            num_pages_avail = (self.page_idx_heap_end - self.page_idx_heap_start) - self.must_reserve_pages

            # TODO: Implement
            # Consider self.attn_tensor_size, self.ffn_tensor_size and num_pages_avail,
            # 1. Try to pin Attn tensors in memory, in uniform manner
            pass

        self.sys.transfer(others_batch + pin_batch)
        return

    def runtime(self, retired_jobs: list[BaseJob]) -> None:
        return
