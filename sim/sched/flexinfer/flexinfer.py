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
        attn_big_num_pages = self.layers[0].attn_big[0].num_pages
        attn_small_num_pages = self.layers[0].attn_small[0].num_pages
        ffn_num_pages = self.layers[0].ffn[0].num_pages
        num_layers = len(self.layers)

        # Get total size of "other tensors"
        others_size_num_pages = 0
        for tensor in self.other_tensors:
            others_size_num_pages += tensor.num_pages

        # Check operation mode
        memory_size = self.memory.space.num_total_pages

        cost_sufficient = others_size_num_pages + num_layers * (3 * ffn_num_pages + attn_big_num_pages)
        cost_intermediate_2 = (
            others_size_num_pages
            + num_layers * (2 * ffn_num_pages)
            + self.prefetch_window * (1 * ffn_num_pages + 2 * attn_big_num_pages)
        )
        cost_intermediate_1 = (
            others_size_num_pages
            + num_layers * (1 * ffn_num_pages)
            + self.prefetch_window * (2 * ffn_num_pages + 2 * attn_big_num_pages)
        )
        minimum_size = others_size_num_pages + self.prefetch_window * (3 * ffn_num_pages + 2 * attn_big_num_pages)

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

    def layout(self, init_storage: BaseStorage) -> bool:
        """  FlexInfer Algorithm
        Input: Attention tensor size 𝑠𝑖𝑧𝑒𝑎𝑡𝑡𝑒 , FFN tensor size
        𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 , Layer number 𝑁 , Memory budget 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 ,
        Output: Tensor preservation plan 𝑃
        1: if 𝑠𝑖𝑧𝑒𝑚𝑒𝑚 >= 𝑠𝑖𝑧𝑒𝐹 𝐹 𝑁 ∗ 𝑁 ∗ 3 + 𝑠𝑖𝑧𝑒𝑎𝑡𝑡𝑛 ∗ 𝑁 ∗ 2 then
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

        # Pin "FlexInfer" tensors, based on FlexInfer Algorithm

        # Pin tensors based on algorithm
        for layer in self.flex_layers:
            match self.mode:
                case FlexInferMode.MEMORY_SUFFICIENT:
                    pass
                case FlexInferMode.MEMORY_INTERMEDIATE_2:
                    pass
                case FlexInferMode.MEMORY_INTERMEDIATE_1:
                    pass
                case FlexInferMode.MEMROY_LIMITED:
                    pass

        # Pin attn tensors, if possible

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



        # [CLAUDE_END]


        return
