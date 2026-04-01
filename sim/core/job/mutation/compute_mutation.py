from sim.core import System
from sim.core.job import ComputeJob
from sim.core.trace import NodeStatus, Node

from sim.hw.common import DataRegionAccess
from sim.hw.memory.common import BaseMemory

from .utils import invalidate


def begin_mutation(job: ComputeJob, sys: System) -> None:
    node: Node = job.node
    hw = job.running_on[0]

    # 0. Add job to hardware running slot
    for hw in job.running_on:
        hw.run(job)

    # 1. Mark node as running
    node.status = NodeStatus.RUNNING

    # 2. Regions holding input Tensors
    for i_mem_region in job.input_regions:
        i_mem_region.access_status = DataRegionAccess.BEING_READ
        i_mem_region.access_count += 1

    # 3. Output Tensors
    # Invalidate all regions holding this Tensor
    for o_tensor_id in node.output_tensors:
        invalidate(sys, o_tensor_id)

    # Regions holding output Tensors
    for o_mem_region in job.output_regions:
        o_mem_region.access_status = DataRegionAccess.BEING_WRITTEN
        o_mem_region.is_latest = True
        o_mem_region.is_ready = False

    return


def end_mutation(job: ComputeJob, sys: System) -> None:
    node: Node = job.node
    hw = job.running_on[0]
    memory: BaseMemory = hw.memory

    # 0. Retire job from hardware running slot
    for hw in job.running_on:
        hw.retire(job)

    # 1. Mark node as done
    node.status = NodeStatus.DONE

    # 2. Input Tensors
    for i_tensor_id in node.input_tensors:
        candidates = memory.space.get_by_tensor_id(i_tensor_id)

        for i_mem_region in candidates:
            i_mem_region.access_count -= 1
            if i_mem_region.access_count == 0:
                i_mem_region.access_status = DataRegionAccess.IDLE

    # 3. Output Tensors
    for o_tensor_id in node.output_tensors:
        candidates = memory.space.get_by_tensor_id(o_tensor_id)

        for o_mem_region in candidates:
            o_mem_region.access_status = DataRegionAccess.IDLE
            o_mem_region.is_ready = True

    return
