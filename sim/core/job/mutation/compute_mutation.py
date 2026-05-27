from __future__ import annotations

from typing import TYPE_CHECKING
import copy

from sim.core.trace import NodeStatus, Node

from sim.hw.common.data_region import DataRegionAccess
from sim.hw.memory.common import BaseMemory

from .utils import invalidate

if TYPE_CHECKING:
    from sim.core.system import System
    from ..compute_job import ComputeJob


def begin_mutation(job: ComputeJob, sys: System) -> None:
    node: Node = job.node
    hw = job.running_on[0]
    memory = hw.memory

    # 0. Add job to hardware running slot
    for hw in job.running_on:
        hw.run(job)

    # 1. Mark node as running
    node.status = NodeStatus.RUNNING
    # Clear binding regions (just in case)
    job.input_regions = []
    job.output_regions = []

    # 2. Regions holding input Tensors
    for i_tensor_id in node.input_tensors:
        candidates = memory.space.get_by_tensor_id(i_tensor_id)

        for i_mem_region in candidates:
            OK = i_mem_region.is_ready and i_mem_region.is_latest and \
                (i_mem_region.access_status in (DataRegionAccess.IDLE, DataRegionAccess.BEING_READ))
            if OK:
                job.input_regions.append(i_mem_region)
                break

    for i_mem_region in job.input_regions:
        i_mem_region.access_status = DataRegionAccess.BEING_READ
        i_mem_region.access_count += 1

    # 3. Output Tensors
    # Only for real_output - Some tensor are used for both input and output!
    real_output_tensors = copy.deepcopy(node.output_tensors)
    real_output_tensors = [tensor_id for tensor_id in real_output_tensors if tensor_id not in node.input_tensors]

    # Invalidate all regions holding this Tensor
    for o_tensor_id in real_output_tensors:
        candidates = memory.space.get_by_tensor_id(o_tensor_id)

        for o_mem_region in candidates:
            OK = o_mem_region.access_status == DataRegionAccess.IDLE
            if OK:
                job.output_regions.append(o_mem_region)
                break

    for o_tensor_id in real_output_tensors:
        invalidate(sys, o_tensor_id)

    # Regions holding output Tensors
    for o_mem_region in job.output_regions:
        o_mem_region.access_status = DataRegionAccess.BEING_WRITTEN
        o_mem_region.is_latest = True
        o_mem_region.is_ready = False

    return


def end_mutation(job: ComputeJob, sys: System) -> None:
    node: Node = job.node

    # 0. Retire job from hardware running slot
    for hw in job.running_on:
        hw.retire(job)

    # 1. Mark node as done
    node.status = NodeStatus.DONE

    # 2. Input Tensors
    for i_mem_region in job.input_regions:
        i_mem_region.access_count -= 1
        if i_mem_region.access_count == 0:
            i_mem_region.access_status = DataRegionAccess.IDLE

    # 3. Output Tensors
    for o_mem_region in job.output_regions:
        o_mem_region.access_status = DataRegionAccess.IDLE
        o_mem_region.is_latest = True
        o_mem_region.is_ready = True

    return
