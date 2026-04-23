from __future__ import annotations

from typing import TYPE_CHECKING

from sim.core.trace import NodeStatus
from sim.hw.common.data_region import DataRegionAccess

if TYPE_CHECKING:
    from sim.core.system import System
    from ..compute_job import ComputeJob


def assertion(job: ComputeJob, sys: System) -> bool:
    # 0. Hardware Availability
    hw = job.running_on[0]
    if "HW_type" in job.node.args:
        if "HW_type" in hw.args:
            if job.node.args["HW_type"] != hw.args["HW_type"]:
                sys.abort({"from": sys.engine.name, "msg": f"You cannot run a job with HW_type: {job.node.args['HW_type']} on {hw.args['HW_type']}"})
        else:
            sys.abort({"from": sys.engine.name, "msg": f"You cannot run a job with HW_type: {job.node.args['HW_type']} on {hw.name}"})

    if not hw.can_run(job):
        return False

    node = job.node
    node_map = sys.trace.node_map
    # 1. Control Dependency
    for p_node_id in node.parent_nodes:
        p_node = node_map[p_node_id]
        if p_node.status != NodeStatus.DONE:
            return False

    memory = hw.memory    # Tensors must be in compute hardware's local memory
    # 2. Data Dependency
    # Input Tensors
    for i_tensor_id in node.input_tensors:
        candidates = memory.space.get_by_tensor_id(i_tensor_id)
        if len(candidates) == 0:
            return False

        FOUND = False
        for i_mem_region in candidates:
            OK = i_mem_region.is_ready and i_mem_region.is_latest and \
                (i_mem_region.access_status in (DataRegionAccess.IDLE, DataRegionAccess.BEING_READ))

            if OK:
                FOUND = True
                break

        if not FOUND:
            return False

    # Output Tensors
    for o_tensor_id in node.output_tensors:
        candidates = memory.space.get_by_tensor_id(o_tensor_id)
        if len(candidates) == 0:
            return False

        FOUND = False
        for o_mem_region in candidates:
            OK = o_mem_region.access_status == DataRegionAccess.IDLE
            if OK:
                FOUND = True
                break

        if not FOUND:
            return False

    return True
