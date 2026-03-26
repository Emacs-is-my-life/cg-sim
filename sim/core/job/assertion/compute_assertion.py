from sim.core import System
from sim.core.job import ComputeJob
from sim.core.trace import NodeStatus
from sim.hw.common import DataRegionAccess


def assertion(job: ComputeJob, sys: System) -> bool:
    # 0. Hardware Availability
    hw = job.running_on[0]
    if not hw.is_avail():
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
        candidates = memory.space.get_by_tensor_id[i_tensor_id]
        if len(candidates) == 0:
            return False

        i_mem_region = candidates[0]
        OK = i_mem_region.is_ready and i_mem_region.is_latest and \
            (i_mem_region.access_status in (DataRegionAccess.IDLE, DataRegionAccess.BEING_READ))
        if not OK:
            return False

    # Output Tensors
    for o_tensor_id in node.output_tensors:
        candidates = memory.space.get_by_tensor_id[o_tensor_id]
        if len(candidates) == 0:
            return False

        o_mem_region = candidates[0]
        OK = o_mem_region.access_status == DataRegionAccess.IDLE
        if not OK:
            return False

    return True
