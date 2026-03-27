from typing import Any

from sim.core import SimObject
from sim.core.log import Log, TrackID

from .node import Node, TerminalNode
from .tensor import Tensor


# TODO: implement graph traversal check of node_map
def map_check(node_map: dict[int, Node], tensor_map: dict[int, Tensor]):
    """Check if there is no missing Node or Tensor in Maps"""

    for node in node_map.values():
        # Check if input tensors are all defined in the TensorMap
        for tensor_id in node.input_tensors:
            if tensor_id not in tensor_map:
                raise Exception(f"[Loader] Node {node.id}'s input tensor: Tensor {tensor_id} does not exist in the TensorMap.")

        # Check if output tensors are all defined in the TensorMap
        for tensor_id in node.output_tensors:
            if tensor_id not in tensor_map:
                raise Exception(f"[Loader] Node {node.id}'s output tensor: Tensor {tensor_id} does not exist in the TensorMap.")

        # Check if last node is present
        last_node_id = next(reversed(node_map))
        last_node = node_map[last_node_id]
        if not isinstance(last_node, TerminalNode):
            raise Exception(f"[Loader] The last node in node_map, is not TerminalNode. Simulation will either end prematurely, or never end.")

    return


class Trace(SimObject):
    """
    Datastructure representing the ML workload
    """

    def __init__(self, obj_id: int, name: str, log: Log, node_map: dict[int, Node], tensor_map: dict[int, Tensor]):
        super().__init__(obj_id, name, log)

        map_check(node_map, tensor_map)
        self.node_map: dict[int, Node] = node_map
        self.tensor_map: dict[int, Tensor] = tensor_map

        self.log.record(Log.subtrack(TrackID.State, self.id, self.name))
        return

    def log_counters(self) -> dict[str, Any] | None:
        """Log its counter according to logging format"""
        return None

    def log_states(self) -> dict[str, Any] | None:
        """Log its state according to logging format"""

        states = {
            "nodes": []
        }

        for _, node in self.node_map.items():
            step = 0
            if "step" in node.args:
                step = node.args["step"]

            states["nodes"].append({
                "step": step,
                "id": node.id,
                "name": node.name,
                "status": node.status.name
            })

        return states
