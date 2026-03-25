from typing import Any

from sim.core import SimObject

from .node import Node
from .tensor import Tensor


def map_check(node_map: dict[int, Node], tensor_map: dict[int, Tensor]):
    """Check if there is no missing Node or Tensor in Maps"""

    for node in node_map.values():
        # Check if input tensors are all defined in the TensorMap
        for tensor_id in node.id_input_tensors:
            if tensor_id not in tensor_map:
                raise Exception(f"Node {node.id}'s input tensor: Tensor {tensor_id} does not exist in the TensorMap.")

        # Check if output tensors are all defined in the TensorMap
        for tensor_id in node.id_output_tensors:
            if tensor_id not in tensor_map:
                raise Exception(f"Node {node.id}'s output tensor: Tensor {tensor_id} does not exist in the TensorMap.")

    return


class Trace(SimObject):
    """
    Datastructure representing the ML workload
    """

    def __init__(self, node_map: dict[int, Node], tensor_map: dict[int, Tensor]):
        obj_id: int = 0
        name = "Trace"
        super().__init__(obj_id, name)

        map_check(node_map, tensor_map)
        self.node_map: dict[int, Node] = node_map
        self.tensor_map: dict[int, Tensor] = tensor_map
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
            step = -1
            if "step" in node.args:
                step = node.args["step"]

            states["nodes"].append({
                "step": step,
                "id": node.id,
                "name": node.name,
                "status": node.status.name
            })

        return states
