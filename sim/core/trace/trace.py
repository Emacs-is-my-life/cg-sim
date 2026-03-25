from typing import Any

from .node import Node
from .tensor import Tensor


def map_check(NodeMap: dict[int, Node], TensorMap: dict[int, Tensor]):
    """Check if there is no missing Node or Tensor in Maps"""

    for node in NodeMap.values():
        # Check if input tensors are all defined in the TensorMap
        for tensor_id in node.id_input_tensors:
            if not TensorMap.get(tensor_id):
                raise Exception(f"Node {node.id}'s input tensor: Tensor {tensor_id} does not exist in the TensorMap.")

        # Check if output tensors are all defined in the TensorMap
        for tensor_id in node.id_output_tensors:
            if not TensorMap.get(tensor_id):
                raise Exception(f"Node {node.id}'s output tensor: Tensor {tensor_id} does not exist in the TensorMap.")

    return


# TODO: Trace class
