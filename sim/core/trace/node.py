from __future__ import annotations
from enum import Enum, auto
from typing import Any


class NodeStatus(Enum):
    """Execution status of a Node"""
    TODO = auto()      # Not scheduled yet
    WAITING = auto()   # Node is in the job queue
    RUNNING = auto()   # Node is being processed now
    DONE = auto()      # Finished execution


class Node:
    """
    Node represents an atomic unit of workload, in a compute graph.
    Only actual computation nodes! (GGML_OP_NONE nodes (leaf node) in llama.cpp aren't real Node in our definition.)
    """

    def __init__(self, step: int, node_id: int, node_name: str,  compute_time_micros: float, node_status: NodeStatus = NodeStatus.TODO, node_args: dict[str, Any] | None = None):
        """Initialize a Node, with it's computation characteristics"""
        self.step: int = step
        self.id: int = node_id
        self.name: int = node_name
        self.compute_time_micros: float = compute_time_micros
        self.status: NodeStatus = node_status
        self.args: dict[str, Any] = {}

        """
        tensor_input:  To check if tensors are in memory for execution of this node
        tensor_output: Same as above
        """
        self.id_input_tensors: [int] = []  # Data Dependency
        self.id_output_tensors: [int] = []

        """
        node_parents:  Required to check if previous jobs are finished
        node_children: For easier traversal of compute graph

        These will be initialized through Node.add_parent(Node) method.
        """
        self.id_parent_nodes: [int] = []   # Control Dependency
        self.id_children_nodes: [int] = []
        return

    def add_parent_node(self, node_id: int):
        """
        Adds a parent Node, for building a Compute Graph
        To fully add a (common) Node as a parent
        """

        # Check deuplicates
        for p_node_id in self.id_parent_nodes:
            if node_id == p_node_id:
                return

        self.id_parent_nodes.append(node_id)
        return

    def add_child_node(self, node_id: int):
        for c_node_id in self.id_children_nodes:
            if node_id == c_node_id:
                return

        self.id_children_nodes.append(node_id)
        return

    def add_input_tensor(self, tensor_id: int):
        """Adds an input Tensor, for Tensor placement check before computation"""

        # Check duplicates
        for i_tensor_id in self.id_input_tensors:
            if tensor_id == i_tensor_id:
                return

        self.id_input_tensors.append(tensor_id)
        return

    def add_output_tensor(self, tensor_id: int):
        # Check duplicates
        for o_tensor_id in self.id_output_tensors:
            if tensor_id == o_tensor_id:
                return

        self.id_output_tensors.append(tensor_id)
        return
