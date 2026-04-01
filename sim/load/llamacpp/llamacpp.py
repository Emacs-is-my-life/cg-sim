from pathlib import Path
import networkx as nx
from networkx.drawing.nx_agraph import read_dot
import polars as pl


from sim.core.trace import Node, TerminalNode, Tensor, Trace, TraceLoader
from sim.hw.storage.common import BaseStorage

from .utils import node_name_canonicalizer, get_tensor_type, TensorWithSign, get_real_tensor_id


class Llamacpp(TraceLoader):
    """Trace loader for llama.cpp profiling data"""

    def load(self) -> Trace:
        input_dir = Path(self.args["input_path"]).parent

        # Load the dot graph
        dot_graph_path = Path(self.args["graph_path"])
        if not dot_graph_path.is_absolute():
            dot_graph_path = input_dir / dot_graph_path

        dot_graph = read_dot(dot_graph_path) 

        # Load profiled record data
        record_path = Path(self.args["record_path"])
        if not record_path.is_absolute():
            record_path = input_dir / record_path

        df_profile_records = pl.read_csv(record_path, has_header=True)

        # Data structures for tracking Nodes and Tensors
        vNodeMap: dict[int, Node] = {}
        vTensorMap: dict[int, TensorWithSign] = {}
        VerticeMap: dict[int, Node | TensorWithSign] = {}

        # ID tracker. Increment by 1 everytime adding a Node/Tensor
        v_node_id = 0
        v_tensor_id = 0

        # Iterate over all Vertices(Nodes + Tensors) in the dot graph
        for v_id, v_attr in dot_graph.nodes(data=True):
            v_label = v_attr["label"]
            tensor_type = get_tensor_type(v_label)
            args = {
                "tensor_type": tensor_type
            }
            tensor_sign = v_attr["addr"]
            tensor_size_bytes = int(v_attr["size"])

            # This vertice is only a tensor, not a computation node (= leaf)
            if v_label.startswith('<x>'):
                tensor_name = node_name_canonicalizer(v_label[3:])  # Ditch "<x>" part

                __tensor_id = get_real_tensor_id(vTensorMap, tensor_sign)
                if __tensor_id == -1:
                    new_tensor = TensorWithSign(v_tensor_id, tensor_name, tensor_size_bytes, args, tensor_sign)
                    vTensorMap[v_tensor_id] = new_tensor
                    v_tensor_id += 1

                    VerticeMap[v_id] = new_tensor
                else: 
                    VerticeMap[v_id] = vTensorMap[__tensor_id]

            # This vertice is a node, actual computation 
            else:
                tensor_name = node_name_canonicalizer(v_label)

                __tensor_id = get_real_tensor_id(vTensorMap, tensor_sign)
                if __tensor_id == -1:
                    new_tensor = TensorWithSign(v_tensor_id, tensor_name, tensor_size_bytes, args, tensor_sign)
                    vTensorMap[v_tensor_id] = new_tensor
                    __tensor_id = v_tensor_id
                    v_tensor_id += 1

                # Create a new (Virtual) Node
                node_name = tensor_name
                compute_time_micros = -1  # Virtual Node
                new_node = Node(v_node_id, node_name, compute_time_micros, {})
                new_node.add_output_tensor(__tensor_id)
                vNodeMap[v_node_id] = new_node
                v_node_id += 1

                VerticeMap[v_id] = new_node

        # Iterate over edges in the dot graph, to install dependency between:
        # Node <-> Node    (Control Dependency)
        # Node <-> Tensor  (Data Dependency)
        for parent_v_id, child_v_id in dot_graph.edges():
            child = VerticeMap[child_v_id]
            parent = VerticeMap[parent_v_id]

            # Parent is a Node (Control Dependency)
            if isinstance(parent, Node):
                # Add Control Dependency
                parent.add_child_node(child.id)
                child.add_parent_node(parent.id)

                # Add Data Dependency
                for tensor_id in parent.output_tensors:
                    child.add_input_tensor(tensor_id)

            # Parent is a Tensor (Leaf, Data Dependency)
            else:
                # Add Data Dependency
                child.add_input_tensor(parent.id)

        # Prepare real NodeMap and TensorMap
        NodeMap: dict[int, Node] = {}
        TensorMap: dict[int, Tensor] = {}

        # vTensorMap -> TensorMap
        for tensor in vTensorMap.values():
            real_tensor = Tensor(tensor.id, tensor.name, tensor.size_bytes, tensor.args)
            TensorMap[real_tensor.id] = real_tensor

        # vNodeMap + df_profile_records -> NodeMap
        node_id = 0
        step_till = 10
        for step in range(step_till+1):
            df_step = df_profile_records.filter(
                pl.col("step") == step
            )

            step_node_id_base = node_id  # Starting node_id of this step

            for _node_data in df_step.iter_rows():
                _node_id = _node_data[1]
                _node_name = _node_data[2]
                _node_compute_time_ns = _node_data[4]
                _node_compute_time_micros = float(_node_compute_time_ns / 1_000)

                new_node = Node(node_id, _node_name, _node_compute_time_micros, {"step": step})
                if (step > 0) and (_node_id == 0):
                    # Link to the existing compute graph.
                    last_node = NodeMap[node_id - 1]

                    # Add Control Dependency
                    new_node.add_parent_node(last_node.id)
                    last_node.add_child_node(new_node.id)

                    # Add Data Dependency
                    for tid in last_node.output_tensors:
                        new_node.add_input_tensor(tid)

                # Copy traits from the vNodeMap
                virtual_node = vNodeMap[_node_id]
                for nid in virtual_node.parent_nodes:
                    new_node.add_parent_node(step_node_id_base + nid)

                for nid in virtual_node.children_nodes:
                    new_node.add_child_node(step_node_id_base + nid)

                for tid in virtual_node.input_tensors:
                    new_node.add_input_tensor(tid)

                for tid in virtual_node.output_tensors:
                    new_node.add_output_tensor(tid)

                # Add new_node to NodeMap
                NodeMap[node_id] = new_node
                node_id += 1

        # Add a last node to NodeMap
        # Last node is a mechanism to notify the simulator that it has reached the end
        prev_node = NodeMap[node_id - 1]
        last_node = TerminalNode(node_id, "TERMINAL_NODE")
        last_node.add_parent_node(prev_node.id)
        prev_node.add_child_node(node_id)
        NodeMap[node_id] = last_node
        node_id += 1

        return Trace(self.id, self.name, self.log, NodeMap, TensorMap)

    def placement(self, trace: Trace, storage: BaseStorage) -> None:
        tensor_map = trace.tensor_map

        for tensor in tensor_map:
            """
            Tensors are one of:

            - WEIGHT
            - KVCACHE
            - INPUT
            - INTERMEDIATE
            """

            tensor_type = tensor.args["tensor_type"]
            if tensor_type in ("WEIGHT", "INPUT"):
                # Place tensor in the storage device
                stor_region = storage.space.claim(tensor.id, -1, tensor.num_pages)
                stor_region.is_ready = True
                stor_region.is_latest = True
                # Other tensors are generated by result of computation
                # They don't exist on startup.

        return
