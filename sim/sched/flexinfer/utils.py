from sim.core.trace import Tensor
import re


class Layer:
    def __init__(self, layer_number: int):
        self.N = layer_number
        self.attn: list[Tensor] = []
        self.ffn: list[Tensor] = []
        return


def categorize_tensors(tensor_map: dict[int, Tensor]) -> tuple[list[Layer], dict[int, int], int]:
    others_size_num_pages: int = 0
    tensor_layer: dict[int, int] = {}

    layer_pattern = re.compile(r"^blk\.(\d+)\.")

    max_layer = -1
    for tensor in tensor_map.values():
        match = layer_pattern.match(tensor.name)
        if match is not None:
            layer_num = int(match.group(1))
            if layer_num > max_layer:
                max_layer = layer_num

    layers = [Layer(i) for i in range(max_layer + 1)]

    for tensor in tensor_map.values():
        name = tensor.name
        tensor_type = tensor.args.get("tensor_type")

        match = layer_pattern.match(name)
        if tensor_type == "WEIGHT" and match is not None:
            n_this_layer = int(match.group(1))
            tensor.args["layer"] = n_this_layer
            layer = layers[n_this_layer]

            if "attn_q" in name or "attn_output" in name:
                layer.attn.append(tensor)
                tensor_layer[tensor.id] = n_this_layer
            elif "ffn_down" in name or "ffn_gate" in name or "ffn_up" in name:
                layer.ffn.append(tensor)
                tensor_layer[tensor.id] = n_this_layer
            else:
                others_size_num_pages += tensor.num_pages
                tensor_layer[tensor.id] = -1
        else:
            others_size_num_pages += tensor.num_pages
            tensor_layer[tensor.id] = -1

    return layers, tensor_layer, others_size_num_pages
