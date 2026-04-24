from sim.core.trace import Tensor
import re


class Layer:
    def __init__(self, layer_number: int):
        self.N = layer_number

        self.attn_big = list[Tensor]
        self.attn_small = list[Tensor]
        self.ffn = list[Tensor]
        return


def categorize_tensors(tensor_map: dict[int, Tensor]) -> tuple[list[Layer], list[Tensor]]:
    other_tensors: list[Tensor] = []
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

            tensor.args["flexinfer_loaded"] = False

            if "attn_q" in name or "attn_output" in name:
                layer.attn_big.append(tensor)
            elif "attn_k" in name or "attn_v" in name:
                layer.attn_small.append(tensor)
            elif "ffn_down" in name or "ffn_gate" in name or "ffn_up" in name:
                layer.ffn.append(tensor)
            else:
                other_tensors.append(tensor)
        else:
            other_tensors.append(tensor)

    return layers, other_tensors
