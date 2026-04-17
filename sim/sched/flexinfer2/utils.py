from sim.core.trace import Tensor
import re


class Layer:
    def __init__(self, layer_number: int):
        self.N = layer_number
        self.attn_big: list[Tensor] = []
        self.attn_small: list[Tensor] = []
        self.ffn: list[Tensor] = []
        return


def categorize_tensors(tensor_map: dict[int, Tensor]) -> tuple[list[Layer], list[Tensor]]:
    layer_pattern = re.compile(r"^blk\.(\d+)\.")
    max_layer = -1
    for tensor in tensor_map.values():
        m = layer_pattern.match(tensor.name)
        if m is not None:
            layer_num = int(m.group(1))
            if layer_num > max_layer:
                max_layer = layer_num

    flex_layers = [Layer(i) for i in range(max_layer + 1)]
    other_tensors = []

    p_attn_big = r"^blk\.(\d+)\.(attn_(?:q|output)\.weight$"
    p_attn_small = r"^blk\.(\d+)\.(attn_(?:k|v)\.weight$"
    p_ffn = r"^blk\.(\d+)\.ffn_(?:down|up|gate))\.weight$"
    for tensor in tensor_map.values():
        m = re.match(p_attn_big)
        if m:
            # Attention Big Tensor
            layer_idx = int(m.group(1))
            flex_layers[layer_idx].attn_big.append(tensor)
            continue

        m = re.match(p_attn_small)
        if m:
            # Attention Small Tensor
            layer_idx = int(m.group(1))
            flex_layers[layer_idx].attn_small.append(tensor)
            continue

        m = re.match(p_ffn)
        if m:
            # FFN Tensor
            layer_idx = int(m.group(1))
            flex_layers[layer_idx].ffn.append(tensor)
            continue

        # Didn't match any of above
        other_tensors.append(tensor)

    return flex_layers, other_tensors
