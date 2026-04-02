from sim.core.trace import Tensor


class Layer:
    def __init__(self, layer_number: int):
        self.N = layer_number
        self.attn: list[Tensor]
        self.ffn: list[Tensor]
        return


def categorize_tensors(tensor_map: dict[int, Tensor]) -> (list[Layer], dict[int, int], int):
    others_size_num_pages: int = 0
    tensor_layer: dict[int, int] = {}

    max_layer = 0
    for tensor in tensor_map.values():
        # if this tensor is part of a layer:
        #    tensor.name is like: blk.<layer>.*
        #    Get layer_num
        #    if layer_num > max_layer:
        #      max_layer = layer_num
        pass

    layers = [Layer(i) for i in range(max_layer)]
    for tensor in tensor_map.values():
        name = tensor.name
        # if [tensor.args["tensor_type"] == "WEIGHT"  and from name, check if this tensor is part of a layer]
        #    n_this_layer = [Pull layer number from the tensor name]
        #    layer = layers[n_this_layer]
        #
        #    if ["attn_q" or "attn_output" in name]
        #      layer.attn.append(tensor)
        #      tensor_layer[tensor.id] = n_this_layer
        #    elif ["ffn_down" or "ffn_gate" or "ffn_up" in name]
        #      layer.ffn.append(tensor)
        #      tensor_layer[tensor.id] = n_this_layer
        #    else
        #      others_size_num_pages += tensor.num_pages
        #      tensor_layer[tensor.id] = -1
        # else
        #      others_size_num_pages += tensor.num_pages
        #      tensor_layer[tensor.id] = -1
        pass

    return
