from sim.core.trace import Tensor, Node
import re


class Layer:
    def __init__(self, layer_number: int):
        self.N = layer_number

        self.attn_big: list[Tensor] = []
        self.attn_small: list[Tensor] = []
        self.ffn: list[Tensor] = []
        return


# Matches a trailing "-<N>" optionally followed by parenthesised qualifiers
# e.g. "Qcur-0", "Qcur-0 (reshaped)", "cache_k_l0 (view) (permuted)" (via the second regex)
_node_layer_dash_re = re.compile(r"-(\d+)(?:\s+\(.*\))?$")
_node_layer_ul_re = re.compile(r"_l(\d+)(?:\s+\(.*\))?$")

_PRE_LAYER_NAMES = {"inp_embd"}
_POST_LAYER_NAMES = {"norm", "result_norm", "result_output", "TERMINAL_NODE"}


def categorize_nodes(
    node_map: dict[int, Node],
) -> tuple[list[list[list[int]]], list[list[int]], list[list[int]]]:
    """
    Group nodes per (step, layer), with per-step pre- and post-layer buckets.

    Returns (layer_node_ids, pre_node_ids, post_node_ids) where:
      layer_node_ids[step][layer] -> ordered list of node ids
      pre_node_ids[step]          -> ordered list of step's pre-layer node ids
      post_node_ids[step]         -> ordered list of step's post-layer node ids
                                     (TerminalNode lands in the final step's
                                      post bucket, so simulation exits only
                                      after all steps complete)

    Step is taken from node.args["step"] (set by the loader). TerminalNode
    has no step arg and is placed into the last step's post bucket.

    Unlabelled nodes (e.g. " (copy)", "node_986") inherit the most recently
    seen layer within their own step, matching their graph position.
    """
    # First pass: determine dimensions
    max_step = 0
    max_layer = -1
    for node in node_map.values():
        step = node.args.get("step")
        if step is not None and step > max_step:
            max_step = step
        m = _node_layer_dash_re.search(node.name) or _node_layer_ul_re.search(node.name)
        if m is not None:
            layer = int(m.group(1))
            if layer > max_layer:
                max_layer = layer

    num_steps = max_step + 1
    num_layers = max_layer + 1

    layer_node_ids: list[list[list[int]]] = [
        [[] for _ in range(num_layers)] for _ in range(num_steps)
    ]
    pre_node_ids: list[list[int]] = [[] for _ in range(num_steps)]
    post_node_ids: list[list[int]] = [[] for _ in range(num_steps)]

    # Second pass: place nodes within their own step
    last_layer_per_step: dict[int, int] = {}
    in_post_per_step: dict[int, bool] = {}

    for nid, node in node_map.items():
        # TerminalNode has no step arg; attach it to the last step's post
        step = node.args.get("step", num_steps - 1)

        m = _node_layer_dash_re.search(node.name) or _node_layer_ul_re.search(node.name)
        if m is not None:
            layer = int(m.group(1))
            layer_node_ids[step][layer].append(nid)
            last_layer_per_step[step] = layer
            continue

        name = node.name
        if name in _POST_LAYER_NAMES:
            post_node_ids[step].append(nid)
            in_post_per_step[step] = True
            continue

        if name in _PRE_LAYER_NAMES and step not in last_layer_per_step:
            pre_node_ids[step].append(nid)
            continue

        # Unlabelled node. Attach within this step to the current region.
        if in_post_per_step.get(step, False):
            post_node_ids[step].append(nid)
        elif step not in last_layer_per_step:
            pre_node_ids[step].append(nid)
        else:
            layer_node_ids[step][last_layer_per_step[step]].append(nid)

    return layer_node_ids, pre_node_ids, post_node_ids


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
