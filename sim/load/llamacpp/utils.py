import re
from typing import Any

from sim.core.trace import Tensor


NODE_LAYER_PAT = re.compile(r'-(\d+)\b')
NODE_POST_PAT = re.compile(r'^(norm|result_)')
def categorize_node(name: str) -> str:
    if NODE_POST_PAT.search(name):
        return "POST"

    m = NODE_LAYER_PAT.search(name)
    if m:
        return str(int(m.group(1)))

    return "PRE"

TENSOR_LAYER_PAT = re.compile(r'(?:-|_l|blk\.)(\d+)\b')
def get_node_layer(name):
    m = TENSOR_LAYER_PAT.search(name)
    return int(m.group(1)) if m else None


def categorize_tensor(name: str) -> str:
    layer = get_node_layer(name)

    if layer is None:
        if re.search(r'(output|result|final)', name):
            return "POST"
        else:
            return "PRE"
    else:
        layer


def node_name_canonicalizer(label: str) -> str:
    """Dot graph node label -> canonical_name"""
    first_part = label.split('|')[0]
    canonical_name = re.sub(r'\s\([^)]*\)$', '', first_part)
    return canonical_name


def get_tensor_type(label: str) -> str:
    """Returns a TensorType, inferred from the label name of the node"""
    if label.startswith("<x>"):
        if ".weight" in label:
            return "WEIGHT"
        elif ("cache_k" in label) or ("cache_v" in label):
            return "KVCACHE"
        elif "leaf" in label:
            return "LEAF"
        elif "inp_embd" in label:
            return "INPUT"
    elif ("cache_k" in label) or ("cache_v" in label):
        return "KVCACHE"
    elif "leaf" in label:
        return "LEAF"
    elif "inp_embd" in label:
        return "INPUT"
    else:
        return "INTERMEDIATE"


class TensorWithSign(Tensor):
    """
    Tensor data type, but with its runtime address as a signature
    Signature is required to distinguish real Tensor and virtual Tensor from traces
    """
    def __init__(self, tensor_id: int, tensor_name: str, size_bytes: int = 0, args: dict[str, Any] | None = None, tensor_sign: str = "X"):
        args = args if args is not None else {}
        super().__init__(tensor_id, tensor_name, size_bytes, args)
        self.sign = tensor_sign
        return

    def get_Tensor(self):
        return Tensor(self.id, self.name, self.size_bytes, self.args)


def get_real_tensor_id(TensorWithSignMap: dict[int, TensorWithSign], tensor_sign: str) -> int:
    """
    Using tensor_sign, look for the existing Tensor in TensorWithSignMap.
    If there is no such Tensor, returns -1.
    """

    for tensor_id, tensor in TensorWithSignMap.items():
        if tensor_sign == tensor.sign:
            return tensor_id

    return -1
