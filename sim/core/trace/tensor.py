from typing import Any
import numpy as np


"""
class TensorType(Enum):
    INPUT = auto()           # mutable: no  / write back on eviction: no  / reuse: yes / inter-step reuse: no
    WEIGHT = auto()          # mutable: no  / write back on eviction: no  / resue: yes / inter-step reuse: yes
    INTERMEDIATE = auto()    # mutable: yes / write back on eviction: yes / reuse: yes / inter-step reuse: no
    KVCache = auto()         # mutable: yes / write back on eviction: yes / reuse: yes / inter-step reuse: yes

Don't track the above in the Tensor data structure itself.
Just add them in args field, as hints
"""


class Tensor:
    """Tensor represents data being processed in model inference"""
    def __init__(self, tensor_id: int, tensor_name: str, size_bytes: int = 0, args: dict[str, Any] | None = None):
        self.id: int = tensor_id
        self.name: str = tensor_name
        self.size_bytes: int = size_bytes
        self.args: dict[str, Any] = args if args is not None else {}

        align_bytes = 64        # 64 B in AMD64 for optimal performance
        page_size_bytes = 4096  # 4 kB in AMD64
        tensor_aligned_size_bytes = ((size_bytes + align_bytes - 1) // align_bytes) * align_bytes
        self.num_pages: int = int(np.ceil(tensor_aligned_size_bytes / page_size_bytes))
