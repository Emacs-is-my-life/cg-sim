from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

from .node import NodeStatus

if TYPE_CHECKING:
    from sim.core.system import System
    from sim.core.job.compute_job import ComputeJob


class CustomDep(ABC):
    """
    User-defined runnability predicate attached to a Node via
    `node.custom_deps`, for exotic operations that do not fit the standard
    data-flow model (barriers, side-effect-only ops, non-local I/O, ...).

    Semantics: if `node.custom_deps` is non-empty, compute_assertion
    BYPASSES its built-in control/input/output checks and relies solely
    on these predicates. Hardware admission (`hw.can_run` and the
    NodeHW-vs-hw-class match) still runs unconditionally, so an exotic
    op cannot be dispatched to a busy or wrong-type device.

    Contract for `check` (matches existing assertion functions):
      - return True  -> dep is satisfied
      - return False -> not runnable yet; engine retries next tick
      - to fail fatally: call sys.abort(args) AND return False
    """

    @abstractmethod
    def check(self, job: "ComputeJob", sys: "System") -> bool:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NodeDoneDep(CustomDep):
    """Block until another node (not necessarily a graph parent) is DONE."""

    def __init__(self, node_id: int):
        self.node_id = node_id

    def check(self, job: "ComputeJob", sys: "System") -> bool:
        return sys.trace.node_map[self.node_id].status == NodeStatus.DONE

    def __repr__(self) -> str:
        return f"NodeDoneDep(node_id={self.node_id})"


class TensorAtHWDep(CustomDep):
    """
    Block until `tensor_id` has at least one ready+latest DataRegion on the
    hardware tagged `custom_dep_str`. Useful when the built-in check (which
    only inspects the *compute hw's local memory*) is not what you want --
    e.g. "wait until the tensor has been written back to storage".

    The tag refers to a hw entry whose YAML config sets
    `args.custom_dep: "<this string>"`. Tags must be unique across all
    hardware; uniqueness is enforced by Simulator startup validation.
    """

    def __init__(self, tensor_id: int, custom_dep_tag: str):
        self.tensor_id = tensor_id
        self.custom_dep_tag = custom_dep_tag

    def check(self, job: "ComputeJob", sys: "System") -> bool:
        # Linear scan over sys.hw is fine -- simulations have few devices,
        # and startup validation guarantees exactly one match.
        for hw in sys.hw.values():
            if hw.args.get("custom_dep_tag") == self.custom_dep_tag:
                regions = sys.find(hw, self.tensor_id)
                return any(r.is_ready and r.is_latest for r in regions)
        return False

    def __repr__(self) -> str:
        return f"TensorAtHWDep(tensor_id={self.tensor_id}, custom_dep_tag={self.custom_dep_tag!r})"


class MinTimestampDep(CustomDep):
    """Block until simulation time reaches `t_micros`."""

    def __init__(self, t_micros: float):
        self.t_micros = t_micros

    def check(self, job: "ComputeJob", sys: "System") -> bool:
        return sys.engine.timestamp_now >= self.t_micros

    def __repr__(self) -> str:
        return f"MinTimestampDep(t_micros={self.t_micros})"


class LambdaDep(CustomDep):
    """
    Escape hatch for one-off lambdas, so you don't subclass for every case.
    Not serializable; prefer a real subclass for anything reusable.
    """

    def __init__(
        self,
        fn: Callable[["ComputeJob", "System"], bool],
        label: str = "<lambda>",
    ):
        self.fn = fn
        self.label = label

    def check(self, job: "ComputeJob", sys: "System") -> bool:
        return self.fn(job, sys)

    def __repr__(self) -> str:
        return f"LambdaDep({self.label})"
