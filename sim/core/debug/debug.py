import sys
from IPython.terminal.embed import InteractiveShellEmbed
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from enum import Enum, auto

from sim.core.sim_object import SimObject
from sim.core.log import Log, TrackID, Level


_shell = InteractiveShellEmbed()


@dataclass
class _Symbol:
    name: str
    _type: str
    description: str


class Debugger(SimObject):
    """
    Debugging tool
    """

    def __init__(self, obj_id: int, name: str, log: Log):
        super().__init__(obj_id, name, log)
        self._log = log

        # Create subtracks for logging
        self._log.record(Log.subtrack(TrackID.Engine, self.id, "Debug"))

        # Debugging flags
        self.BREAK_AFTER_TRACE_INIT: bool = False
        self.BREAK_AFTER_HW_INIT: bool = False
        self.BREAK_AFTER_COMPILE_STAGE: bool = False
        self.BREAK_AFTER_LAYOUT_STAGE: bool = False
        self.BREAK_AFTER_RUNTIME_STAGE: bool = False
        return

    def log(self) -> None:
        return

    def welcome_prompt(self) -> None:
        """
        Asks user to register breakpoints for debugging
        """
        return

    def break_after_trace_init(self) -> None:
        variables = [
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor")
        ]

        caller_frame = sys._getframe(1)
        _shell.mainloop(
            local_ns=caller_frame.f_locals,
            module=sys.modules.get(caller_frame.f_globals.get("__name__")),
        )
        return

    def break_after_hw_init(self, hw) -> None:
        variables = [
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
        ]

        caller_frame = sys._getframe(1)
        _shell.mainloop(
            local_ns=caller_frame.f_locals,
            module=sys.modules.get(caller_frame.f_globals.get("__name__")),
        )
        return

    def break_after_compile_stage(self) -> None:
        variables = [
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor")
        ]

        caller_frame = sys._getframe(1)
        _shell.mainloop(
            local_ns=caller_frame.f_locals,
            module=sys.modules.get(caller_frame.f_globals.get("__name__")),
        )
        return

    def break_after_layout_stage(self, hw) -> None:
        variables = [
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
        ]

        caller_frame = sys._getframe(1)
        _shell.mainloop(
            local_ns=caller_frame.f_locals,
            module=sys.modules.get(caller_frame.f_globals.get("__name__")),
        )
        return

    def break_after_runtime_stage(self) -> None:
        variables = [
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor")
        ]

        caller_frame = sys._getframe(1)
        _shell.mainloop(
            local_ns=caller_frame.f_locals,
            module=sys.modules.get(caller_frame.f_globals.get("__name__")),
        )
        return
