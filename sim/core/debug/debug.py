from __future__ import annotations

import json
import shutil
import sys
import textwrap
from IPython.terminal.embed import InteractiveShellEmbed
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from enum import Enum, auto

from sim.core.sim_object import SimObject
from sim.core.log import Log, TrackID, Level

if TYPE_CHECKING:
    from sim.core.engine import Engine


_shell = InteractiveShellEmbed()


@dataclass
class _Symbol:
    name: str
    _type: str
    description: str


def symbolPrint(variables: list[_Symbol]) -> None:
    """
    Pretty-print a list of `_Symbol` as an ASCII table with width-aligned
    columns. The `description` cell wraps across multiple lines when it
    exceeds the available width.
    """
    if not variables:
        return

    headers = ("Variable", "Type", "Description")
    name_w = max(len(headers[0]), *(len(v.name) for v in variables))
    type_w = max(len(headers[1]), *(len(v._type) for v in variables))

    # Borders + padding consume: "| " + " | " + " | " + " |" = 10 cols.
    term_w = shutil.get_terminal_size((100, 20)).columns
    desc_w = max(20, term_w - name_w - type_w - 10)

    sep = f"|{'-' * (name_w + 2)}+{'-' * (type_w + 2)}+{'-' * (desc_w + 2)}|"

    def _wrap_desc(desc: str) -> list[str]:
        lines: list[str] = []
        for paragraph in desc.splitlines() or [""]:
            lines.extend(textwrap.wrap(paragraph, width=desc_w) or [""])
        return lines

    def _row(name: str, _type: str, desc: str) -> str:
        desc_lines = _wrap_desc(desc)
        out: list[str] = []
        for i, dl in enumerate(desc_lines):
            n = name if i == 0 else ""
            t = _type if i == 0 else ""
            out.append(f"| {n:<{name_w}} | {t:<{type_w}} | {dl:<{desc_w}} |")
        return "\n".join(out)

    print(f"| {headers[0]:<{name_w}} | {headers[1]:<{type_w}} | {headers[2]:<{desc_w}} |")
    print(sep)
    for v in variables:
        print(_row(v.name, v._type, v.description))
    return


class Debugger(SimObject):
    """
    Debugging tool
    """

    def __init__(self, obj_id: int, name: str, log: Log):
        super().__init__(obj_id, name, log)
        self._log = log
        self.engine: Engine | None = None

        # Create subtracks for logging
        self._log.record(Log.subtrack(TrackID.Engine, self.id, "Debug"))

        # Debugging flags
        self.BREAK_AFTER_TRACE_INIT: bool = False
        self.BREAK_AFTER_HW_INIT: bool = False
        self.BREAK_AFTER_COMPILE_STAGE: bool = False
        self.BREAK_AFTER_LAYOUT_STAGE: bool = False
        self.BREAK_AFTER_RUNTIME_STAGE: bool = False

        # Active breakpoint context (set on entry, consumed by `help()`).
        self._current_breakpoint: str | None = None
        self._current_variables: list[_Symbol] = []
        return

    def log_counters(self) -> dict[str, Any] | None:
        return

    def log_states(self) -> dict[str, Any] | None:
        return

    def record(self, args: dict) -> None:
        try:
            json.dumps(args)
        except (TypeError, ValueError) as exc:
            print(f"Cannot log: argument is not JSON-serializable ({exc}).")
            return

        timestamp = 0 if self.engine is None else self.engine.timestamp_now
        self._log.record(Log.engine(self.id, "DEBUG_MSG", timestamp, args))
        return

    def welcome_prompt(self) -> None:
        """
        Asks user to register breakpoints for debugging.

        Prints an Org-style table of available breakpoints (the
        `BREAK_AFTER_*` flags) with their current On/Off status, then runs
        a REPL loop: each integer input toggles the matching breakpoint to
        On and re-renders the table. Returns when every breakpoint is On
        or when the user enters 'c'.
        """
        breakpoints = [
            name for name, value in vars(self).items()
            if name.startswith("BREAK_AFTER_") and isinstance(value, bool)
        ]
        if not breakpoints:
            return

        num_w = max(len("#"), len(str(len(breakpoints) - 1)))
        name_w = max(len("Breakpoint"), *(len(b) for b in breakpoints))
        stat_w = max(len("Status"), len("Off"))
        sep = f"|{'-' * (num_w + 2)}+{'-' * (name_w + 2)}+{'-' * (stat_w + 2)}|"
        table_h = len(breakpoints) + 2  # header + separator + rows

        def _print_table() -> None:
            print(f"| {'#':<{num_w}} | {'Breakpoint':<{name_w}} | {'Status':<{stat_w}} |")
            print(sep)
            for i, b in enumerate(breakpoints):
                status = "On" if getattr(self, b) else "Off"
                print(f"| {i:<{num_w}} | {b:<{name_w}} | {status:<{stat_w}} |")

        def _erase(n: int) -> None:
            if n > 0:
                # ESC[F = cursor up one line; ESC[2K = clear entire line.
                sys.stdout.write("\033[F\033[2K" * n)
                sys.stdout.flush()

        def _finish() -> None:
            # ESC[2J = erase entire screen; ESC[H = cursor to top-left.
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            print("Breakpoints registered. Starting simulator...")

        while True:
            _print_table()
            if all(getattr(self, b) for b in breakpoints):
                _finish()
                return

            # Inner loop: keep prompting until input is valid or 'c'. Each
            # `input()` consumes one screen line; each error message adds
            # one more — track both so we can erase everything on success.
            extra = 0
            while True:
                user_input = input("Enter breakpoint # to enable ('c' to continue): ").strip()
                extra += 1
                if user_input.lower() == "c":
                    _finish()
                    return
                try:
                    idx = int(user_input)
                    if 0 <= idx < len(breakpoints):
                        break
                    print(f"Out of range: {idx}")
                except ValueError:
                    print(f"Invalid input: {user_input!r}")
                extra += 1

            setattr(self, breakpoints[idx], True)
            _erase(table_h + extra)
        return

    def _print_banner(self, name: str) -> None:
        """
        Print the common header shown at every breakpoint, followed by an
        Org-style table of available REPL commands.
        """
        print()
        bar = "=" * max(60, len(name) + 20)
        print(bar)
        print(f"[BREAKPOINT] {name}")
        print(bar)
        print()

        commands = [
            ("debug.help()", "Show breakpoint context"),
            ("debug.record(dict)", "Leave information in simulation log file, in Engine -> Debug track."),
            ("exit()", "Continue simulator execution"),
        ]
        cmd_w = max(len("Command"), *(len(c) for c, _ in commands))
        desc_w = max(len("Description"), *(len(d) for _, d in commands))
        sep = f"|{'-' * (cmd_w + 2)}+{'-' * (desc_w + 2)}|"
        print(f"| {'Command':<{cmd_w}} | {'Description':<{desc_w}} |")
        print(sep)
        for cmd, desc in commands:
            print(f"| {cmd:<{cmd_w}} | {desc:<{desc_w}} |")
        print()
        return

    def help(self) -> None:
        """
        Re-print the entry message (banner + variable table) for the
        currently-active breakpoint. Intended to be called from the REPL.
        """
        if self._current_breakpoint is None:
            print("No active breakpoint.")
            return
        self._print_banner(self._current_breakpoint)
        symbolPrint(self._current_variables)
        return

    def _enter_breakpoint(self, name: str, variables: list[_Symbol]) -> None:
        """
        Register `name`/`variables` as the active breakpoint, print the
        entry message, and drop into the IPython shell scoped to the
        caller of the originating `break_after_*` method.
        """
        self._current_breakpoint = name
        self._current_variables = variables
        self.help()
        # Frames: 0=_enter_breakpoint, 1=break_after_*, 2=actual caller.
        caller_frame = sys._getframe(2)
        _shell.mainloop(
            local_ns=caller_frame.f_locals,
            module=sys.modules.get(caller_frame.f_globals.get("__name__")),
        )
        # Clear active breakpoint context and wipe the terminal so the
        # simulator's subsequent log output starts on a clean screen.
        self._current_breakpoint = None
        self._current_variables = []
        # ESC[2J = erase entire screen; ESC[H = move cursor to top-left.
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        print(f"[{name}] Resuming simulator execution...")
        return

    def break_after_trace_init(self) -> None:
        self._enter_breakpoint("break_after_trace_init", [
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor"),
        ])
        return

    def break_after_hw_init(self, hw) -> None:
        self._enter_breakpoint("break_after_hw_init", [
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
        ])
        return

    def break_after_compile_stage(self) -> None:
        self._enter_breakpoint("break_after_compile_stage", [
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor"),
        ])
        return

    def break_after_layout_stage(self, hw) -> None:
        self._enter_breakpoint("break_after_layout_stage", [
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
        ])
        return

    def break_in_runtime_stage(self) -> None:
        self._enter_breakpoint("break_after_runtime_stage", [
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor"),
        ])
        return

    def break_after_runtime_stage(self) -> None:
        self._enter_breakpoint("break_after_runtime_stage", [
            _Symbol("hw", "dict[str, BaseHardware]", "Dictionary: hw_name -> BaseHardware"),
            _Symbol("trace", "Trace", "Execution trace"),
            _Symbol("trace.node_map", "dict[int, Node]", "Dictionary: node_id -> Node"),
            _Symbol("trace.tensor_map", "dict[int, Tensor]", "Dictionary: tensor_id -> Tensor"),
        ])
        return
