import queue
import time
import threading
from pathlib import Path
import orjson
from enum import Enum, auto
from typing import Any


class TrackID(Enum):
    Engine = 0
    Event = 1
    Counter = 2
    State = 3


class Level(Enum):
    ENGINE = 0
    EVENT = 1
    COUNTER = 2
    STATE = 3


class Log:
    """
    Log collects log from modules(core, compute, mm, disk, ...),
    then periodically flush them to disk.
    """

    def __init__(self, args: dict, flush_period: float = 0.5):
        # Set log output file
        output_path = Path("./sim_run_result.json")
        node_path = Path("./node_map.json")
        tensor_path = Path("./tensor_map.json")

        if "output_path" in args:
            p = Path(args["output_path"])
            if not p.is_absolute():
                base_path = Path.cwd()
                p = base_path / p
                node_path = base_path / node_path
                tensor_path = base_path / tensor_path

            if p.suffix:
                p.parent.mkdir(parents=True, exist_ok=True)
            else:
                p.mkdir(parents=True, exist_ok=True)
                p = p / Path("sim_run_result.json")
                node_path = p / node_path
                tensor_path = p / tensor_path

            output_path = p

        self.output_path = output_path
        self.node_path = node_path
        self.tensor_path = tensor_path

        # On/Off Switch
        self.on = True
        # Set logging level
        log_level = Level.EVENT
        if "log_level" in args:
            _level = int(args["log_level"])
            if _level == 1:
                log_level = Level.EVENT
            elif _level == 2:
                log_level = Level.COUNTER
            else:
                log_level = Level.STATE
        self.level = log_level

        # Set logging infrastructure
        self.log_queue = queue.Queue()
        self.flush_period = flush_period
        self.file_ptr = None
        self._first_event = True

        # Prepare another thread for background flushing
        self.stop_event = threading.Event()
        self.worker = None
        return

    def _open_file(self):
        self.file_ptr = open(self.output_path, "w", encoding="utf-8")
        self._first_event = True
        self.file_ptr.write('{"traceEvents": [\n')
        self.file_ptr.flush()

    def _close_file(self):
        self.file_ptr.write("\n]}\n")
        self.file_ptr.flush()
        self.file_ptr.close()

    def _flush(self):
        """Drain the log_queue and write events to the json file on disk."""
        log_events = []
        while True:
            try:
                log_events.append(self.log_queue.get_nowait())
            except queue.Empty:
                break

        if not log_events:
            return

        parts = []
        for log_event in log_events:
            if self._first_event:
                self._first_event = False
            else:
                parts.append(",\n")

            parts.append(orjson.dumps(log_event, option=orjson.OPT_INDENT_2).decode().replace("  ", "\t"))

        self.file_ptr.write("".join(parts))
        self.file_ptr.flush()

    def _run(self):
        """
        Internal loop run by the log writer thread, periodically flushing log events to the disk.
        """
        while not self.stop_event.is_set():
            self._flush()
            time.sleep(self.flush_period)

        # Cleanup
        self._flush()
        self._close_file()
        return

    def _create_tracks(self):
        self.record(Log.track(TrackID.Engine, "Simulation"))
        self.record(Log.track(TrackID.Event, "Event"))
        self.record(Log.track(TrackID.Counter, "Counter"))
        self.record(Log.track(TrackID.State, "State"))
        return

    def start(self):
        """
        Starts the log writer module, running in a separate thread
        """

        if self.file_ptr is not None:
            return

        self._open_file()
        self.worker = threading.Thread(target=self._run, name="LogWriterThread")
        self.worker.start()
        self._create_tracks()
        return

    def stop(self):
        """
        Signals the log writing thread to finish, flush all remaining events,
        then close the json file.
        """
        if self.file_ptr is None:
            return

        self.stop_event.set()
        if self.worker is not None:
            self.worker.join()

        self.worker = None
        self.file_ptr = None

        return

    def record(self, log_event: dict, level: Level = Level.EVENT):
        """
        Other modules call this method to record a log event.
        This method is thread-safe.
        """

        if self.on and level.value <= self.level.value:
            self.log_queue.put(log_event)

        return

    def get_trace_log(self, trace) -> (dict[str, Any], dict[str, Any]):
        nodes = {"nodes": []}
        tensors = {"tensors": []}

        for node in trace.node_map.values():
            nodes["nodes"].append({
                "id": node.id,
                "name": node.name,
                "control_deps": {
                    "parent_nodes": node.parent_nodes,
                    "children_nodes": node.children_nodes
                },
                "data_deps": {
                    "input_tensors": node.input_tensors,
                    "output_tensors": node.output_tensors
                }
            })

        for tensor in trace.tensor_map.values():
            tensor_type = tensor.args["tensor_type"] if "tensor_type" in tensor.args else ""
            tensors["tensors"].append({
                "id": tensor.id,
                "name": tensor.name,
                "tensor_type": tensor_type,
                "size_KB": 4 * tensor.num_pages
            })

        return nodes, tensors

    @staticmethod
    def track(track: TrackID, name: str):
        return {
            "ph": "M",
            "name": "process_name",
            "pid": track.value,
            "args": {
                "name": name
            }
        }

    @staticmethod
    def subtrack(track: TrackID, obj_id: int, name: str):
        return {
            "ph": "M",
            "name": "thread_name",
            "pid": track.value,
            "tid": obj_id,
            "args": {
                "name": name
            }
        }

    @staticmethod
    def engine(obj_id: int, title: str, timestamp: float, args: dict[str, Any] | None = None):
        return {
            "pid": TrackID.Engine.value,
            "tid": obj_id,
            "cat": "Engine",
            "name": title,
            "ph": "i",
            "ts": timestamp,
            "s": "t",
            "args": args if args is not None else {}
        }

    @staticmethod
    def event_instant(obj_id: int, title: str, timestamp: float, args: dict[str, Any] | None = None):
        return {
            "pid": TrackID.Event.value,
            "tid": obj_id,
            "cat": "Event",
            "name": title,
            "ph": "i",
            "ts": timestamp,
            "s": "t",
            "args": args if args is not None else {}
        }

    @staticmethod
    def event_begin(obj_id: int, title: str, timestamp: float, args: dict[str, Any] | None = None):
        return {
            "pid": TrackID.Event.value,
            "tid": obj_id,
            "cat": "Event",
            "name": title,
            "ph": "B",
            "ts": timestamp,
            "args": args if args is not None else {}
        }

    @staticmethod
    def event_end(obj_id: int, title: str, timestamp: float, args: dict[str, Any] | None = None):
        return {
            "pid": TrackID.Event.value,
            "tid": obj_id,
            "cat": "Event",
            "name": title,
            "ph": "E",
            "ts": timestamp,
            "args": args if args is not None else {}
        }

    @staticmethod
    def event_complete(obj_id: int, title: str, timestamp_start: float, duration: float, args: dict[str, Any] | None = None):
        return {
            "pid": TrackID.Event.value,
            "tid": obj_id,
            "cat": "Event",
            "name": title,
            "ph": "X",
            "ts": timestamp_start,
            "dur": duration,
            "args": args if args is not None else {}
        }

    @staticmethod
    def counter(obj_id: int, title: str, timestamp: float, counters: dict[str, Any] | None = None):
        return {
            "pid": TrackID.Counter.value,
            "tid": obj_id,
            "cat": "Counter",
            "name": title,
            "ph": "C",
            "ts": timestamp,
            "args": counters if counters is not None else {}
        }

    @staticmethod
    def state(obj_id: int, title: str, timestamp: float, states: dict[str, Any] | None = None):
        return {
            "pid": TrackID.State.value,
            "tid": obj_id,
            "cat": "State",
            "name": title,
            "ph": "O",
            "ts": timestamp,
            "id": obj_id,
            "args": {
                "snapshot": states if states is not None else {}
            }
        }
