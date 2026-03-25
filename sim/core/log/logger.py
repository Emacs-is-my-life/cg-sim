import queue
import time
import threading
from pathlib import Path
import orjson
from enum import Enum


class TrackID(Enum):
    Debug = 0
    Event = 1
    Counter = 2
    State = 3


class Logger:
    """
    Logger collects log from modules(core, compute, mm, disk, ...),
    then periodically flush them to disk.
    """

    def __init__(self, args: dict, flush_period: float = 0.5):
        # Set log output file
        output_path = Path("./sim_run_result.json")
        if "output_path" in args:
            p = Path(args["output_path"])
            if not p.is_absolute():
                p = Path(args["input_path"]).parent / p

            if p.suffix:
                p.parent.mkdir(parents=True, exist_ok=True)
            else:
                p.mkdir(parents=True, exist_ok=True)
                p = p / Path("sim_run_result.json")
            output_path = p
        self.output_path = output_path

        # Set log level
        """
        log_level:
        - 0: Log Debug, Event
        - 1: + Log counter states
        - 2: + Log all states
        """
        log_level = 0
        if "log_level" in args:
            log_level = int(args["log_level"])
        self.log_level = log_level

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
        self.file_ptr = open(self.output_path, "w")
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
            parts.append(orjson.dumps(log_event).decode())

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
        self._close_file()
        return

    def _create_tracks(self):
        self.record(self.format_track(TrackID.Debug, "Debug"))
        self.record(self.format_track(TrackID.Event, "Event"))
        self.record(self.format_track(TrackID.Counter, "Counter"))
        self.record(self.format_track(TrackID.State, "State"))
        return

    def start(self):
        """
        Starts the log writer module, running in a separate thread
        """
        self._open_file()
        self.worker = threading.Thread(target=self._run, name="LogWriterThread")
        self.worker.start()
        self._create_tracks()

    def stop(self):
        """
        Signals the log writing thread to finish, flush all remaining events,
        then close the json file.
        """
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join()

    def record(self, log_event: dict, log_level: int = 0):
        """
        Other modules call this method to record a log event.
        This method is thread-safe.
        """

        if log_level <= self.log_level:
            self.log_queue.put(log_event)

    @staticmethod
    def format_track(self, track: TrackID, name: str):
        return {
            "ph": "M",
            "name": "process_name",
            "pid": track.value,
            "args": {
                "name": name
            }
        }

    @staticmethod
    def format_subtrack(self, track: TrackID, tid: int, name: str):
        return {
            "ph": "M",
            "name": "thread_name",
            "pid": track.value,
            "tid": tid,
            "args": {
                "name": name
            }
        }
