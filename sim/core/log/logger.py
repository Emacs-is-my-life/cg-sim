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

        # Prepare another thread for background flushing
        self.stop_event = threading.Event()
        self.worker = None
        return

    def _open_file(self):
        self.file_ptr = open(self.output_path, "w")
        self.file_ptr.write('{"traceEvents": [\n')
        self.file_ptr.flush()

    def _close_file(self):
        self.file_ptr.flush()
        # Take care of the last trailing ",\n"
        pos = self.file_ptr.tell()
        read_back = 2
        self.file_ptr.seek(max(pos - read_back, 0))
        tail = self.file_ptr.read(read_back)

        if tail == ",\n":
            self.file_ptr.seek(max(pos - read_back, 0))
            self.file_ptr.truncate()

        # Append the JSON structure closing, then close the file
        self.file_ptr.write("\n]}\n")
        self.file_ptr.close()

    def _flush(self):
        """Drain the log_queue and writes events to the json file on disk."""
        arr_log_event = []
        while True:
            try:
                log_event = self.log_queue.get_nowait()
                arr_log_event.append(log_event)
            except queue.Empty:
                break

        if arr_log_event:
            # Write each log event to the file
            text_to_write = ""
            for log_event in arr_log_event:
                log_text = orjson.dumps(log_event).decode() + ",\n"
                text_to_write += log_text

            self.file_ptr.write(text_to_write)
            self.file_ptr.flush()

    def _run(self):
        """
        Internal loop run by the log writer thread, periodically flushing log events to the disk.
        """
        try:
            while not self.stop_event.is_set():
                self._flush()
                time.sleep(self.flush_period)
            # Final one flush when the stop signal is delivered
            self._flush()
        except Exception as e:  # TODO exception format
            # Simulator exploded somehow
            print(f"[Logger] Encountered an exception: {e}")
            self._flush()
        finally:
            self._close_file()

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

    def format_track(self, pid: TrackID, name: str):
        return {
            "ph": "M",
            "name": "process_name",
            "pid": pid,
            "args": {
                "name": name
            }
        }

    def format_subtrack(pid: TrackID, tid: int, name: str):
        return {
            "ph": "M",
            "name": "thread_name",
            "pid": pid,
            "tid": tid,
            "args": {
                "name": name
            }
        }
