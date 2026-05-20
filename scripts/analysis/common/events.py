"""Chrome-trace event loading and typed parsers.

The on-disk shape of the events we care about is set by
`sim.core.job.logging.{compute,transfer}_logging.end_log`:

  COMPUTE_JOB  args = { Hardware: {id, name},
                        Payload:  {id, name, work_total},
                        Lifecycle:{timestamp_queued, _at_head, _begin, _end} }
  TRANSFER_JOB args = { Hardware: {src: {id, name}, dest: {id, name}},
                        Payload:  {size_KB, transfer_KBps, batch: [{tensor_id}]},
                        Lifecycle:{timestamp_queued, _at_head, _begin, _end} }

Events with `ts < RUNTIME_STAGE_START.ts` are dropped — matching the
convention established by `parse_stall_time.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


TRACK_ENGINE = 0
TRACK_EVENT = 1


def load_events(path: Path | str) -> list[dict]:
    """Load Chrome-trace events from `path`, tolerating a mid-flush truncation."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        patched = text.rstrip().rstrip(",")
        if not patched.endswith("]}"):
            patched += "\n]}"
        data = json.loads(patched)
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return list(data)


def find_runtime_start(events: list[dict]) -> float:
    for ev in events:
        if (
            ev.get("pid") == TRACK_ENGINE
            and ev.get("ph") == "i"
            and ev.get("name") == "RUNTIME_STAGE_START"
        ):
            return float(ev.get("ts", 0.0))
    raise RuntimeError("RUNTIME_STAGE_START event not found in event log")


@dataclass(slots=True)
class ComputeJob:
    ts_us: float
    end_us: float
    dur_us: float
    name: str
    node_id: int | None
    hw_name: str
    queued_us: float
    at_head_us: float
    begin_us: float


@dataclass(slots=True)
class TransferJob:
    ts_us: float
    end_us: float
    dur_us: float
    src_name: str
    dest_name: str
    size_KB: float
    rate_KBps: float
    tensor_ids: list[int]
    queued_us: float
    at_head_us: float
    begin_us: float


def parse_compute_jobs(
    events: list[dict], hw_name: str | None, t_start: float
) -> list[ComputeJob]:
    out: list[ComputeJob] = []
    for ev in events:
        if ev.get("pid") != TRACK_EVENT or ev.get("ph") != "X":
            continue
        if not str(ev.get("name", "")).startswith("COMPUTE_JOB"):
            continue
        args = ev.get("args") or {}
        hw = (args.get("Hardware") or {}).get("name")
        if hw_name is not None and hw != hw_name:
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        if dur <= 0.0 or ts < t_start:
            continue
        payload = args.get("Payload") or {}
        lc = args.get("Lifecycle") or {}
        out.append(
            ComputeJob(
                ts_us=ts,
                end_us=ts + dur,
                dur_us=dur,
                name=str(payload.get("name", "")),
                node_id=payload.get("id"),
                hw_name=str(hw) if hw is not None else "",
                queued_us=float(lc.get("timestamp_queued", ts)),
                at_head_us=float(lc.get("timestamp_at_head", ts)),
                begin_us=float(lc.get("timestamp_begin", ts)),
            )
        )
    out.sort(key=lambda j: j.ts_us)
    return out


def parse_transfer_jobs(
    events: list[dict], dest_name: str | None, t_start: float
) -> list[TransferJob]:
    out: list[TransferJob] = []
    for ev in events:
        if ev.get("pid") != TRACK_EVENT or ev.get("ph") != "X":
            continue
        if not str(ev.get("name", "")).startswith("TRANSFER_JOB"):
            continue
        args = ev.get("args") or {}
        hw_args = args.get("Hardware") or {}
        dest = (hw_args.get("dest") or {}).get("name")
        if dest_name is not None and dest != dest_name:
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        if dur <= 0.0 or ts < t_start:
            continue
        src = (hw_args.get("src") or {}).get("name", "")
        payload = args.get("Payload") or {}
        lc = args.get("Lifecycle") or {}
        tensor_ids = [b.get("tensor_id") for b in (payload.get("batch") or [])]
        rate = payload.get("transfer_KBps", 0.0)
        try:
            rate_f = float(rate)
        except (TypeError, ValueError):
            rate_f = 0.0
        out.append(
            TransferJob(
                ts_us=ts,
                end_us=ts + dur,
                dur_us=dur,
                src_name=str(src),
                dest_name=str(dest or ""),
                size_KB=float(payload.get("size_KB", 0.0)),
                rate_KBps=rate_f,
                tensor_ids=tensor_ids,
                queued_us=float(lc.get("timestamp_queued", ts)),
                at_head_us=float(lc.get("timestamp_at_head", ts)),
                begin_us=float(lc.get("timestamp_begin", ts)),
            )
        )
    out.sort(key=lambda j: j.ts_us)
    return out


def module_key(name: str, depth: int) -> str:
    """First `depth` dot-separated components of `name` (LLM-trace module path)."""
    if not name:
        return "<unknown>"
    parts = name.split(".")
    return ".".join(parts[:depth]) if parts else "<unknown>"
