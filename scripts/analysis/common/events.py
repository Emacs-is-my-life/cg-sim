"""Chrome-trace event loading and typed parsers.

The on-disk shape of the events we care about is set by
`sim.core.job.logging.{compute,transfer}_logging.end_log`:

  COMPUTE_JOB  args = { Hardware: {id, name},
                        Payload:  {id, name, work_total},
                        Lifecycle:{timestamp_queued, _at_head, _begin, _end} }
  TRANSFER_JOB args = { Hardware: {src: {id, name}, dest: {id, name}},
                        Payload:  {size_KB, transfer_KBps, batch: [{tensor_id}]},
                        Lifecycle:{timestamp_queued, _at_head, _begin, _end} }

Events with `ts < RUNTIME_STAGE_START.ts` are dropped — only runtime
events should figure into analysis.
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
class ClaimJob:
    """A CLAIM_JOB instant event — a memory region reserved for a tensor.

    Emitted by ``sim.core.job.logging.claim_logging.begin_log`` for every
    ``sys.claim`` call. The event's ``tid`` carries the memory hw's id,
    so analyses can filter by destination memory.
    """
    ts_us: float
    tensor_id: int
    page_idx_start: int
    num_pages: int
    hw_id: int


def parse_claim_jobs(
    events: list[dict], hw_id: int | None
) -> list[ClaimJob]:
    """Read CLAIM_JOB instant events on ``hw_id`` (or all if None).

    No t_start filtering: layout-time claims (ts == t_start) tell
    analyses that a tensor was placed during the layout stage, which
    is the per-output analog of an early-arriving input.
    """
    out: list[ClaimJob] = []
    for ev in events:
        if ev.get("pid") != TRACK_EVENT or ev.get("ph") != "i":
            continue
        if ev.get("name") != "CLAIM_JOB":
            continue
        tid = ev.get("tid")
        if hw_id is not None and tid != hw_id:
            continue
        args = ev.get("args") or {}
        out.append(
            ClaimJob(
                ts_us=float(ev.get("ts", 0.0)),
                tensor_id=int(args.get("tensor_id", -1)),
                page_idx_start=int(args.get("page_idx_start", 0)),
                num_pages=int(args.get("num_pages", 0)),
                hw_id=int(tid) if tid is not None else -1,
            )
        )
    out.sort(key=lambda j: j.ts_us)
    return out


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


@dataclass(slots=True)
class NodeMeta:
    node_id: int
    name: str
    input_tensors: list[int]
    output_tensors: list[int]
    parent_nodes: list[int]


@dataclass(slots=True)
class TensorMeta:
    tensor_id: int
    name: str
    tensor_type: str
    size_KB: float


def parse_nodes(events: list[dict]) -> dict[int, NodeMeta]:
    """Read the single ``NODES`` instant event into ``{node_id: NodeMeta}``.

    NODES is emitted once during the layout/runtime hand-off and carries
    the full compute-graph topology — every analysis that needs per-node
    tensor I/O (heatmaps, lifetime analyses) reads it from here rather
    than re-parsing the trace file.
    """
    for ev in events:
        if (
            ev.get("pid") == TRACK_ENGINE
            and ev.get("ph") == "i"
            and ev.get("name") == "NODES"
        ):
            raw = (ev.get("args") or {}).get("nodes") or []
            out: dict[int, NodeMeta] = {}
            for n in raw:
                nid = n.get("id")
                if nid is None:
                    continue
                dd = n.get("data_deps") or {}
                cd = n.get("control_deps") or {}
                out[int(nid)] = NodeMeta(
                    node_id=int(nid),
                    name=str(n.get("name", "")),
                    input_tensors=[int(t) for t in (dd.get("input_tensors") or [])],
                    output_tensors=[int(t) for t in (dd.get("output_tensors") or [])],
                    parent_nodes=[int(p) for p in (cd.get("parent_nodes") or [])],
                )
            return out
    return {}


def parse_sim_config(events: list[dict]) -> dict:
    """Read the single ``SIM_CONFIG`` instant event's args.

    Emitted by ``sim.core.simulator.Simulator.__init__`` after all hw
    are constructed (so IDs exist). The returned dict has the shape
    ``{"config": <resolved Hydra cfg with cg-sim metadata>,
       "id_map": {<name>: <obj_id>}}``. ``config["hardware"]`` and
    ``config["scheduler"]`` mirror the input YAML; ``config["cg-sim"]``
    carries ``git_commit``, ``git_dirty``, and ``timestamp``.

    Returns ``{}`` if the event is absent (shouldn't happen for logs
    written by the current simulator — backward compat dropped per
    project decision).
    """
    for ev in events:
        if (
            ev.get("pid") == TRACK_ENGINE
            and ev.get("ph") == "i"
            and ev.get("name") == "SIM_CONFIG"
        ):
            return dict(ev.get("args") or {})
    return {}


def parse_tensors(events: list[dict]) -> dict[int, TensorMeta]:
    """Read the single ``TENSORS`` instant event into ``{tensor_id: TensorMeta}``."""
    for ev in events:
        if (
            ev.get("pid") == TRACK_ENGINE
            and ev.get("ph") == "i"
            and ev.get("name") == "TENSORS"
        ):
            raw = (ev.get("args") or {}).get("tensors") or []
            out: dict[int, TensorMeta] = {}
            for t in raw:
                tid = t.get("id")
                if tid is None:
                    continue
                out[int(tid)] = TensorMeta(
                    tensor_id=int(tid),
                    name=str(t.get("name", "")),
                    tensor_type=str(t.get("tensor_type", "")),
                    size_KB=float(t.get("size_KB", 0.0)),
                )
            return out
    return {}
