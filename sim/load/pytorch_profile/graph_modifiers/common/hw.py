"""Hardware parameters parser for weight streaming schedulers.

Reads the same ``hw_pcie4.json`` format used by the legacy llamasim scripts
so schedulers can stay agnostic of the source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HwParams:
    """PCIe / SSD / VRAM bandwidth + latency parameters.

    All bandwidths are in bytes/ns (matching ProblemInstance in the old
    graph_modifiers). Latencies are in ns. ``vram_cap_mb`` is MB (0 = no
    cap). ``cpu_per_launch_ns`` inflates per-iter wall time by this many
    ns per kernel launch to model CPU dispatch overhead.

    ``profiler_overhead_per_event_ns`` is a calibration constant that
    subtracts fixed instrumentation cost from every CPU event in the
    trace. The Kineto/profile wall clock is inflated by
    ``record_function`` per-event cost; this constant strips it so the
    simulator's iter length approximates a non-profiled run. Calibrate
    once per (PyTorch version × machine) by comparing profiled vs
    non-profiled iter wall:

        k = (wall_profiled_ns - wall_real_ns) / num_cpu_events

    Leave at 0 to preserve legacy behavior (profiled wall = simulated
    wall). SDXL on PCIe 4.0 typically needs ~2500-3000 ns.
    """

    ssd_read_bw: float          # bytes/ns
    ssd_write_bw: float
    ssd_read_latency_ns: int
    ssd_write_latency_ns: int
    h2d_bw: float
    d2h_bw: float
    h2d_latency_ns: int
    d2h_latency_ns: int
    vram_cap_mb: float = 0.0
    cpu_per_launch_ns: int = 0
    profiler_overhead_per_event_ns: int = 0
    # Optional aggregate PCIe byte budget per iter (bytes). When set, LP
    # schedulers add Σ (1−z) · 2 · size  ≤  max_pcie_bytes_per_iter — the
    # round-trip bytes for evicted+reloaded tensors must fit. Captures the
    # H2D/D2H share-the-lane reality under duplex=False that per-rank
    # deadlines miss when individual tensors' deadlines are spread late
    # in the iter. None / 0 = no aggregate cap (legacy behavior).
    max_pcie_bytes_per_iter: float | None = None
    # Number of concurrent H2D copy streams the runtime will use.
    # Modern GPUs have multiple copy engines (RTX 4090: 2; H100/A100: 4-8).
    # With N streams, fires round-robin across streams, reducing
    # head-of-line blocking and amortizing per-fire launch overhead.
    # PCIe link bandwidth is still shared (single physical link), so
    # this helps when fires bunch but does NOT multiply total
    # throughput. Default 1 (single-stream, legacy behavior).
    n_h2d_streams: int = 1


def load_hw_params(path: str | Path) -> HwParams:
    with open(path) as f:
        cfg = json.load(f)
    ssd_bw = cfg["ssd_bandwidth_Bps"] / 1e9  # B/s -> B/ns
    h2d_Bps = cfg.get("pcie_h2d_bandwidth_Bps", 0)
    d2h_Bps = cfg.get("pcie_d2h_bandwidth_Bps", 0)
    return HwParams(
        ssd_read_bw=ssd_bw,
        ssd_write_bw=ssd_bw,
        ssd_read_latency_ns=int(cfg.get("ssd_read_latency_ns", 0)),
        ssd_write_latency_ns=int(cfg.get("ssd_write_latency_ns", 0)),
        h2d_bw=h2d_Bps / 1e9 if h2d_Bps else 0.0,
        d2h_bw=d2h_Bps / 1e9 if d2h_Bps else 0.0,
        h2d_latency_ns=int(cfg.get("pcie_h2d_latency_ns", 0)),
        d2h_latency_ns=int(cfg.get("pcie_d2h_latency_ns", 0)),
        vram_cap_mb=float(cfg.get("vram_cap_mb", 0)),
        cpu_per_launch_ns=int(cfg.get("cpu_per_launch_ns", 0)),
        profiler_overhead_per_event_ns=int(cfg.get("profiler_overhead_per_event_ns", 0)),
        max_pcie_bytes_per_iter=(
            float(cfg["max_pcie_bytes_per_iter"])
            if cfg.get("max_pcie_bytes_per_iter") else None
        ),
        n_h2d_streams=int(cfg.get("n_h2d_streams", 1)),
    )


def effective_h2d_bw(hw: HwParams) -> float:
    """Effective per-pipeline H2D bandwidth seen by the LP.

    With ``n_h2d_streams`` concurrent copy streams, multiple transfers
    can be in flight simultaneously (one per stream). The PCIe link
    itself is shared, but the EFFECTIVE throughput from the scheduler's
    perspective rises because head-of-line blocking is reduced and the
    link can be kept busier.  Modeled as a linear scaling: streams ×
    per-stream bandwidth.  This is an APPROXIMATION — once total bytes
    saturate the link, additional streams give no further benefit.
    Schedulers should use this helper instead of ``hw.h2d_bw`` directly
    whenever computing per-tensor h2d_dur or per-iter byte budgets.
    """
    n = max(1, int(getattr(hw, "n_h2d_streams", 1)))
    return float(hw.h2d_bw) * float(n)
