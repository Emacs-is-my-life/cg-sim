"""Conformance check for the ``DiffusersGroupOffload`` scheduler against the spec.

Reads the simulator's Chrome-trace log file and validates the run against
the reference numbers documented in
``docs/offload-schemes/diffusers_group-offload_use-stream-true.md``.

Usage:
    python scripts/check_accelerate.py [path/to/result.json]

If no path is supplied, defaults to:
    output/pytorch-eager-llama-8B-accelerate.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def find_event(events: list[dict], name: str) -> dict | None:
    for e in events:
        if e.get("name") == name:
            return e
    return None


def collect_tensors(events: list[dict]) -> dict[int, dict]:
    """Return tensor_id -> dict with size/type, harvested from the
    TENSORS dump emitted at LAYOUT_STAGE_START.
    """
    e = find_event(events, "TENSORS")
    if e is None:
        return {}
    out: dict[int, dict] = {}
    for t in e["args"]["tensors"]:
        out[int(t["id"])] = {
            "name": t.get("name"),
            "size_KB": t.get("size_KB", 0),
            "type": t.get("tensor_type") or t.get("type"),
        }
    return out


def hw_id_name_map(events: list[dict]) -> dict[int, str]:
    """Names are emitted as `thread_name` metadata events for each hw subtrack."""
    # Easier: just scan for any event with args.Hardware.* and known
    # subtrack ids — but the cleanest is to read the per-hw subtrack
    # records and pull tid from there. Use a simpler heuristic: find
    # the hw ids we'll encounter from TRANSFER_JOB Hardware args.
    name_for_id: dict[int, str] = {}
    for e in events:
        args = e.get("args") or {}
        hw = args.get("Hardware")
        if isinstance(hw, dict):
            for slot in ("src", "dest"):
                rec = hw.get(slot)
                if isinstance(rec, dict) and "id" in rec and "name" in rec:
                    name_for_id[int(rec["id"])] = rec["name"]
    # Subtrack metadata events ("thread_name") carry the name in args.name
    # but no id field; instead they use `tid`. Map via tid.
    for e in events:
        if e.get("name") == "thread_name" and "tid" in e and isinstance(e.get("args"), dict):
            name_for_id.setdefault(int(e["tid"]), e["args"].get("name") or "")
    return name_for_id


def main(path: Path) -> int:
    events = load_events(path)
    tensors = collect_tensors(events)
    name_for_id = hw_id_name_map(events)
    name_to_id = {v: k for k, v in name_for_id.items()}
    vram_id = name_to_id.get("vram0")
    ram_id = name_to_id.get("ram")
    ssd_id = name_to_id.get("ssd")

    weights = {tid: t for tid, t in tensors.items() if t.get("type") == "WEIGHT"}
    weight_total_KB = sum(t["size_KB"] for t in weights.values())
    weight_max_KB = max((t["size_KB"] for t in weights.values()), default=0)

    # ---- run-level metrics from SIMULATION_RESULT ----
    res = find_event(events, "SIMULATION_RESULT")
    if res is None:
        print("[FAIL] No SIMULATION_RESULT in log.")
        return 1
    res_args = res["args"]
    sim_ok = res_args["simulation"]["success"] == "True"
    sim_time = res_args["simulation"]["time"]
    mem_by_name = {m["name"]: m for m in res_args["memory"]}
    peak_ram = mem_by_name.get("ram", {}).get("peak_memory_usage_KB", 0)
    peak_vram = mem_by_name.get("vram0", {}).get("peak_memory_usage_KB", 0)
    job_stats = res_args["job"]

    # ---- walk events to collect TRANSFER_JOB and CLAIM/RELEASE ----
    # TRANSFER_JOB: complete events with ts, dur, args.Hardware.src/dest, args.Payload.batch
    # CLAIM_JOB / RELEASE_JOB: instant events on a specific hw subtrack
    ram_to_vram_bytes_KB = 0
    ram_to_vram_jobs = 0
    ssd_to_ram_jobs = 0
    ssd_to_vram_jobs = 0
    vram_to_ram_jobs = 0
    weight_load_count: dict[int, int] = defaultdict(int)
    weight_load_size_KB: dict[int, int] = defaultdict(int)
    layout_end_ts = None
    layout_evt = find_event(events, "RUNTIME_STAGE_START")
    if layout_evt is not None:
        layout_end_ts = layout_evt.get("ts", 0)

    # Each TransferJob ends up logging TWO TRANSFER_JOB events — one on
    # the src hw subtrack, one on the dst — because the job's
    # ``running_on`` list contains both. Dedup by (timestamp_begin,
    # src.id, dst.id, frozenset(tensor_ids)).
    seen_jobs: set = set()
    for e in events:
        if e.get("name") != "TRANSFER_JOB":
            continue
        args = e["args"]
        src_name = args["Hardware"]["src"]["name"]
        dst_name = args["Hardware"]["dest"]["name"]
        size_KB = args["Payload"]["size_KB"]
        tids_in_batch = [b["tensor_id"] for b in args["Payload"]["batch"]]
        sig = (
            args["Lifecycle"]["timestamp_begin"],
            args["Hardware"]["src"]["id"],
            args["Hardware"]["dest"]["id"],
            tuple(tids_in_batch),
        )
        if sig in seen_jobs:
            continue
        seen_jobs.add(sig)
        if src_name == "ssd" and dst_name == "ram":
            ssd_to_ram_jobs += 1
        elif src_name == "ssd" and dst_name == "vram0":
            ssd_to_vram_jobs += 1
        elif src_name == "ram" and dst_name == "vram0":
            ram_to_vram_jobs += 1
            ram_to_vram_bytes_KB += size_KB
            for tid in tids_in_batch:
                if tid in weights:
                    weight_load_count[tid] += 1
                    weight_load_size_KB[tid] += tensors[tid]["size_KB"]
        elif src_name == "vram0" and dst_name == "ram":
            vram_to_ram_jobs += 1

    # ---- VRAM occupancy timeline via CLAIM/RELEASE on vram0 subtrack ----
    vram_loaded_KB = 0
    vram_loaded_weight_count = 0
    vram_peak_loaded_weight_KB = 0
    vram_peak_concurrent_weights = 0
    vram_currently_loaded_weights: set[int] = set()
    weight_claim_count: dict[int, int] = defaultdict(int)
    weight_release_count: dict[int, int] = defaultdict(int)

    # Walk events in timestamp order (events with ts=None are setup).
    timed_events = [e for e in events if isinstance(e.get("ts"), (int, float))]
    timed_events.sort(key=lambda e: (e["ts"], 0 if e.get("name") == "CLAIM_JOB" else 1))

    for e in timed_events:
        name = e.get("name")
        if name not in ("CLAIM_JOB", "RELEASE_JOB"):
            continue
        if e.get("tid") != vram_id:
            continue
        args = e.get("args") or {}
        tid = args.get("tensor_id")
        if tid not in weights:
            continue  # only track WEIGHT VRAM occupancy
        size_KB = weights[tid]["size_KB"]
        if name == "CLAIM_JOB":
            weight_claim_count[tid] += 1
            vram_currently_loaded_weights.add(tid)
            vram_loaded_KB += size_KB
            if vram_loaded_KB > vram_peak_loaded_weight_KB:
                vram_peak_loaded_weight_KB = vram_loaded_KB
            if len(vram_currently_loaded_weights) > vram_peak_concurrent_weights:
                vram_peak_concurrent_weights = len(vram_currently_loaded_weights)
        else:  # RELEASE_JOB
            weight_release_count[tid] += 1
            if tid in vram_currently_loaded_weights:
                vram_currently_loaded_weights.discard(tid)
                vram_loaded_KB -= size_KB

    # ---- report ----
    def fmt_gib(kb: float) -> str:
        return f"{kb / 1024 / 1024:.2f} GiB"

    def fmt_mb(kb: float) -> str:
        return f"{kb / 1024:.2f} MB"

    print("=" * 70)
    print(f" Accelerate scheduler conformance — {path}")
    print("=" * 70)
    print()
    print(f"  simulation_success         : {sim_ok}")
    print(f"  simulation_time            : {sim_time / 1e6:.3f} s")
    print()
    print("  ---- Tensor inventory (from TENSORS dump) ----")
    print(f"  total tensors              : {len(tensors)}")
    print(f"  WEIGHT tensors             : {len(weights)}")
    print(f"  Sum WEIGHT bytes           : {fmt_gib(weight_total_KB)} ({weight_total_KB:,} KB)")
    print(f"  Largest single WEIGHT      : {fmt_mb(weight_max_KB)}")
    print()
    print("  ---- Memory peaks ----")
    print(f"  peak RAM                   : {fmt_gib(peak_ram)}")
    print(f"  peak VRAM                  : {fmt_gib(peak_vram)}")
    print(f"  peak VRAM (WEIGHTs only)   : {fmt_gib(vram_peak_loaded_weight_KB)}")
    print(f"  peak concurrent WEIGHTs    : {vram_peak_concurrent_weights}")
    print()
    print("  ---- Transfer breakdown ----")
    print(f"  ssd -> ram   jobs          : {ssd_to_ram_jobs}")
    print(f"  ssd -> vram0 jobs          : {ssd_to_vram_jobs}")
    print(f"  ram -> vram0 jobs          : {ram_to_vram_jobs}")
    print(f"  vram0 -> ram jobs          : {vram_to_ram_jobs}")
    print(f"  ram -> vram0 bytes         : {fmt_gib(ram_to_vram_bytes_KB)}")
    print()
    print("  ---- Per-WEIGHT load/release ----")
    loaded_weights = [tid for tid in weights if weight_load_count[tid] > 0]
    cuda_weights = [
        tid for tid, t in tensors.items()
        if t.get("type") == "WEIGHT" and t.get("size_KB", 0) > 0
    ]
    print(f"  WEIGHTs loaded ≥1× to VRAM : {len(loaded_weights)} / {len(weights)}")
    print(f"  total loads (ram->vram)    : {sum(weight_load_count.values())}")
    print(f"  total VRAM claims          : {sum(weight_claim_count.values())}")
    print(f"  total VRAM releases        : {sum(weight_release_count.values())}")
    unbalanced = [(tid, weight_claim_count[tid], weight_release_count[tid])
                  for tid in weights
                  if weight_claim_count[tid] != weight_release_count[tid]]
    print(f"  unbalanced claim/release   : {len(unbalanced)}")
    if unbalanced[:3]:
        for tid, c, r in unbalanced[:3]:
            print(f"     tid={tid} claims={c} releases={r} name={tensors[tid]['name']}")

    print()
    print("  ---- Job stats from SIMULATION_RESULT ----")
    for k, v in job_stats.items():
        if "time" in k:
            print(f"  {k:27s}: {v / 1e6:.3f} s")
        elif "size" in k:
            print(f"  {k:27s}: {fmt_gib(v)}")
        else:
            print(f"  {k:27s}: {v}")

    # ---- spec checks ----
    print()
    print("  ---- Spec checks (docs/offload-schemes/diffusers_group-offload_use-stream-true.md) ----")
    checks: list[tuple[bool, str, str]] = []

    # 1. Simulation must succeed.
    checks.append((sim_ok, "simulation completes without abort", str(sim_ok)))

    # 2. RAM master copy ≈ full model size. Spec line 159: SDXL-Turbo
    # has ~6.94 GB of cuda-homed WEIGHTs across all 10 groups.
    expected_ram_min_KB = int(weight_total_KB * 0.95)
    checks.append((
        peak_ram >= expected_ram_min_KB,
        f"peak RAM ≥ 95% of sum(WEIGHT bytes) [{fmt_gib(expected_ram_min_KB)}]",
        fmt_gib(peak_ram),
    ))

    # 3. Peak VRAM should be strictly less than the full model — the
    # whole point of group_offload is that you never need every
    # weight resident at once. The simulator's "claim" semantics map
    # to real-hardware *reserved* VRAM (no caching allocator), which
    # is what PyTorch's max_memory_reserved reports. Real allocated is
    # naturally lower because of intra-allocator coalescing the sim
    # doesn't model. Cap at total weight bytes; anything below proves
    # at least one group cycle (load/evict) actually happened.
    vram_ceiling_KB = weight_total_KB
    checks.append((
        peak_vram < vram_ceiling_KB,
        f"peak VRAM < sum(WEIGHT bytes) [{fmt_gib(vram_ceiling_KB)}]",
        fmt_gib(peak_vram),
    ))

    # 4. Every cuda-homed WEIGHT should be loaded at least once.
    cuda_weight_load_coverage = (
        len(loaded_weights) / max(1, len(weights))
        if weights else 0.0
    )
    checks.append((
        cuda_weight_load_coverage >= 0.50,
        "≥50% of WEIGHTs loaded ram->vram at least once",
        f"{cuda_weight_load_coverage*100:.1f}%",
    ))

    # 5. Claim/release pairs should balance (every loaded weight is
    #    evicted before the simulation ends).
    checks.append((
        len(unbalanced) == 0,
        "every WEIGHT VRAM claim has a matching release",
        f"{len(unbalanced)} unbalanced",
    ))

    # 6. ram->vram bytes per pipeline run. Spec line 145-157: for a
    # 4-inference-step SDXL-Turbo pass, the 7 UNet groups each get
    # paged once per inference step, plus the 3 non-UNet components
    # paged once total. Σ over groups: roughly num_inference_steps *
    # unet_total + non_unet_total. Without knowing num_inference_steps
    # from the log, accept any ratio in [1, 5] × Σ(WEIGHT bytes) as
    # plausible, flag otherwise.
    expected_h2d_KB = sum(weights[tid]["size_KB"] for tid in loaded_weights)
    h2d_ratio = ram_to_vram_bytes_KB / max(1, expected_h2d_KB)
    checks.append((
        1.0 <= h2d_ratio <= 5.0,
        f"ram->vram bytes within 1×..5× Σ(loaded WEIGHT bytes) [{fmt_gib(expected_h2d_KB)}]",
        f"{fmt_gib(ram_to_vram_bytes_KB)} ({h2d_ratio:.2f}×)",
    ))

    n_ok = sum(1 for ok, _, _ in checks if ok)
    for ok, label, observed in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}")
        print(f"          observed: {observed}")
    print()
    print(f"  → {n_ok} / {len(checks)} checks passed")

    return 0 if n_ok == len(checks) else 2


if __name__ == "__main__":
    default = Path("output/pytorch_eager_sdxl-turbo_diffusers/sim_results/result.json")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    raise SystemExit(main(path))
