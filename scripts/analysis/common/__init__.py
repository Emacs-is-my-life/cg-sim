"""Shared helpers for `scripts/analysis/*.py`.

Split by feature:
  * events    — Chrome-trace event loading, ComputeJob/TransferJob parsers
  * intervals — interval-set math and percentile helper
  * io        — structured side-output (`--out DIR`, meta.json, CSVs)

Top-level re-exports cover the common usage; reach for a submodule if
you only need one thing.
"""
from .events import (
    ClaimJob,
    ComputeJob,
    NodeMeta,
    TensorMeta,
    TransferJob,
    find_runtime_start,
    load_events,
    module_key,
    parse_claim_jobs,
    parse_compute_jobs,
    parse_nodes,
    parse_sim_config,
    parse_tensors,
    parse_transfer_jobs,
)
from .intervals import merge_intervals, percentile, union_length
from .io import (
    default_out_dir,
    ensure_out_dir,
    parse_out_flag,
    write_meta,
    write_table,
)

__all__ = [
    "ClaimJob",
    "ComputeJob",
    "NodeMeta",
    "TensorMeta",
    "TransferJob",
    "find_runtime_start",
    "load_events",
    "module_key",
    "parse_claim_jobs",
    "parse_compute_jobs",
    "parse_nodes",
    "parse_sim_config",
    "parse_tensors",
    "parse_transfer_jobs",
    "merge_intervals",
    "percentile",
    "union_length",
    "default_out_dir",
    "ensure_out_dir",
    "parse_out_flag",
    "write_meta",
    "write_table",
]
