from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any

from sim.core.trace import Node, Tensor, TerminalNode, Trace, TraceLoader
from sim.core.trace.custom_dep import NodeDoneDep, MinTimestampDep
from sim.hw.storage.common import BaseStorage

from .utils import (
    normalize_device,
    parse_pytorch_dot,
    parse_float,
    parse_int,
    parse_shape,
    profile_id_sort_key,
    resolve_path,
    tensor_type_from_row,
)


class PytorchOffloadLoader(TraceLoader):
    """Trace loader for flattened PyTorch profiler runtime bundles."""

    @staticmethod
    def _num_pages(size_bytes: int) -> int:
        align_bytes = 64
        page_size_bytes = 4096
        tensor_aligned_size_bytes = ((size_bytes + align_bytes - 1) // align_bytes) * align_bytes
        return int(math.ceil(tensor_aligned_size_bytes / page_size_bytes))

    def _bundle_paths(self) -> tuple[Path, dict[str, Any]]:
        input_dir = Path(self.args["input_path"]).parent
        profile_dir_arg = self.args.get("profile_dir")
        profile_dir = resolve_path(profile_dir_arg, input_dir) if profile_dir_arg else input_dir
        manifest_path = resolve_path(
            self.args.get("bundle_manifest", "llama_bundle/manifest.json"),
            profile_dir,
        )

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        return manifest_path.parent, manifest

    @staticmethod
    def _read_rows_by_id(csv_path: Path, id_key: str) -> dict[str, dict[str, str]]:
        rows_by_id: dict[str, dict[str, str]] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_id = row[id_key]
                if row_id in rows_by_id:
                    raise Exception(f"[PytorchOffloadLoader] Duplicate row id {row_id} in {csv_path}.")
                rows_by_id[row_id] = row
        return rows_by_id

    def _handle_validation_failure(self, msg: str) -> None:
        if bool(self.args.get("strict_dot_validation", True)):
            raise Exception(f"[PytorchOffloadLoader] {msg}")
        print(f"[PytorchOffloadLoader] Warning: {msg}")
        return

    def _read_tensors(self, tensor_csv_path: Path) -> tuple[dict[int, Tensor], dict[str, int]]:
        tensor_map: dict[int, Tensor] = {}
        profile_to_tensor: dict[str, int] = {}

        with open(tensor_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._add_tensor_from_row(row, tensor_map, profile_to_tensor)

        return tensor_map, profile_to_tensor

    def _add_tensor_from_row(
        self,
        row: dict[str, str],
        tensor_map: dict[int, Tensor],
        profile_to_tensor: dict[str, int],
    ) -> int | None:
        size_bytes = int(row.get("tensor_size_bytes") or 0)
        if bool(self.args.get("skip_zero_byte_tensors", True)) and size_bytes <= 0:
            return None

        profile_tensor_id = row["tensor_node_id"]

        tensor_id = len(tensor_map)
        tensor_type = tensor_type_from_row(row)
        args = {
            "tensor_type": tensor_type,
            "device": normalize_device(row.get("device")),
            "dtype": row.get("dtype") or None,
            "shape": parse_shape(row.get("shape")),
            "producer_count": int(row.get("producer_count") or 0),
            "consumer_count": int(row.get("consumer_count") or 0),
            "profile_tensor_id": profile_tensor_id,
            "pytorch_tensor_id": parse_int(row.get("tensor_id")),
            "storage_id": parse_int(row.get("storage_id")),
            "storage_offset": parse_int(row.get("offset"), 0),
            "tensor_kind": row.get("tensor_kind"),
            "profile_tensor_aliases": [profile_tensor_id],
        }
        name = row.get("tensor_name") or profile_tensor_id
        tensor = Tensor(tensor_id, name, size_bytes, args)
        tensor_map[tensor_id] = tensor
        profile_to_tensor[profile_tensor_id] = tensor_id
        return tensor_id

    def _read_nodes(self, node_csv_path: Path) -> tuple[dict[int, Node], dict[str, int]]:
        node_map: dict[int, Node] = {}
        profile_to_node: dict[str, int] = {}

        with open(node_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = len(node_map)
                node = self._node_from_row(node_id, row)
                node_map[node_id] = node
                profile_to_node[row["node_id"]] = node_id

        return node_map, profile_to_node

    def _node_from_row(self, node_id: int, row: dict[str, str]) -> Node:
        profile_node_id = row["node_id"]
        duration_ns = parse_float(row.get("duration_ns"))
        runtime_role = row.get("runtime_role") or ""
        op_name = row.get("op_name") or ""

        # Subtract per-op probe-effect calibration if a table was loaded for
        # this trace. Clamped at 0 to avoid negative compute times when a
        # short call's duration falls below the calibrated overhead.
        probe_effect_ns = self._probe_effect_ns.get(op_name, 0) if self._probe_effect_ns else 0
        effective_duration_ns = max(0.0, duration_ns - probe_effect_ns)
        if probe_effect_ns and effective_duration_ns == 0.0 and duration_ns < probe_effect_ns:
            self._probe_clamp_count += 1

        compute_time_micros = (
            0.0
            if bool(self.args.get("zero_wait_nodes", True)) and runtime_role == "wait"
            else effective_duration_ns / 1_000
        )

        module_path = row.get("module_path") or None
        module_class = row.get("module_class") or None
        module_info = (
            {
                "module_path": module_path,
                "module_class": module_class,
                "module_size_bytes": parse_int(row.get("module_size_bytes")),
                "module_has_parameters": parse_int(row.get("module_has_parameters")),
                "module_has_buffers": parse_int(row.get("module_has_buffers")),
            }
            if (module_path or module_class)
            else None
        )

        args = {
            "step": int(row.get("step") or 0),
            "profile_node_id": profile_node_id,
            "node_n": parse_int(row.get("node_n")),
            "op_name": row.get("op_name"),
            "node_kind": row.get("node_kind"),
            "runtime_role": runtime_role,
            "device_type": row.get("device_type") or "CPU",
            "device_index": parse_int(row.get("device_index")),
            "thread_id": parse_int(row.get("thread_id")),
            "stream_id": parse_int(row.get("stream_id")),
            "resource_kind": row.get("resource_kind"),
            "resource_id": row.get("resource_id"),
            "start_ns": parse_int(row.get("start_ns")),
            "end_ns": parse_int(row.get("end_ns")),
            "correlation_id": parse_int(row.get("correlation_id")),
            "linked_correlation_id": parse_int(row.get("linked_correlation_id")),
            "rf_id": parse_int(row.get("rf_id")),
            "kernel_file": row.get("kernel_file") or None,
            "compiled_graph_id": parse_int(row.get("compiled_graph_id")),
            "compiled_launch_id": parse_int(row.get("compiled_launch_id")),
            "module": module_info,
        }
        name = row.get("node_name") or profile_node_id
        return Node(node_id, name, compute_time_micros, args)

    @staticmethod
    def _add_control_edge(node_map: dict[int, Node], parent_id: int, child_id: int) -> None:
        if parent_id == child_id:
            return
        node_map[parent_id].add_child_node(child_id)
        node_map[child_id].add_parent_node(parent_id)
        return

    @staticmethod
    def _is_start_gated_edge(
        node_map: dict[int, Node],
        parent_id: int,
        child_id: int,
        edge_kind: str,
    ) -> bool:
        """A kineto `submit` edge from a submit-role node into a
        gpu_runtime kernel models the cudaLaunchKernel→kernel async-
        enqueue boundary: the kernel can dispatch as soon as the
        launch begins; it does not need to wait for the CPU side's
        RecordFunction-wrapped duration to complete. These edges are
        carved out of the control graph entirely (no parent_nodes /
        children_nodes entry) and are routed through
        `trace.args["start_gated_edges"]` for the scheduler to gate
        on START rather than DONE. Other control edges (thread_order,
        stream_order, wait) stay in the control graph."""
        if edge_kind != "submit":
            return False
        if parent_id == child_id:
            return False
        parent = node_map.get(parent_id)
        child = node_map.get(child_id)
        if parent is None or child is None:
            return False
        if (parent.args.get("runtime_role") or "") != "submit":
            return False
        if (child.args.get("runtime_role") or "") != "gpu_runtime":
            return False
        return True

    # Runtime roles that correspond to pointer-passing CPU operations —
    # they launch / synchronize GPU work but do not read or write tensor
    # data on the CPU side. Any data_input / data_output edge landing on
    # one of these nodes is a profiler artifact (the tensor reference
    # appears in the kernel launch signature) and must not be materialized
    # as a simulated memory access.
    _POINTER_ONLY_ROLES = frozenset({"submit", "wait"})

    def _is_pointer_only(self, node: Node) -> bool:
        return (node.args.get("runtime_role") or "") in self._POINTER_ONLY_ROLES

    def _read_edges(
        self,
        edge_csv_path: Path,
        node_map: dict[int, Node],
        profile_to_node: dict[str, int],
        profile_to_tensor: dict[str, int],
    ) -> None:
        control_edge_kinds = {"thread_order", "stream_order", "submit", "wait"}

        with open(edge_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row["src_node_id"]
                dst = row["dst_node_id"]
                edge_kind = row["edge_kind"]

                if edge_kind == "data_input":
                    tensor_id = profile_to_tensor.get(src)
                    node_id = profile_to_node.get(dst)
                    if tensor_id is None or node_id is None:
                        continue
                    node = node_map[node_id]
                    if self._is_pointer_only(node):
                        continue
                    node.add_input_tensor(tensor_id)
                elif edge_kind == "data_output":
                    node_id = profile_to_node.get(src)
                    tensor_id = profile_to_tensor.get(dst)
                    if node_id is None or tensor_id is None:
                        continue
                    node = node_map[node_id]
                    if self._is_pointer_only(node):
                        continue
                    node.add_output_tensor(tensor_id)
                elif edge_kind in control_edge_kinds:
                    parent_id = profile_to_node.get(src)
                    child_id = profile_to_node.get(dst)
                    if parent_id is None or child_id is None:
                        continue
                    if self._is_start_gated_edge(node_map, parent_id, child_id, edge_kind):
                        self._start_gated_edges.append((parent_id, child_id))
                    else:
                        self._add_control_edge(node_map, parent_id, child_id)

        return

    @staticmethod
    def _edge_kind_from_vertices(src: str, dst: str, attrs: dict[str, str] | None = None) -> str | None:
        if src.startswith("t") and dst.startswith("k"):
            return "data_input"
        if src.startswith("k") and dst.startswith("t"):
            return "data_output"
        if src.startswith("k") and dst.startswith("k"):
            if attrs:
                return attrs.get("label")
            return None
        return None

    def _dot_paths(self, bundle_dir: Path, manifest: dict[str, Any]) -> list[Path]:
        dot_files = manifest.get("step_dot_files") or []
        if not dot_files:
            raise Exception("[PytorchOffloadLoader] manifest does not define step_dot_files.")
        return [resolve_path(dot_file, bundle_dir) for dot_file in dot_files]

    def _read_dot_bundle(
        self,
        bundle_dir: Path,
        manifest: dict[str, Any],
    ) -> tuple[dict[str, dict[str, str]], list[tuple[str, str, dict[str, str]]]]:
        vertices: dict[str, dict[str, str]] = {}
        edges: list[tuple[str, str, dict[str, str]]] = []

        for dot_path in self._dot_paths(bundle_dir, manifest):
            dot_vertices, dot_edges = parse_pytorch_dot(dot_path)
            for vertex_id, attrs in dot_vertices.items():
                if vertex_id in vertices:
                    self._handle_validation_failure(f"Duplicate DOT vertex {vertex_id} in {dot_path}.")
                    continue
                vertices[vertex_id] = attrs
            edges.extend(dot_edges)

        return vertices, edges

    def _validate_dot_vertices(
        self,
        dot_node_ids: list[str],
        dot_tensor_ids: list[str],
        node_rows: dict[str, dict[str, str]],
        tensor_rows: dict[str, dict[str, str]],
        manifest: dict[str, Any],
    ) -> None:
        if "node_count" in manifest and len(dot_node_ids) != int(manifest["node_count"]):
            self._handle_validation_failure(
                f"DOT node count {len(dot_node_ids)} != manifest node_count {manifest['node_count']}."
            )

        if "tensor_count" in manifest and len(dot_tensor_ids) != int(manifest["tensor_count"]):
            self._handle_validation_failure(
                f"DOT tensor count {len(dot_tensor_ids)} != manifest tensor_count {manifest['tensor_count']}."
            )

        missing_node_rows = sorted(set(dot_node_ids) - set(node_rows), key=profile_id_sort_key)
        if missing_node_rows:
            self._handle_validation_failure(
                f"{len(missing_node_rows)} DOT nodes are missing runtime_nodes.csv rows; sample={missing_node_rows[:10]}."
            )

        missing_tensor_rows = sorted(set(dot_tensor_ids) - set(tensor_rows), key=profile_id_sort_key)
        if missing_tensor_rows:
            self._handle_validation_failure(
                f"{len(missing_tensor_rows)} DOT tensors are missing pytorch_runtime_tensors.csv rows; sample={missing_tensor_rows[:10]}."
            )

        return

    def _validate_dot_edges(
        self,
        edge_csv_path: Path,
        dot_edges: list[tuple[str, str, dict[str, str]]],
        manifest: dict[str, Any],
    ) -> None:
        direction_counts = {
            "data_input": 0,
            "data_output": 0,
            "control": 0,
            "invalid": 0,
        }
        dot_pair_to_attrs: dict[tuple[str, str], dict[str, str]] = {}

        for src, dst, attrs in dot_edges:
            dot_pair_to_attrs[(src, dst)] = attrs
            if src.startswith("t") and dst.startswith("k"):
                direction_counts["data_input"] += 1
            elif src.startswith("k") and dst.startswith("t"):
                direction_counts["data_output"] += 1
            elif src.startswith("k") and dst.startswith("k"):
                direction_counts["control"] += 1
            else:
                direction_counts["invalid"] += 1

        if direction_counts["invalid"]:
            self._handle_validation_failure(f"DOT contains {direction_counts['invalid']} unsupported edge directions.")

        if "data_input_edge_count" in manifest and direction_counts["data_input"] != int(manifest["data_input_edge_count"]):
            self._handle_validation_failure(
                f"DOT t->k edge count {direction_counts['data_input']} != manifest data_input_edge_count {manifest['data_input_edge_count']}."
            )

        if "data_output_edge_count" in manifest and direction_counts["data_output"] != int(manifest["data_output_edge_count"]):
            self._handle_validation_failure(
                f"DOT k->t edge count {direction_counts['data_output']} != manifest data_output_edge_count {manifest['data_output_edge_count']}."
            )

        expected_control = sum(int(manifest.get(key, 0)) for key in ("thread_order_edge_count", "stream_order_edge_count", "submit_edge_count", "wait_edge_count"))
        if expected_control and direction_counts["control"] != expected_control:
            self._handle_validation_failure(
                f"DOT k->k edge count {direction_counts['control']} != manifest control edge count {expected_control}."
            )

        dot_pairs = set(dot_pair_to_attrs)
        csv_pairs: set[tuple[str, str]] = set()
        with open(edge_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row["src_node_id"]
                dst = row["dst_node_id"]
                edge_kind = row["edge_kind"]
                csv_pairs.add((src, dst))

                dot_kind = self._edge_kind_from_vertices(src, dst, dot_pair_to_attrs.get((src, dst)))
                if edge_kind in {"thread_order", "stream_order", "submit", "wait"}:
                    if dot_kind != edge_kind:
                        self._handle_validation_failure(
                            f"DOT edge {src}->{dst} label {dot_kind!r} != runtime_edges.csv kind {edge_kind!r}."
                        )
                elif dot_kind != edge_kind:
                    self._handle_validation_failure(
                        f"DOT edge {src}->{dst} inferred kind {dot_kind!r} != runtime_edges.csv kind {edge_kind!r}."
                    )

        if dot_pairs != csv_pairs:
            missing_in_dot = sorted(csv_pairs - dot_pairs)[:10]
            extra_in_dot = sorted(dot_pairs - csv_pairs)[:10]
            self._handle_validation_failure(
                f"DOT edge set differs from runtime_edges.csv: missing_in_dot={missing_in_dot}, extra_in_dot={extra_in_dot}."
            )

        return

    def _read_dot_tensors(
        self,
        dot_tensor_ids: list[str],
        tensor_rows: dict[str, dict[str, str]],
    ) -> tuple[dict[int, Tensor], dict[str, int]]:
        tensor_map: dict[int, Tensor] = {}
        profile_to_tensor: dict[str, int] = {}

        for profile_tensor_id in dot_tensor_ids:
            row = tensor_rows.get(profile_tensor_id)
            if row is None:
                continue
            self._add_tensor_from_row(row, tensor_map, profile_to_tensor)

        return tensor_map, profile_to_tensor

    def _read_dot_nodes(
        self,
        dot_node_ids: list[str],
        node_rows: dict[str, dict[str, str]],
    ) -> tuple[dict[int, Node], dict[str, int]]:
        node_map: dict[int, Node] = {}
        profile_to_node: dict[str, int] = {}

        for profile_node_id in dot_node_ids:
            row = node_rows.get(profile_node_id)
            if row is None:
                continue
            node_id = len(node_map)
            node = self._node_from_row(node_id, row)
            node_map[node_id] = node
            profile_to_node[profile_node_id] = node_id

        return node_map, profile_to_node

    def _read_dot_edges(
        self,
        dot_edges: list[tuple[str, str, dict[str, str]]],
        node_map: dict[int, Node],
        profile_to_node: dict[str, int],
        profile_to_tensor: dict[str, int],
    ) -> None:
        for src, dst, _attrs in dot_edges:
            if src.startswith("t") and dst.startswith("k"):
                tensor_id = profile_to_tensor.get(src)
                node_id = profile_to_node.get(dst)
                if tensor_id is None or node_id is None:
                    continue
                node = node_map[node_id]
                if self._is_pointer_only(node):
                    continue
                node.add_input_tensor(tensor_id)
                continue

            if src.startswith("k") and dst.startswith("t"):
                node_id = profile_to_node.get(src)
                tensor_id = profile_to_tensor.get(dst)
                if node_id is None or tensor_id is None:
                    continue
                node = node_map[node_id]
                if self._is_pointer_only(node):
                    continue
                node.add_output_tensor(tensor_id)
                continue

            if src.startswith("k") and dst.startswith("k"):
                parent_id = profile_to_node.get(src)
                child_id = profile_to_node.get(dst)
                if parent_id is not None and child_id is not None:
                    edge_kind = (_attrs or {}).get("label") or ""
                    if self._is_start_gated_edge(node_map, parent_id, child_id, edge_kind):
                        self._start_gated_edges.append((parent_id, child_id))
                    else:
                        self._add_control_edge(node_map, parent_id, child_id)
                continue

            self._handle_validation_failure(f"Unsupported DOT edge direction: {src}->{dst}.")

        return

    def _load_dot(self, bundle_dir: Path, manifest: dict[str, Any]) -> Trace:
        tensor_csv_path = resolve_path(manifest["tensor_csv"], bundle_dir)
        node_csv_path = resolve_path(manifest["node_csv"], bundle_dir)
        edge_csv_path = resolve_path(manifest["edge_csv"], bundle_dir)

        tensor_rows = self._read_rows_by_id(tensor_csv_path, "tensor_node_id")
        node_rows = self._read_rows_by_id(node_csv_path, "node_id")
        dot_vertices, dot_edges = self._read_dot_bundle(bundle_dir, manifest)

        dot_node_ids = sorted((vid for vid in dot_vertices if vid.startswith("k")), key=profile_id_sort_key)
        dot_tensor_ids = sorted((vid for vid in dot_vertices if vid.startswith("t")), key=profile_id_sort_key)

        self._validate_dot_vertices(dot_node_ids, dot_tensor_ids, node_rows, tensor_rows, manifest)
        if bool(self.args.get("validate_dot_edges", True)):
            self._validate_dot_edges(edge_csv_path, dot_edges, manifest)

        tensor_map, profile_to_tensor = self._read_dot_tensors(dot_tensor_ids, tensor_rows)
        node_map, profile_to_node = self._read_dot_nodes(dot_node_ids, node_rows)
        self._read_dot_edges(dot_edges, node_map, profile_to_node, profile_to_tensor)
        self._apply_storage_aliasing(node_map, tensor_map)
        self._mark_implicit_inputs(node_map, tensor_map)
        # Temporal edges must be added BEFORE annotating alias/dispatcher
        # custom_deps: annotate iterates `node.parent_nodes` to attach
        # NodeDoneDeps, and that list needs to include the temporal-edge
        # parents (e.g. an HtoD memcpy gpu_runtime whose only kineto incoming
        # edge is a start-gated submit has empty parent_nodes until the
        # temporal pass adds its data-producer parent).
        if bool(self.args.get("add_temporal_data_control_edges", False)):
            self._add_temporal_data_control_edges(node_map)
        self._annotate_alias_dispatcher_deps(node_map, tensor_map)
        self._add_terminal_node(node_map)

        return Trace(self.id, self.name, self.log, node_map, tensor_map)

    def _add_temporal_data_control_edges(self, node_map: dict[int, Node]) -> None:
        # Add a control edge from the most recent real producer of
        # each tensor to each real consumer that follows it. Alias /
        # in-place nodes (tid appears in both input and output of the
        # same node) are excluded from BOTH the producer and the
        # consumer aggregation: they don't create the tensor's data
        # and they don't consume it, they touch metadata only.
        # Counting them as producers would let a view be picked as
        # the "most recent producer" before a downstream kernel,
        # missing the real writer further back.
        producers_by_tensor: dict[int, list[int]] = {}
        consumers_by_tensor: dict[int, list[int]] = {}

        for node in node_map.values():
            ins = set(node.input_tensors)
            outs = set(node.output_tensors)
            for tensor_id in outs:
                if tensor_id in ins:
                    continue
                producers_by_tensor.setdefault(tensor_id, []).append(node.id)
            for tensor_id in ins:
                if tensor_id in outs:
                    continue
                consumers_by_tensor.setdefault(tensor_id, []).append(node.id)

        for tensor_id, consumer_ids in consumers_by_tensor.items():
            producer_ids = sorted(producers_by_tensor.get(tensor_id, []))
            if not producer_ids:
                continue

            for consumer_id in consumer_ids:
                idx = bisect_left(producer_ids, consumer_id) - 1
                if idx < 0:
                    continue
                self._add_control_edge(node_map, producer_ids[idx], consumer_id)

        return

    _INITIAL_TYPES = frozenset({"WEIGHT", "INPUT", "LEAF"})

    def _apply_storage_aliasing(self, node_map: dict[int, Node], tensor_map: dict[int, Tensor]) -> None:
        """Merge tensors that actually share storage at the same time.

        Fully data-driven from the tensor map — no op-name heuristics.

        Lifetime-aware storage dedup: group every tensor by
        (device, storage_id). Within a group, two tensors are aliases
        of the *same live allocation* iff their lifetimes overlap.
        PyTorch profile rows for views / aliases / register_buffer
        copies all share the same storage_id with overlapping lifetimes
        — those merge. Storage that gets freed and reallocated by the
        CUDA caching allocator appears as multiple rows with the same
        storage_id but DISJOINT lifetimes — those stay as separate
        cgsim tids and rely on the scheduler's per-tensor
        claim-at-producer / release-at-last-consumer lifecycle to model
        the slot's reuse over time. Peak VRAM then reflects actual
        concurrent occupancy on the slot, not a union-lifetime
        approximation.

        Tensors with type WEIGHT/INPUT/LEAF are treated as alive for
        the entire run (birth=0, death=inf), so any other tensor
        sharing their storage automatically merges into them.
        """
        remap: dict[int, int] = {}

        def resolve(tid: int) -> int:
            seen = []
            while tid in remap:
                seen.append(tid)
                tid = remap[tid]
            for s in seen:
                remap[s] = tid  # path compression
            return tid

        # ---- Compute per-tensor lifetime [birth_ns, death_ns]. ----
        INF = float("inf")
        birth: dict[int, float] = {}
        death: dict[int, float] = {}

        prod_birth: dict[int, float] = {}   # min start_ns over PRODUCING nodes
        first_use: dict[int, float] = {}    # min start_ns over CONSUMING nodes
        for node in node_map.values():
            s = node.args.get("start_ns")
            e = node.args.get("end_ns")
            if s is None: s = 0
            if e is None: e = s
            for tid in node.output_tensors:
                if tid not in prod_birth or s < prod_birth[tid]:
                    prod_birth[tid] = s
                if tid not in death or e > death[tid]:
                    death[tid] = e
            for tid in node.input_tensors:
                # consumed during this node — extends death
                if tid not in death or e > death[tid]:
                    death[tid] = e
                if tid not in first_use or s < first_use[tid]:
                    first_use[tid] = s
        # birth = first production if produced, else FIRST CONSUMPTION (not 0).
        # A producer-less tensor — its producer was a machinery / cross-device
        # dispatcher node the offload reconstruction stripped — is born when it
        # is first read, NOT at time 0. Defaulting such tensors to birth=0 made
        # every sequential reincarnation of a reused storage slot appear to
        # start at the same instant, so the lifetime-overlap clustering below
        # collapsed thousands of 0.1 ms buffers into one union-lifetime mega-tid
        # held resident for seconds — the dominant VRAM over-hold (SD3 peak).
        for tid in set(prod_birth) | set(first_use):
            birth[tid] = prod_birth[tid] if tid in prod_birth else first_use[tid]

        # Permanent tensors (WEIGHT/INPUT/LEAF) live for the whole run.
        for tid, tensor in tensor_map.items():
            if tensor.args.get("tensor_type") in self._INITIAL_TYPES:
                birth[tid] = 0
                death[tid] = INF

        def lifetime(tid: int) -> tuple[float, float]:
            return (birth.get(tid, 0), death.get(tid, INF))

        # Tensor ids that turn out to be aliases of a permanent buffer
        # (WEIGHT/INPUT/LEAF). Their "producer" node is a view-setup or
        # in-place op against the permanent buffer, not a real data write,
        # so we drop them from any node.output_tensors during the rewrite.
        # Otherwise the producer's begin_mutation would invalidate the
        # permanent's region and claim a duplicate, double-allocating the
        # buffer.
        dropped_outputs: set[int] = set()

        def merge_into(keeper_tid: int, victim_tid: int) -> None:
            """Merge victim into keeper. Keeper's `size_bytes` is *never*
            bumped up from a victim — as_strided/expand views can
            over-state the storage size by replicating / striding into
            an oversized numel, so growing from a victim could compound
            that error. The cluster-build loop picks the keeper as the
            largest WEIGHT/INPUT/LEAF row when one exists (its size is
            the real parameter allocation), otherwise the largest member
            of the cluster (smaller members are partial views into the
            storage, so the max is the closest available bound on the
            true allocation).
            """
            keeper = tensor_map[keeper_tid]
            victim = tensor_map[victim_tid]
            keeper.args["profile_tensor_aliases"].append(victim.args.get("profile_tensor_id"))
            # Promote tensor_type: WEIGHT > INPUT > LEAF > INTERMEDIATE.
            v_type = victim.args.get("tensor_type")
            k_type = keeper.args.get("tensor_type")
            if v_type == "WEIGHT":
                keeper.args["tensor_type"] = "WEIGHT"
            elif v_type == "INPUT" and k_type not in ("WEIGHT",):
                keeper.args["tensor_type"] = "INPUT"
            elif v_type == "LEAF" and k_type not in ("WEIGHT", "INPUT"):
                keeper.args["tensor_type"] = "LEAF"
            # If the keeper is a permanent buffer, the victim's producer
            # is a view setup, not a write — drop it from outputs.
            if keeper.args.get("tensor_type") in self._INITIAL_TYPES:
                dropped_outputs.add(victim_tid)
            # Update lifetime in case the keeper's was narrower.
            kb, kd = lifetime(keeper_tid)
            vb, vd = lifetime(victim_tid)
            birth[keeper_tid] = min(kb, vb)
            death[keeper_tid] = max(kd, vd)
            remap[victim_tid] = keeper_tid

        # ---- Lifetime-aware dedup grouped by (device, storage_id). ----
        groups: dict[tuple[str | None, object], list[int]] = {}
        for tid, tensor in tensor_map.items():
            sid = tensor.args.get("storage_id")
            if sid is None:
                continue
            key = (tensor.args.get("device"), sid)
            groups.setdefault(key, []).append(tid)

        for key, tids in groups.items():
            if len(tids) < 2:
                continue
            # Lifetime-aware clustering: members whose [birth, death]
            # intervals overlap are concurrent aliases of the same
            # live allocation; members with disjoint intervals are
            # sequential reincarnations of the same physical slot
            # (allocator reuse) and stay separate. Permanents have
            # death=inf, so any group containing a permanent collapses
            # into a single cluster anchored on it — that case falls
            # out of the same algorithm without a separate branch.
            tids_sorted = sorted(tids, key=lambda t: lifetime(t)[0])
            clusters: list[list[int]] = []
            cluster_deaths: list[float] = []
            for tid in tids_sorted:
                b, d = lifetime(tid)
                placed = False
                for ci in range(len(clusters)):
                    if b <= cluster_deaths[ci]:
                        clusters[ci].append(tid)
                        if d > cluster_deaths[ci]:
                            cluster_deaths[ci] = d
                        placed = True
                        break
                if not placed:
                    clusters.append([tid])
                    cluster_deaths.append(d)
            # Within each cluster, pick an anchor and merge the rest.
            # Anchor preference: largest permanent row (its size is
            # the real parameter allocation), or largest member if no
            # permanent (an upper bound on the underlying storage —
            # smaller members are partial views).
            for c in clusters:
                if len(c) < 2:
                    continue
                permanent = [
                    t for t in c
                    if tensor_map[t].args.get("tensor_type") in self._INITIAL_TYPES
                ]
                if permanent:
                    anchor = max(permanent, key=lambda t: tensor_map[t].size_bytes)
                else:
                    anchor = max(c, key=lambda t: tensor_map[t].size_bytes)
                for victim in c:
                    if victim == anchor:
                        continue
                    merge_into(anchor, victim)

        if not remap:
            return

        def rewrite(lst: list[int], drop_set: set[int] | None = None) -> list[int]:
            seen: set[int] = set()
            out: list[int] = []
            for t in lst:
                if drop_set is not None and t in drop_set:
                    continue
                r = resolve(t)
                if r not in seen:
                    seen.add(r)
                    out.append(r)
            return out

        for node in node_map.values():
            node.input_tensors = rewrite(node.input_tensors)
            node.output_tensors = rewrite(node.output_tensors, drop_set=dropped_outputs)

        for removed_tid in list(remap.keys()):
            tensor_map.pop(removed_tid, None)

        return

    def _annotate_alias_dispatcher_deps(self, node_map: dict[int, Node], tensor_map: dict[int, Tensor]) -> None:
        """Tag pure-alias and dispatcher nodes with `custom_deps` so the
        engine's compute_assertion bypasses its built-in
        input-residency / output-IDLE checks — those assume "tensor data
        is read on compute.memory", which is wrong for CPU-thread
        pointer ops on CUDA tensors and for cross-device allocators.

          - **Alias node** (every output_tensor_id appears in input_tensors
            after storage-aliasing): a view / in-place op. Pure pointer
            work, doesn't read/write tensor data. Only real dependency is
            "all control-graph parents must be DONE".

          - **Dispatcher node** (output home memory != compute.memory —
            classic: aten::empty(device=cuda) on the CPU thread): same
            thing. The scheduler pre-claims the output region on its
            home memory before submit, then we **clear the dispatcher's
            output_tensors** — otherwise core's begin_mutation would
            invalidate every region of those tensors (marking the
            pre-claimed region not-latest) before failing to find any
            on compute.memory. Tagging the node `dispatcher` in args
            lets the scheduler still know which outputs to pre-claim.

        Both kinds get a NodeDoneDep per control-graph parent. That
        replaces the engine's built-in `for p in parent_nodes: status==DONE`
        check exactly, while skipping the inappropriate residency checks.

        Standard nodes (real computes whose inputs/outputs match
        compute.memory) get no custom_deps and run through the normal
        data-flow path.

        No-parent gap: an IN-PLACE alias (output==input) is excluded from the
        temporal data edges (it touches metadata, not data), so it can end up
        with NO parents at all. With empty parent_nodes the loop below leaves
        `custom_deps` empty -> the engine takes the NORMAL path and demands the
        (cuda) tensor be resident in the cpu op's local memory (RAM) -> deadlock
        (seen on 8B: a root `aten::as_strided` on a cuda activation). So when the
        parent loop leaves custom_deps empty, fall back to a dep on the producer
        of each cuda input (correct ordering + triggers the bypass); if even that
        is empty (cuda input is an always-resident initial), a MinTimestampDep(0)
        forces the bypass branch without adding a real wait.
        """
        producer_of: dict[int, int] = {}
        for n in node_map.values():
            for tid in n.output_tensors:
                if tid not in n.input_tensors:
                    producer_of.setdefault(tid, n.id)

        def is_cuda_tid(tid: int) -> bool:
            t = tensor_map.get(tid)
            return t is not None and str(t.args.get("device") or "cpu").lower().startswith("cuda")

        for node in node_map.values():
            outs = node.output_tensors
            ins = node.input_tensors

            is_alias = bool(outs) and all(t in ins for t in outs)
            is_dispatcher = self._loader_is_dispatcher(node, tensor_map)

            # A cpu_leaf node whose outputs were dropped by storage
            # aliasing (because they aliased a permanent buffer) AND
            # whose inputs are all on CUDA is a pointer-only op on a
            # CUDA tensor: aten::view of a weight, aten::as_strided
            # into a permanent's storage, in-place metadata ops, etc.
            # The CPU thread cannot read the CUDA tensor's bytes
            # without an explicit sync/copy (those would be separate
            # nodes), and the node writes nothing (outputs dropped),
            # so the op's recorded duration is RecordFunction
            # enter/exit overhead, not data work.
            #
            # Restricted to `bool(ins)` and `all ins CUDA` on purpose
            # — those are the constraints that distinguish this
            # class from other no-data-flow ops that DO have CPU
            # work (allocations like aten::empty, CPU API queries
            # like cudaDeviceGetAttribute, cudaMemsetAsync issues,
            # etc.). Broadening to any "no data flow" cpu_leaf
            # over-matches.
            #
            # Without custom_deps such a node would fall through to
            # the normal compute path, whose _ensure_inputs_resident
            # claims a cpu.memory region for each CUDA input and
            # fires a VRAM->RAM transfer to "land" data the op never
            # actually reads. Tagging it alias-class skips that path.
            role = node.args.get("runtime_role") or ""
            is_pointer_metadata = (
                role == "cpu_leaf"
                and not outs
                and bool(ins)
                and all(
                    (tensor_map.get(t) is not None)
                    and ((tensor_map[t].args.get("device") or "cpu").lower().startswith("cuda"))
                    for t in ins
                )
            )

            if not (is_alias or is_dispatcher or is_pointer_metadata):
                continue

            if is_dispatcher and not is_alias:
                # Stash the cross-device outputs for the scheduler to
                # pre-claim, then clear them from the node so the engine
                # doesn't invalidate the pre-claimed region.
                cross = [t for t in outs if t not in ins]
                node.args["dispatcher_outputs"] = list(cross)
                node.output_tensors = [t for t in outs if t not in cross]

            for parent_id in node.parent_nodes:
                node.custom_deps.append(NodeDoneDep(parent_id))

            # No-parent fallback (see docstring): keep custom_deps NON-empty so
            # the residency bypass actually fires for a cpu pointer-op on a cuda
            # tensor. Add a real CONTROL EDGE producer->node (so a dependency-
            # driven scheduler waits for the data before SUBMITTING this node --
            # custom_deps gate the engine's runnability but NOT scheduler
            # submission order) AND a matching NodeDoneDep (engine bypass). If a
            # cuda input has no producer (it is an always-resident initial), a
            # MinTimestampDep(0) forces the bypass branch without a real wait.
            if not node.custom_deps:
                for tid in node.input_tensors:
                    if is_cuda_tid(tid):
                        pr = producer_of.get(tid)
                        if pr is not None and pr != node.id:
                            self._add_control_edge(node_map, pr, node.id)
                            node.custom_deps.append(NodeDoneDep(pr))
                if not node.custom_deps and any(is_cuda_tid(t) for t in node.input_tensors):
                    node.custom_deps.append(MinTimestampDep(0.0))

            # NOTE: previous iterations of this branch zeroed
            # `compute_time_micros` for alias / dispatcher /
            # pointer-metadata cpu_leaf nodes under the framing
            # "recorded duration is RecordFunction overhead, not real
            # work." That framing was fit-shaped — the recorded
            # duration also contains the actual op cost (C++ dispatch,
            # TensorImpl construction, allocator call), not just the
            # RF wrapper. Zeroing removed real CPU work and let the
            # sim's CPU pipeline race ahead of the GPU. The sim now
            # uses the recorded duration as-is for every cpu_leaf;
            # the custom_deps annotation above still bypasses the
            # spurious VRAM→RAM transfers that would otherwise fire
            # on alias / dispatcher / pointer-metadata nodes via
            # `_ensure_inputs_resident`.

    @staticmethod
    def _loader_is_dispatcher(node: Node, tensor_map: dict[int, Tensor]) -> bool:
        """A node is dispatcher-style if it produces (non-aliased) outputs
        on a device different from where the node itself runs. Determined
        purely from the profile: cpu_leaf nodes producing CUDA tensors
        and gpu_runtime nodes producing CPU tensors are dispatchers."""
        role = node.args.get("runtime_role", "")
        compute_dev = "cuda" if role == "gpu_runtime" else "cpu"
        for tid in node.output_tensors:
            if tid in node.input_tensors:
                continue
            tensor = tensor_map.get(tid)
            if tensor is None:
                continue
            tdev = (tensor.args.get("device") or "cpu").lower()
            tdev_short = "cuda" if tdev.startswith("cuda") else "cpu"
            if tdev_short != compute_dev:
                return True
        return False

    def _mark_implicit_inputs(self, node_map: dict[int, Node], tensor_map: dict[int, Tensor]) -> None:
        # A tensor is an "implicit input" iff there is no real data
        # producer for it inside the trace. We approximate "real
        # producer" / "real consumer" by EXCLUDING any node that uses
        # the tensor as both input AND output — those are view / alias
        # / in-place ops (aten::view, aten::as_strided, aten::detach_,
        # in-place RMW kernels) which neither *create* nor *consume*
        # the tensor's data — they just touch metadata or rewrite
        # in-place. Both sides of the same check must skip them.
        #
        # The producer-exclusion was already in place; the consumer
        # side used to count alias nodes as consumers. On diffusion
        # eager traces (sdxl/sd3) the pattern conv(t)->view(t) makes
        # the view node the *first* node referencing tid (view's input
        # AND output is t, since storage-aliasing has collapsed all
        # views of the same storage onto t). Counting view as a
        # consumer made `first_consumer(view) <= first_producer(conv)`
        # trip the check, so thousands of conv-output intermediates
        # were retyped INPUT and pre-claimed at layout. sd3_med_eager
        # over-claimed 22 GB this way and aborted with VRAM exhaustion;
        # sdxl_turbo_eager inflated peak VRAM by 1.2 GB.
        producers_by_tensor: dict[int, list[int]] = {}
        consumers_by_tensor: dict[int, list[int]] = {}

        for node in node_map.values():
            ins = set(node.input_tensors)
            outs = set(node.output_tensors)
            # Real producer / real consumer = node where the tid is on
            # exactly one side (a real write OR a real read), not an
            # alias / in-place op where the tid is on both sides.
            for tensor_id in outs:
                if tensor_id in ins:
                    continue
                producers_by_tensor.setdefault(tensor_id, []).append(node.id)
            for tensor_id in ins:
                if tensor_id in outs:
                    continue
                consumers_by_tensor.setdefault(tensor_id, []).append(node.id)

        for tensor_id, tensor in tensor_map.items():
            if tensor.args.get("tensor_type") == "WEIGHT":
                continue

            producer_ids = producers_by_tensor.get(tensor_id, [])
            consumer_ids = consumers_by_tensor.get(tensor_id, [])
            if not consumer_ids:
                continue

            first_consumer = min(consumer_ids)
            first_producer = min(producer_ids) if producer_ids else None
            if first_producer is None or first_consumer <= first_producer:
                tensor.args["tensor_type"] = "INPUT"
                tensor.args["implicit_input"] = True

        return

    def _add_terminal_node(self, node_map: dict[int, Node]) -> None:
        terminal_id = len(node_map)
        terminal = TerminalNode(terminal_id, "TERMINAL_NODE")

        leaves = [node.id for node in node_map.values() if not node.children_nodes]
        if not leaves and node_map:
            leaves = [next(reversed(node_map))]

        node_map[terminal_id] = terminal
        for parent_id in leaves:
            self._add_control_edge(node_map, parent_id, terminal_id)

        return

    def _load_probe_effect_table(self, trace_dir: Path) -> dict[str, int]:
        """Per-trace calibration of kineto-induced duration inflation.

        Only invoked when the sim-config arg
        `cpu_node_probe_effect_compensate` is truthy. Looks for
        `probe_effect_table.csv` in the trace directory (sibling of the
        bundle dir). CSV schema:
            op_name,probe_effect_ns[,note]

        Missing file → returns empty dict (no correction applied).
        See docs/eager-lazy-probing-effect.md for derivation.
        """
        path = trace_dir / "probe_effect_table.csv"
        if not path.exists():
            return {}
        table: dict[str, int] = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    table[r["op_name"]] = int(float(r["probe_effect_ns"]))
                except (KeyError, ValueError):
                    continue
        print(f"[PytorchOffloadLoader] Loaded probe_effect_table.csv ({len(table)} entries) from {path}")
        return table

    # ------------------------------------------------------------------ offload
    # Matches BOTH "Memcpy HtoD (Pageable -> Device)" (accelerate) and
    # "Memcpy HtoD (Pinned -> Device)" (diffusers, low_cpu_mem_usage=False pins
    # the cpu masters). The prefix union is harmless for accelerate (pageable-only).
    _MEMCPY_HTOD_PREFIX = "Memcpy HtoD"
    # A non-master cpu tensor consumed by a gpu kernel is a by-value scalar arg
    # (diffusion-scheduler sigmas / scales), not a device-memory operand. Anything
    # larger is treated as a possible MISSING transfer and left in place (surfaces).
    _SCALAR_ARG_MAX_BYTES = 4096

    def _reconstruct_offload(self, node_map: dict[int, Node], tensor_map: dict[int, Tensor]) -> None:
        """Reconstruct cpu_offload weights from the reincarnation soup.

        Runs on the RAW graph (after `_read_edges`, BEFORE `_apply_storage_aliasing`),
        gated by the `offload_reconstruct` arg. See `docs/known_problems.md` for the
        evidence. Summary of the HF-accelerate pattern this undoes:

          - A weight is re-streamed to the GPU every forward via a `Memcpy HtoD`
            from a persistent cpu tensor (the RAM master). The GEMM does NOT read
            the Memcpy's dest via a data edge (a `detach` severs it with no output);
            it reads a *view* (`as_strided`/`empty_strided`) that shares the dest's
            cuda storage_id. So the load<->use link is by SHARED CUDA STORAGE, not
            dataflow.
          - cuda buffers are a small reused pool (~39 storages) holding different
            weights — and activations — over time, so we disambiguate by lifetime
            EPOCH (recorded start_ns) and IDENTITY (view bytes == master bytes).

        Transform (validated: 255 masters, 3826 loads=108 GB, 8492 weight reads,
        0 epoch/identity conflicts):
          1. Group Memcpys by cpu-source storage_id -> 255 masters (one persistent
             RAM tensor each, kept as the cpu-source tid).
          2. For each cuda buffer storage S, the sorted Memcpy times define epochs;
             epoch i holds master P_i.
          3. A cuda tensor on a buffer storage whose bytes == its epoch master's
             bytes is weight machinery (buffer / view). Redirect its gpu_runtime
             consumers to read the master tid; drop it from non-gpu (metadata) ops;
             remove it from the tensor map. => the GEMM now depends on the master,
             which the scheduler streams RAM->VRAM (load-before-use as a real dep).
          4. Neutralize the recorded `Memcpy HtoD` (and `cudaMemcpyAsync`) compute
             so the H2D is modeled once, by the scheduler's transfer (P3).
          5. Emit `trace.args["offload"]` = {master_tids, evict_after_node}: per
             weight, the LAST gpu use in each epoch is an eviction point, so the
             scheduler frees VRAM per-forward and the next forward re-streams it
             => modelled volume == recorded 3826 loads by construction (P5/P9).
        """
        def dev(t: Tensor) -> str:
            return (t.args.get("device") or "cpu").lower()

        def is_cuda(t: Tensor) -> bool:
            return dev(t).startswith("cuda")

        def sid(t: Tensor):
            return t.args.get("storage_id")

        node_start = {nid: (n.args.get("start_ns") or 0) for nid, n in node_map.items()}

        memcpys = [n for n in node_map.values()
                   if (n.args.get("op_name") or "").startswith(self._MEMCPY_HTOD_PREFIX)]
        if not memcpys:
            print("[PytorchOffloadLoader] offload_reconstruct: no Memcpy HtoD nodes; "
                  "trace is not an offload bundle, skipping.")
            return

        # --- variant detection: accelerate (synchronous H2D, single compute stream)
        #     vs diffusers (async H2D on a side stream + prefetch). The decisive,
        #     trace-grounded signal is whether the host cudaMemcpyAsync BLOCKS for the
        #     copy: ratio = sum(host cudaMemcpyAsync dur) / sum(device Memcpy HtoD dur).
        #       accelerate ~1.0  -> synchronous copy; the host dur IS the H2D, so
        #                           zeroing cudaMemcpyAsync avoids a +H2D double-count (P3).
        #       diffusers ~0.02-0.05 -> quick async submit; the device Memcpy carries
        #                           the H2D, the host call is genuine ~5us CPU dispatch
        #                           on the critical path -> KEEP it.
        #     `offload_variant` arg overrides ("accelerate"/"diffusers"); default "auto".
        dev_htod_dur = sum(n.compute_time_micros for n in memcpys)
        host_async_dur = sum(n.compute_time_micros for n in node_map.values()
                             if (n.args.get("op_name") or "") == "cudaMemcpyAsync")
        ratio = (host_async_dur / dev_htod_dur) if dev_htod_dur else 1.0
        variant = str(self.args.get("offload_variant", "auto")).lower()
        if variant == "auto":
            variant = "accelerate" if ratio >= 0.5 else "diffusers"
        zero_cuda_async = (variant == "accelerate")
        print(f"[PytorchOffloadLoader] offload_variant={variant} "
              f"(host/device dur ratio={ratio:.3f}; "
              f"cudaMemcpyAsync {'zeroed' if zero_cuda_async else 'kept as CPU dispatch'})")

        # --- 1+2. Memcpy events: cuda buffer storage -> sorted (time, master P, mc) ---
        events_by_S: dict[Any, list[tuple[int, Any, int]]] = defaultdict(list)
        master_tid_by_storage: dict[Any, int] = {}
        master_size_by_storage: dict[Any, int] = {}
        loads_by_master: dict[Any, list[int]] = defaultdict(list)
        mc_to_storage: dict[int, Any] = {}   # memcpy node id -> cpu-source storage P
        skipped_mc = 0
        for mc in memcpys:
            cpu_src = [t for t in mc.input_tensors if not is_cuda(tensor_map[t])]
            cuda_dst = [t for t in mc.output_tensors if is_cuda(tensor_map[t])]
            if not cpu_src or not cuda_dst:
                skipped_mc += 1
                continue
            psrc = cpu_src[0]
            P = sid(tensor_map[psrc])
            S = sid(tensor_map[cuda_dst[0]])
            t = node_start[mc.id]
            events_by_S[S].append((t, P, mc.id))
            loads_by_master[P].append(t)
            mc_to_storage[mc.id] = P
            cur = master_tid_by_storage.get(P)
            if cur is None or tensor_map[psrc].size_bytes > tensor_map[cur].size_bytes:
                master_tid_by_storage[P] = psrc
            master_size_by_storage[P] = max(master_size_by_storage.get(P, 0),
                                            tensor_map[psrc].size_bytes)
        for S in events_by_S:
            events_by_S[S].sort()
        times_by_S = {S: [e[0] for e in evs] for S, evs in events_by_S.items()}
        buffer_storages = set(events_by_S)
        master_storages = set(master_tid_by_storage)
        master_tids = set(master_tid_by_storage.values())

        # tensor reference time = earliest start_ns of any node touching it
        # Per-tid helpers: the role of its (non-in-place) producer, and the
        # earliest gpu-consumer time. A weight buffer/view is produced by a
        # cpu_leaf alias/alloc op (empty_strided / as_strided / view / detach /
        # copy_ / _unsafe_view / to) or has NO producer (orphan view recorded by
        # the profiler); a real ACTIVATION is produced by a gpu_runtime KERNEL.
        # That producer-role distinction is the precise discriminator (a weight
        # view and an activation can share bytes + buffer storage). The gpu uses
        # of a view come AFTER its copy, so the consumer time gives the right
        # epoch for the redirect (a buffer's own alloc time is pre-copy).
        producer_role: dict[int, str] = {}
        gpu_consumer_time: dict[int, int] = {}
        for n in node_map.values():
            role = n.args.get("runtime_role") or ""
            for tid in n.output_tensors:
                if tid in n.input_tensors:
                    continue
                producer_role.setdefault(tid, role)
            if role == "gpu_runtime":
                s = node_start[n.id]
                for tid in n.input_tensors:
                    cur = gpu_consumer_time.get(tid)
                    if cur is None or s < cur:
                        gpu_consumer_time[tid] = s
        master_sizes_by_S: dict[Any, set[int]] = {
            S: set(master_size_by_storage[P] for (_, P, _) in evs)
            for S, evs in events_by_S.items()
        }

        # --- 3. classify machinery tids; build remap (-> master) + removal set ---
        tid_remap: dict[int, int] = {}   # machinery / cpu-extra tid -> master tid
        remove_tids: set[int] = set()
        # (a) weight machinery (buffers + views, gpu-consumed or orphan) on a buffer
        # storage, sized like a master, NOT produced by a real gpu kernel.
        for tid, t in tensor_map.items():
            if not is_cuda(t):
                continue
            S = sid(t)
            if S not in buffer_storages:
                continue
            if t.size_bytes not in master_sizes_by_S[S]:
                continue                              # activation of other size -> keep
            if producer_role.get(tid) == "gpu_runtime":
                continue                              # a real activation -> keep
            remove_tids.add(tid)
            tg = gpu_consumer_time.get(tid)
            if tg is None:
                continue                              # orphan/buffer: remove, no redirect
            idx = max(0, bisect_right(times_by_S[S], tg) - 1)
            P = events_by_S[S][idx][1]
            if master_size_by_storage[P] != t.size_bytes:
                P = next((PP for (_, PP, _) in events_by_S[S]
                          if master_size_by_storage[PP] == t.size_bytes), P)
            tid_remap[tid] = master_tid_by_storage[P]
        # (b) collapse any extra cpu tids that share a master's storage.
        for tid, t in tensor_map.items():
            if is_cuda(t):
                continue
            S = sid(t)
            if S in master_storages and tid != master_tid_by_storage[S]:
                tid_remap[tid] = master_tid_by_storage[S]
                remove_tids.add(tid)

        # --- 4. turn each Memcpy into an explicit transfer TRIGGER; neutralize
        #        the host cudaMemcpyAsync; rewrite node tensor io ---
        # FAITHFUL TRANSLATION (per the loader<->scheduler co-design principle):
        # every recorded `Memcpy HtoD` becomes one explicit RAM->VRAM TransferJob,
        # fired by the scheduler when this node reaches its position in the graph
        # (stream_order) -- NOT a residency heuristic. This mirrors the trace as-is
        # and generalises to diffusers (prefetch loads recorded ahead of use, and
        # `Memcpy DtoH` copy-back, become transfers at their own positions).
        #   - The DEVICE Memcpy's recorded duration is dropped (compute_time=0);
        #     the H2D wall time comes from the TransferJob (bytes / bandwidth).
        #   - The HOST cudaMemcpyAsync is also zeroed: measured host/device
        #     duration ratio = 1.012, i.e. the host call BLOCKS for the whole H2D
        #     (synchronous copy), so its duration IS the H2D time -- keeping it
        #     would double-count (cudaStreamSynchronize is already zeroed by
        #     zero_wait_nodes). Genuine cpu_leaf dispatch work is left intact.
        memcpy_ids = {mc.id for mc in memcpys}
        for n in node_map.values():
            if n.id in memcpy_ids:
                P = mc_to_storage.get(n.id)
                n.input_tensors = []
                n.output_tensors = []
                n.compute_time_micros = 0.0
                if P is not None:
                    # scheduler reads this: fire a RAM->VRAM transfer of `master`
                    # when this node is dispatched, mark the node DONE on retire.
                    n.args["offload_transfer"] = {
                        "master": master_tid_by_storage[P],
                        "dir": "h2d",
                    }
                continue
            if zero_cuda_async and (n.args.get("op_name") or "") == "cudaMemcpyAsync":
                n.compute_time_micros = 0.0
            is_gpu = (n.args.get("runtime_role") or "") == "gpu_runtime"
            new_in: list[int] = []
            for tid in n.input_tensors:
                if tid in remove_tids:
                    if is_gpu:
                        r = tid_remap[tid]          # real weight read -> master
                        if r not in new_in:
                            new_in.append(r)
                    # non-gpu metadata op on the weight buffer -> drop (inert)
                else:
                    if tid not in new_in:
                        new_in.append(tid)
            n.input_tensors = new_in
            new_out: list[int] = []
            for tid in n.output_tensors:
                if tid in remove_tids:
                    continue                        # producer stops producing it
                if tid not in new_out:
                    new_out.append(tid)
            n.output_tensors = new_out

        for tid in remove_tids:
            tensor_map.pop(tid, None)

        # --- 5. mark masters (cpu home; INPUT masters lay out in RAM at layout) ---
        for P, mtid in master_tid_by_storage.items():
            mt = tensor_map[mtid]
            mt.args["device"] = "cpu"
            mt.args["offload_master"] = True

        # --- 5b. cold buffers (P8): initial, non-master tensors the profiler labels
        # cpu but that are read ONLY by gpu kernels (rotary inv_freq, scales, ...).
        # They have no Memcpy (never streamed) and the kernel reads them on-device,
        # so retarget -> cuda; layout then makes them VRAM-resident. (Leave any that
        # a cpu op also reads: those would need both homes — not seen in this trace.)
        gpu_consumed: set[int] = set()
        cpu_consumed: set[int] = set()
        for n in node_map.values():
            is_gpu_n = (n.args.get("runtime_role") or "") == "gpu_runtime"
            for tid in n.input_tensors:
                (gpu_consumed if is_gpu_n else cpu_consumed).add(tid)
        cold_retargeted = 0
        for tid, t in tensor_map.items():
            if tid in master_tids:
                continue
            if t.args.get("tensor_type") not in self._INITIAL_TYPES:
                continue
            if str(t.args.get("device") or "cpu").lower().startswith("cuda"):
                continue
            if tid in gpu_consumed and tid not in cpu_consumed:
                t.args["device"] = "cuda:0"
                cold_retargeted += 1

        # --- 5c. tiny cpu scalars consumed by gpu kernels are by-VALUE kernel args,
        # NOT device-memory operands. A gpu kernel like
        # AUnaryFunctor<...MulFunctor<float>> CAPTURES a scalar (diffusion-scheduler
        # sigmas / scales computed on the cpu thread by aten::sub / aten::empty).
        # They are never streamed (no Memcpy, not a master), so cg-sim's residency
        # gate would demand them in VRAM and DEADLOCK (a real gpu node reading a cpu
        # tensor). Drop them from the gpu node's inputs (the cpu producer stays a
        # control parent for ordering); they remain on cpu for any cpu consumer
        # (several are dual-consumed, so retargeting to cuda is wrong). GUARD: a
        # LARGE non-master cpu->gpu input would signal a MISSING transfer, so we do
        # NOT hide it (leave it in place -> it surfaces, per the no-safety-net rule).
        producer_of_now: dict[int, int] = {}
        for n in node_map.values():
            for o in n.output_tensors:
                if o not in n.input_tensors:
                    producer_of_now.setdefault(o, n.id)
        scalar_args_dropped = 0
        big_cpu_gpu_inputs: list[tuple[int, int]] = []
        for n in node_map.values():
            if (n.args.get("runtime_role") or "") != "gpu_runtime":
                continue
            kept: list[int] = []
            for tid in n.input_tensors:
                t = tensor_map.get(tid)
                if (t is None or tid in master_tids
                        or str(t.args.get("device") or "cpu").lower().startswith("cuda")):
                    kept.append(tid)
                    continue
                if t.size_bytes > self._SCALAR_ARG_MAX_BYTES:
                    big_cpu_gpu_inputs.append((tid, t.size_bytes))
                    kept.append(tid)
                    continue
                pr = producer_of_now.get(tid)
                if pr is not None and pr != n.id and pr not in n.parent_nodes:
                    self._add_control_edge(node_map, pr, n.id)
                scalar_args_dropped += 1
            n.input_tensors = kept
        if big_cpu_gpu_inputs:
            print(f"[PytorchOffloadLoader] WARNING: {len(big_cpu_gpu_inputs)} large "
                  f"(> {self._SCALAR_ARG_MAX_BYTES}B) non-master cpu->gpu inputs left "
                  f"in place (possible MISSING transfer): {big_cpu_gpu_inputs[:5]}")

        # --- 6. evict schedule: per master, the LAST gpu use in each epoch ---
        gpu_uses_by_master: dict[Any, list[tuple[int, int]]] = defaultdict(list)
        tid_to_P = {mtid: P for P, mtid in master_tid_by_storage.items()}
        for n in node_map.values():
            if (n.args.get("runtime_role") or "") != "gpu_runtime":
                continue
            s = node_start[n.id]
            for tid in n.input_tensors:
                P = tid_to_P.get(tid)
                if P is not None:
                    gpu_uses_by_master[P].append((s, n.id))

        evict_after_node: dict[int, list[int]] = defaultdict(list)
        epochs_with_use = 0
        epochs_total = 0
        for P, mtid in master_tid_by_storage.items():
            loads = sorted(loads_by_master[P])
            epochs_total += len(loads)
            last_use: dict[int, tuple[int, int]] = {}   # epoch idx -> (start, node)
            for (us, nid) in gpu_uses_by_master.get(P, []):
                ei = bisect_right(loads, us) - 1
                if ei < 0:
                    ei = 0
                if ei not in last_use or us >= last_use[ei][0]:
                    last_use[ei] = (us, nid)
            epochs_with_use += len(last_use)
            for (_us, nid) in last_use.values():
                evict_after_node[nid].append(mtid)

        # --- stash the scheduler contract + stats ---
        self._offload = {
            "variant": variant,
            "master_tids": sorted(master_tids),
            "evict_after_node": {nid: tids for nid, tids in evict_after_node.items()},
        }
        transfer_triggers = sum(1 for n in node_map.values()
                                if "offload_transfer" in n.args)
        self._offload_stats = {
            "variant": variant,
            "memcpys": len(memcpys),
            "transfer_triggers": transfer_triggers,
            "skipped_memcpys": skipped_mc,
            "masters": len(master_tids),
            "cold_buffers_retargeted": cold_retargeted,
            "scalar_args_dropped": scalar_args_dropped,
            "buffer_storages": len(buffer_storages),
            "machinery_tids_removed": len(remove_tids),
            "gpu_weight_reads_redirected": sum(len(v) for v in gpu_uses_by_master.values()),
            "epochs_total": epochs_total,
            "epochs_with_gpu_use": epochs_with_use,
            "evict_points": sum(len(v) for v in evict_after_node.values()),
        }
        print(f"[PytorchOffloadLoader] offload_reconstruct: {self._offload_stats}")
        return

    def load(self) -> Trace:
        bundle_dir, manifest = self._bundle_paths()

        # Per-op probe-effect calibration (opt-in via the
        # `cpu_node_probe_effect_compensate` sim-config arg; default off).
        # Table lives one level up from the bundle (i.e. in the trace
        # directory). Applied in `_node_from_row`.
        self._probe_effect_ns: dict[str, int] = (
            self._load_probe_effect_table(bundle_dir.parent)
            if bool(self.args.get("cpu_node_probe_effect_compensate", False))
            else {}
        )
        self._probe_clamp_count: int = 0

        # Accumulator for start-gated edges (kineto submit edges from
        # a submit-role node into a gpu_runtime kernel). These are
        # carved out of the control graph in `_read_edges` /
        # `_read_dot_edges` and exposed to schedulers via
        # `trace.args["start_gated_edges"]`. See `_is_start_gated_edge`
        # for the predicate.
        self._start_gated_edges: list[tuple[int, int]] = []

        # Offload reconstruction contract (set by `_reconstruct_offload` when the
        # `offload_reconstruct` arg is on); exposed via `trace.args["offload"]`.
        self._offload: dict[str, Any] | None = None

        offload_reconstruct = bool(self.args.get("offload_reconstruct", False))

        graph_source = str(self.args.get("graph_source", "csv")).lower()
        if graph_source == "dot":
            trace = self._load_dot(bundle_dir, manifest)
        elif graph_source == "csv":
            tensor_csv_path = resolve_path(manifest["tensor_csv"], bundle_dir)
            node_csv_path = resolve_path(manifest["node_csv"], bundle_dir)
            edge_csv_path = resolve_path(manifest["edge_csv"], bundle_dir)

            tensor_map, profile_to_tensor = self._read_tensors(tensor_csv_path)
            node_map, profile_to_node = self._read_nodes(node_csv_path)
            self._read_edges(edge_csv_path, node_map, profile_to_node, profile_to_tensor)
            # Offload reconstruction runs on the RAW graph (raw storage_ids +
            # start_ns), BEFORE storage-aliasing scrambles/severs the weight
            # machinery. Gated so non-offload traces are untouched (P12).
            if offload_reconstruct:
                self._reconstruct_offload(node_map, tensor_map)
            self._apply_storage_aliasing(node_map, tensor_map)
            self._mark_implicit_inputs(node_map, tensor_map)
            if bool(self.args.get("add_temporal_data_control_edges", False)):
                self._add_temporal_data_control_edges(node_map)
            self._annotate_alias_dispatcher_deps(node_map, tensor_map)
            self._add_terminal_node(node_map)
            trace = Trace(self.id, self.name, self.log, node_map, tensor_map)
        else:
            raise Exception(f"[PytorchOffloadLoader] Unsupported graph_source: {graph_source}")

        trace.args["start_gated_edges"] = list(self._start_gated_edges)
        if self._offload is not None:
            trace.args["offload"] = self._offload

        if self._probe_effect_ns and self._probe_clamp_count:
            print(f"[PytorchOffloadLoader] probe_effect clamped to 0 for "
                  f"{self._probe_clamp_count} node(s) (duration_ns < probe_effect_ns).")

        # Optional: inject a weight-streaming schedule so DAV simulates
        # the schedule's effect via standard transfer-on-input-mismatch
        # logic, without requiring ScheduleReplay. Path can be relative
        # to the bundle dir.
        inject_path = self.args.get("inject_schedule_path")
        if inject_path:
            inject_path_resolved = resolve_path(inject_path, bundle_dir)
            try:
                from graph_modifiers.inject_schedule import (
                    inject_schedule_into_trace,
                )
            except ImportError as e:
                raise Exception(
                    "[PytorchOffloadLoader] inject_schedule_path requires "
                    "graph_modifiers.inject_schedule "
                    "to be importable. "
                    f"Underlying: {e}"
                )
            print(f"[PytorchOffloadLoader] injecting schedule from {inject_path_resolved}",
                  flush=True)
            inject_schedule_into_trace(
                trace, str(inject_path_resolved),
                bundle_dir=bundle_dir,
                disable_evict=bool(self.args.get("inject_disable_evict", False)),
            )

        # Optional eager-mode injection: schedule keyed by raw cgsim
        # node_id / tid (no compile sidecars). Used by hf_accelerate
        # and any future eager-bundle scheduler. Removed in 8b79b13
        # along with the eager injector module move; restored here
        # because the hf_accelerate workflow depends on it.
        eager_inject_path = self.args.get("inject_eager_schedule_path")
        if eager_inject_path:
            eager_path_resolved = resolve_path(eager_inject_path, bundle_dir)
            try:
                from graph_modifiers.inject_schedule import (
                    inject_eager_schedule_into_trace,
                )
            except ImportError as e:
                raise Exception(
                    "[PytorchOffloadLoader] inject_eager_schedule_path requires "
                    "graph_modifiers.inject_schedule.inject_eager_schedule_into_trace "
                    "to be importable. "
                    f"Underlying: {e}"
                )
            print(
                f"[PytorchOffloadLoader] injecting eager schedule from "
                f"{eager_path_resolved}", flush=True,
            )
            inject_eager_schedule_into_trace(
                trace, str(eager_path_resolved),
                disable_evict=bool(self.args.get("inject_disable_evict", False)),
            )

        return trace

    def placement(self, trace: Trace, storage: BaseStorage) -> None:
        initial_tensor_types = set(self.args.get("initial_tensor_types", ["WEIGHT", "INPUT", "LEAF"]))

        for tensor in trace.tensor_map.values():
            if tensor.args.get("tensor_type") not in initial_tensor_types:
                continue

            stor_region = storage.space.claim(tensor.id, -1, tensor.num_pages)
            stor_region.is_ready = True
            stor_region.is_latest = True

        return
