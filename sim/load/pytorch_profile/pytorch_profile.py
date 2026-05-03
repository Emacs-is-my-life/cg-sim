from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Any

from sim.core.trace import Node, Tensor, TerminalNode, Trace, TraceLoader
from sim.core.trace.custom_dep import NodeDoneDep
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


class PytorchProfile(TraceLoader):
    """Trace loader for flattened PyTorch profiler runtime bundles."""

    @staticmethod
    def _num_pages(size_bytes: int) -> int:
        align_bytes = 64
        page_size_bytes = 4096
        tensor_aligned_size_bytes = ((size_bytes + align_bytes - 1) // align_bytes) * align_bytes
        return int(math.ceil(tensor_aligned_size_bytes / page_size_bytes))

    def _bundle_paths(self) -> tuple[Path, dict[str, Any]]:
        input_dir = Path(self.args["input_path"]).parent
        profile_dir = resolve_path(self.args.get("profile_dir", input_dir), input_dir)
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
                    raise Exception(f"[PytorchProfile] Duplicate row id {row_id} in {csv_path}.")
                rows_by_id[row_id] = row
        return rows_by_id

    def _handle_validation_failure(self, msg: str) -> None:
        if bool(self.args.get("strict_dot_validation", True)):
            raise Exception(f"[PytorchProfile] {msg}")
        print(f"[PytorchProfile] Warning: {msg}")
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
        compute_time_micros = (
            0.0
            if bool(self.args.get("zero_wait_nodes", True)) and runtime_role == "wait"
            else duration_ns / 1_000
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
                    if parent_id is not None and child_id is not None:
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
            raise Exception("[PytorchProfile] manifest does not define step_dot_files.")
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
        self._annotate_alias_dispatcher_deps(node_map, tensor_map)
        if bool(self.args.get("add_temporal_data_control_edges", False)):
            self._add_temporal_data_control_edges(node_map)
        self._add_terminal_node(node_map)

        return Trace(self.id, self.name, self.log, node_map, tensor_map)

    def _add_temporal_data_control_edges(self, node_map: dict[int, Node]) -> None:
        producers_by_tensor: dict[int, list[int]] = {}
        consumers_by_tensor: dict[int, list[int]] = {}

        for node in node_map.values():
            for tensor_id in node.output_tensors:
                producers_by_tensor.setdefault(tensor_id, []).append(node.id)
            for tensor_id in node.input_tensors:
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

        Pass A (lifetime-aware storage dedup): group every tensor by
        (device, storage_id). Within a group, two tensors are aliases of
        the same underlying allocation iff their lifetimes overlap. PyTorch
        profile rows for views / aliases / register_buffer copies all share
        the same storage_id with overlapping lifetimes — those merge.
        Storage that gets freed and reallocated (allocator reuse) appears
        as multiple rows with the same storage_id but disjoint lifetimes —
        those stay separate.

        Tensors with type WEIGHT/INPUT/LEAF are treated as alive for the
        entire run (birth=0, death=inf), so any other tensor sharing their
        storage automatically merges into them.

        Pass B (per-node aliasing): for every node, if any output tensor
        shares (device, storage_id) with any input tensor, merge. This
        is the same-time, same-op view check; with Pass A doing most of
        the work it now mainly catches view ops on intermediates that
        Pass A's lifetime sweep missed by margin.

        Allocator ops (aten::empty, etc.) have no inputs, so per-node
        aliasing never merges their output with a prior tensor — they
        stay distinct unless their lifetime overlaps with another
        tensor's on the same storage_id, which Pass A handles correctly.
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

        for node in node_map.values():
            s = node.args.get("start_ns")
            e = node.args.get("end_ns")
            if s is None: s = 0
            if e is None: e = s
            for tid in node.output_tensors:
                if tid not in birth or s < birth[tid]:
                    birth[tid] = s
                if tid not in death or e > death[tid]:
                    death[tid] = e
            for tid in node.input_tensors:
                # consumed during this node — extends death
                if tid not in death or e > death[tid]:
                    death[tid] = e
                # if no producer ever recorded, consumer-only tensor is
                # treated as alive from time 0
                birth.setdefault(tid, 0)

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
            bumped up from a victim view — view rows can over-state the
            underlying storage size when as_strided/expand creates an
            oversized numel through replication / non-contiguous strides.
            The keeper is chosen by the cluster-build loop to be the
            authoritative size source (the WEIGHT row if one exists,
            otherwise the smallest member of the cluster — which is a
            tight upper bound on a contiguous view of the storage).
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

        # ---- Pass A: lifetime-aware dedup grouped by (device, storage_id). ----
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
            # Sort by birth time. Sweep clusters: tensors whose lifetime
            # overlaps (b <= cluster_max_death) join the cluster.
            tids_sorted = sorted(tids, key=lambda t: lifetime(t)[0])
            clusters: list[list[int]] = []
            cluster_deaths: list[float] = []
            for tid in tids_sorted:
                b, d = lifetime(tid)
                # Find any open cluster this overlaps with.
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
            # Within each cluster, merge into a chosen anchor whose
            # size_bytes best matches the underlying storage. Prefer
            # WEIGHT/INPUT/LEAF rows (their size is set by the
            # parameter's actual allocation, not a viewed numel). Among
            # permanent rows we pick the *largest* — for a real param
            # buffer the row recording the contiguous, full-size view
            # has the buffer's true byte count, while smaller permanent
            # rows may be partial views (e.g. weight-norm splits).
            #
            # If no permanent in the cluster, use the smallest size as a
            # tight upper bound on the storage (views via as_strided /
            # expand can over-state via overlapping strides).
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
                    # Pure-intermediate cluster: take max as the best
                    # estimate of the underlying storage. The downside
                    # (overlapping as_strided views can over-state
                    # numel) is small in practice for SDXL-style traces;
                    # the alternative (min) under-counts real activations.
                    anchor = max(c, key=lambda t: tensor_map[t].size_bytes)
                for victim in c:
                    if victim == anchor:
                        continue
                    merge_into(anchor, victim)

        # Pass B: per-node aliasing. Nodes in id order (= temporal order).
        for node_id in sorted(node_map.keys()):
            node = node_map[node_id]
            for out_tid in list(node.output_tensors):
                if out_tid in node.input_tensors:
                    continue
                out_tensor = tensor_map.get(resolve(out_tid))
                if out_tensor is None:
                    continue
                out_sid = out_tensor.args.get("storage_id")
                if out_sid is None:
                    continue
                out_dev = out_tensor.args.get("device")
                for in_tid in node.input_tensors:
                    in_tensor = tensor_map.get(resolve(in_tid))
                    if in_tensor is None:
                        continue
                    if (in_tensor.args.get("storage_id") == out_sid
                            and in_tensor.args.get("device") == out_dev):
                        r_out = resolve(out_tid)
                        r_in = resolve(in_tid)
                        if r_out != r_in:
                            remap[r_out] = r_in
                        break

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
        """
        for node in node_map.values():
            outs = node.output_tensors
            ins = node.input_tensors

            is_alias = bool(outs) and all(t in ins for t in outs)
            is_dispatcher = self._loader_is_dispatcher(node, tensor_map)

            if not (is_alias or is_dispatcher):
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
        producers_by_tensor: dict[int, list[int]] = {}
        consumers_by_tensor: dict[int, list[int]] = {}

        for node in node_map.values():
            for tensor_id in node.output_tensors:
                producers_by_tensor.setdefault(tensor_id, []).append(node.id)
            for tensor_id in node.input_tensors:
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

    def load(self) -> Trace:
        bundle_dir, manifest = self._bundle_paths()

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
            self._apply_storage_aliasing(node_map, tensor_map)
            self._mark_implicit_inputs(node_map, tensor_map)
            self._annotate_alias_dispatcher_deps(node_map, tensor_map)
            if bool(self.args.get("add_temporal_data_control_edges", False)):
                self._add_temporal_data_control_edges(node_map)
            self._add_terminal_node(node_map)
            trace = Trace(self.id, self.name, self.log, node_map, tensor_map)
        else:
            raise Exception(f"[PytorchProfile] Unsupported graph_source: {graph_source}")

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
                    "[PytorchProfile] inject_schedule_path requires "
                    "graph_modifiers.inject_schedule to be importable. "
                    f"Underlying: {e}"
                )
            print(f"[PytorchProfile] injecting schedule from {inject_path_resolved}",
                  flush=True)
            inject_schedule_into_trace(
                trace, str(inject_path_resolved),
                bundle_dir=bundle_dir,
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
