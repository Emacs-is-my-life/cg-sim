from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return base_dir / p


def parse_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(value)


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def parse_shape(value: str | None):
    if value is None or value == "":
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def tensor_type_from_row(row: dict[str, str]) -> str:
    tensor_kind = row.get("tensor_kind", "")
    producer_count = int(row.get("producer_count") or 0)

    if tensor_kind == "WEIGHT":
        return "WEIGHT"
    if producer_count == 0:
        return "INPUT"
    return "INTERMEDIATE"


def normalize_device(device: str | None) -> str:
    if not device:
        return "cpu"
    return str(device).lower()


def profile_id_sort_key(profile_id: str) -> tuple[str, int]:
    prefix = profile_id[:1]
    suffix = profile_id[1:]
    if suffix.isdigit():
        return (prefix, int(suffix))
    return (prefix, -1)


def parse_dot_attrs(attr_text: str | None) -> dict[str, str]:
    if not attr_text:
        return {}

    attrs: dict[str, str] = {}
    pos = 0
    length = len(attr_text)

    while pos < length:
        while pos < length and attr_text[pos] in " \t\r\n;,":
            pos += 1
        if pos >= length:
            break

        key_start = pos
        while pos < length and attr_text[pos] not in " \t\r\n=":
            pos += 1
        key = attr_text[key_start:pos]

        while pos < length and attr_text[pos] in " \t\r\n":
            pos += 1
        if pos >= length or attr_text[pos] != "=":
            while pos < length and attr_text[pos] not in ";,":
                pos += 1
            continue

        pos += 1
        while pos < length and attr_text[pos] in " \t\r\n":
            pos += 1

        if pos < length and attr_text[pos] == '"':
            pos += 1
            value_chars = []
            while pos < length:
                ch = attr_text[pos]
                if ch == "\\" and pos + 1 < length:
                    value_chars.append(attr_text[pos + 1])
                    pos += 2
                    continue
                if ch == '"':
                    pos += 1
                    break
                value_chars.append(ch)
                pos += 1
            value = "".join(value_chars)
        else:
            value_start = pos
            while pos < length and attr_text[pos] not in " \t\r\n;,":
                pos += 1
            value = attr_text[value_start:pos]

        attrs[key] = value

    return attrs


def parse_pytorch_dot(dot_path: Path) -> tuple[dict[str, dict[str, str]], list[tuple[str, str, dict[str, str]]]]:
    node_pattern = re.compile(r'^\s+"([^"]+)"\s*\[(.*)\]\s*;?\s*$')
    edge_pattern = re.compile(r'^\s+"([^"]+)"\s*->\s*"([^"]+)"(?:\s*\[(.*)\])?\s*;\s*$')

    vertices: dict[str, dict[str, str]] = {}
    edges: list[tuple[str, str, dict[str, str]]] = []

    with open(dot_path, "r") as f:
        for line in f:
            edge_match = edge_pattern.match(line)
            if edge_match:
                src, dst, attr_text = edge_match.groups()
                edges.append((src, dst, parse_dot_attrs(attr_text)))
                continue

            node_match = node_pattern.match(line)
            if node_match:
                vertex_id, attr_text = node_match.groups()
                vertices[vertex_id] = parse_dot_attrs(attr_text)

    return vertices, edges
