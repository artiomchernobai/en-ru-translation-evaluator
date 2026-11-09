"""Shared Vecalign parsing and complete, monotonic coverage validation."""
from __future__ import annotations

import math
import re

ALIGNMENT = re.compile(r"^\s*\[([^\]]*)\]\s*:\s*\[([^\]]*)\]\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$")


def parse_indices(raw: str) -> list[int]:
    if not raw.strip():
        return []
    parts = raw.split(',')
    if any(not re.fullmatch(r'\s*\d+\s*', p) for p in parts):
        raise ValueError(f'Invalid indices: {raw}')
    indices = [int(p) for p in parts]
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(f'Indices must be unique and increasing: {raw}')
    return indices


def validate_alignment(items: list[dict], source_count: int | None = None,
                       target_count: int | None = None) -> None:
    next_index = {'source_indices': 0, 'target_indices': 0}
    for line, item in enumerate(items, 1):
        if not item['source_indices'] and not item['target_indices']:
            raise ValueError(f'Alignment line {line}: both sides are empty')
        if not math.isfinite(item['cost']):
            raise ValueError(f'Alignment line {line}: cost must be finite')
        for key in next_index:
            for index in item[key]:
                if index != next_index[key]:
                    raise ValueError(f'Alignment line {line}: {key} expected {next_index[key]}, got {index}; missing, repeated or reordered sentence')
                next_index[key] += 1
    for key, count in zip(next_index, (source_count, target_count)):
        if count is not None and next_index[key] != count:
            raise ValueError(f'Incomplete alignment: {key} covers {next_index[key]} sentences, expected {count}')


def parse_alignment(text: str) -> list[dict]:
    items = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = ALIGNMENT.fullmatch(line)
        if not match:
            raise ValueError(f'Invalid alignment at line {number}: {line[:100]}')
        items.append({'source_indices': parse_indices(match[1]),
                      'target_indices': parse_indices(match[2]), 'cost': float(match[3])})
    validate_alignment(items)
    return items
