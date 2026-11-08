"""Vecalign's [source_indices]:[target_indices]:cost format and display rows."""
from __future__ import annotations

import re
from pathlib import Path

ALIGNMENT = re.compile(r"^\s*\[([^\]]*)\]\s*:\s*\[([^\]]*)\]\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")


def parse_ids(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_alignment(text: str) -> list[dict]:
    result = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        m = ALIGNMENT.fullmatch(line)
        if not m:
            raise ValueError(f"Некорректная строка файла .align: {number}: {line[:100]}")
        result.append({"ru_ids": parse_ids(m[1]), "en_ids": parse_ids(m[2]), "cost": float(m[3])})
    return result


def sentences(path: Path) -> list[str]:
    # Preserve internal blank lines: indices must match the Vecalign input EXACTLY.
    return path.read_text(encoding="utf-8-sig").splitlines()


def _fragments(lines: list[str], ids: list[int]) -> tuple[str, bool]:
    if not ids:
        return "∅", False
    bad = any(i < 0 or i >= len(lines) for i in ids)
    parts = [lines[i] for i in ids if 0 <= i < len(lines)]
    return "\n".join(parts), bad


def expand_alignment(items: list[dict], source: list[str], target: list[str]) -> list[dict]:
    result = []
    for index, item in enumerate(items):
        ru, ru_bad = _fragments(source, item["ru_ids"])
        en, en_bad = _fragments(target, item["en_ids"])
        result.append({
            "№": index,
            "RU индексы": ", ".join(map(str, item["ru_ids"])),
            "Оригинал": ru,
            "EN индексы": ", ".join(map(str, item["en_ids"])),
            "Перевод": en,
            "Стоимость": item["cost"],
            "Пропуск": not item["ru_ids"] or not item["en_ids"],
            "Индекс вне диапазона": ru_bad or en_bad,
        })
    return result
