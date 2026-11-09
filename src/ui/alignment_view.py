"""Vecalign's [source_indices]:[target_indices]:cost format and display rows."""
from __future__ import annotations

from pathlib import Path
from src.processing.alignment import parse_alignment as _parse_alignment, parse_indices as parse_ids


def parse_alignment(text: str) -> list[dict]:
    return [{"ru_ids": row["source_indices"], "en_ids": row["target_indices"], "cost": row["cost"]}
            for row in _parse_alignment(text)]


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
