"""Read-only TXT/CSV/JSON file inspection. The UI itself never writes these files."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ALLOWED = {".txt", ".csv", ".json", ".jsonl", ".align"}
MAX_BYTES = 15 * 1024 * 1024


def available_files(root: Path) -> list[Path]:
    data = root / "data"
    if not data.is_dir():
        return []
    return sorted(
        (p for p in data.rglob("*")
         if p.is_file() and p.suffix.lower() in ALLOWED
         and "ui_jobs" not in p.parts),
        key=lambda x: str(x.relative_to(root)).lower(),
    )


def safe_file(root: Path, relative: str) -> Path:
    """Only files under the project's data folder may be opened."""
    base = (root / "data").resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(base):
        raise ValueError("Допускаются только файлы внутри data/.")
    if p.suffix.lower() not in ALLOWED:
        raise ValueError(f"Неподдерживаемый формат: {p.suffix}")
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.stat().st_size > MAX_BYTES:
        raise ValueError("Файл превышает 15 МБ — используйте отдельный анализатор.")
    return p


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def file_details(path: Path) -> dict:
    content = read_text(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "lines": len(content.splitlines()),
        "chars": len(content),
        "content": content,
        "type": path.suffix.lower(),
    }


def parse_csv_preview(content: str, max_rows: int = 500) -> tuple[list[str], list[dict], bool]:
    """Keep UI responsive even with large text CSVs."""
    reader = csv.DictReader(io.StringIO(content))
    rows = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            return list(reader.fieldnames or []), rows, True
        rows.append(row)
    return list(reader.fieldnames or []), rows, False


def parse_json(content: str):
    return json.loads(content)


def store_import(root: Path, filename: str, content: bytes) -> Path:
    """A user upload is saved in a dedicated sandbox, never directly to raw."""
    name = Path(filename.replace("\\", "/")).name
    if not name or name in {".", ".."} or Path(name).suffix.lower() not in {".txt", ".csv", ".json"}:
        raise ValueError("Для импорта подходят .txt, .csv, .json")
    if len(content) > MAX_BYTES:
        raise ValueError("Файл превышает 15 МБ")
    target_dir = root / "data" / "imported"
    target_dir.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(name).stem, Path(name).suffix
    target = target_dir / name
    index = 2
    while target.exists():
        target = target_dir / f"{stem}_{index}{suffix}"
        index += 1
    target.write_bytes(content)
    return target


def assign_to_raw(root: Path, imported: Path, document: str) -> Path:
    """Explicit action, TXT only, no implicit replacement of source texts."""
    from .config import DOCUMENTS, folder
    if document not in DOCUMENTS:
        raise ValueError("Неизвестный идентификатор документа")
    src = safe_file(root, str(imported.relative_to(root)))
    if src.suffix.lower() != ".txt":
        raise ValueError("Для исходного корпуса нужен файл TXT")
    destination = folder(root, "raw") / f"{document}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"{destination.name} уже существует — исходник не перезаписан")
    destination.write_bytes(src.read_bytes())
    return destination
