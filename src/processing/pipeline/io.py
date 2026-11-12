"""File prerequisites and JSON output shared by processing stages."""
import json
from pathlib import Path

def check_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    return path


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
