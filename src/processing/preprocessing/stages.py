"""Cleaning and normalization/export stage adapters."""
from __future__ import annotations
from pathlib import Path
from src.processing.pipeline.config import DOCUMENTS, SOURCE, TARGETS, folder
from src.processing.pipeline.io import check_file, save_json

def stage_clean(root: Path, docs: list[str], max_size: int) -> None:
    from src.processing.preprocessing.cleaning import clean_text
    for doc in docs:
        src = check_file(folder(root, "raw") / f"{doc}.txt")
        target = folder(root, "cleaned") / f"{doc}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        text = clean_text(src.read_text(encoding="utf-8-sig"))
        target.write_text(text, encoding="utf-8")
        print(f"[Очистка] {doc}: {len(text)} символов -> {target.relative_to(root)}", flush=True)


def stage_normalize(root: Path, docs: list[str], max_size: int) -> None:
    try:
        from src.processing.preprocessing.normalization import TextNormalizer
        from src.processing.preprocessing.export import TextJsonTxtExporter
    except ImportError as error:
        raise RuntimeError("Нужны normalizer.TextNormalizer и src.processing.preprocessing.export") from error
    normalizer = TextNormalizer()
    exporter = TextJsonTxtExporter()
    for doc in docs:
        src = check_file(folder(root, "segmented") / f"{doc}.json")
        norm = folder(root, "normalized") / f"{doc}.json"
        txt = folder(root, "overlap_input") / f"{doc}.txt"
        norm.parent.mkdir(parents=True, exist_ok=True)
        txt.parent.mkdir(parents=True, exist_ok=True)
        normalizer.process_file(src, norm)
        exporter.export_to_txt_file(norm, txt)
        print(f"[Нормализация] {doc}: {norm.relative_to(root)}; {txt.relative_to(root)}", flush=True)
