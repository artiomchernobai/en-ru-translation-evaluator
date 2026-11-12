"""Paths and document identities. No processing code or Streamlit dependencies."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = {
    "original-rus": "ru",
    "glenny-translation-en": "en",
    "pevear-translation-en": "en",
}
SOURCE = "original-rus"
TARGETS = ("glenny-translation-en", "pevear-translation-en")
STAGE_NAMES = {
    "clean": "Очистка",
    "segment": "Сегментация",
    "normalize": "Нормализация и экспорт",
    "overlap": "Группы предложений",
    "embed": "LASER — эмбеддинги",
    "align": "Vecalign — выравнивание",
    "word_align": "SimAlign — выравнивание слов",
    "corpus": "Корпус соответствий",
}
STAGES = tuple(STAGE_NAMES)


def folder(root: Path, category: str) -> Path:
    """Centralize paths; match the existing example pipeline layout."""
    suffixes = {
        "raw": "data/raw",
        "cleaned": "data/cleaned",
        "segmented": "data/segmented",
        "normalized": "data/normalized",
        "overlap_input": "data/overlaps/input",
        "overlap_output": "data/overlaps/output",
        "embeddings": "data/embeddings",
        "aligned": "data/aligned",
        "word_alignment": "data/word_alignment",
        "corpus": "data/corpus",
        "jobs": "data/ui_jobs",
        "imports": "data/imported",
    }
    return root / suffixes[category]
