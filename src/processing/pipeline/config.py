"""Paths and document identities. No processing code or Streamlit dependencies."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
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

# CLI paths and defaults share the same registry as the UI.
BASE_DIR = ROOT
TARGET_DOCUMENTS = list(TARGETS)
RAW_DIR = folder(ROOT, "raw")
OUTPUT_DIR = folder(ROOT, "segmented")
NORMALIZED_DIR = folder(ROOT, "normalized")
TRAINING_DIR = ROOT / "data" / "training"
OVERLAP_INPUT_DIR = folder(ROOT, "overlap_input")
OVERLAP_OUTPUT_DIR = folder(ROOT, "overlap_output")
EMBEDDINGS_DIR = folder(ROOT, "embeddings")
ALIGNED_DIR = folder(ROOT, "aligned")
CORPUS_DIR = folder(ROOT, "corpus")
WORD_ALIGNMENT_DIR = folder(ROOT, "word_alignment")
