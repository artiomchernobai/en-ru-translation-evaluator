"""Stage dispatch shared by the UI worker and other callers; no Streamlit."""
from pathlib import Path
from .config import DOCUMENTS, SOURCE, STAGES, TARGETS
from src.external_tools.vecalign_tools import DEFAULT_DELETION_PERCENTILE, validate_deletion_percentile
from src.processing.preprocessing.stages import stage_clean, stage_normalize
from src.processing.segmentation.stages import stage_segment
from src.processing.alignment.stages import stage_overlap, stage_embed, stage_align, stage_word_align, stage_corpus

from src.processing.evaluation.stages import stage_cometkiwi

from src.processing.analysis.stages import stage_candidates

FUNCTIONS = {
    "clean": stage_clean,
    "segment": stage_segment,
    "normalize": stage_normalize,
    "overlap": stage_overlap,
    "embed": stage_embed,
    "align": stage_align,
    "word_align": stage_word_align,
    "cometkiwi": stage_cometkiwi,
    "candidates": stage_candidates,
    "corpus": stage_corpus,
}


def run_stage(root: Path, stage: str, docs: list[str], max_size: int,
              del_percentile_frac: float = DEFAULT_DELETION_PERCENTILE) -> None:
    del_percentile_frac = validate_deletion_percentile(del_percentile_frac)
    if stage not in (*STAGES, "all"):
        raise ValueError(f"Неизвестный этап: {stage}")
    if not docs or any(doc not in DOCUMENTS for doc in docs):
        raise ValueError("Недопустимый список документов")
    if stage == "all" and (SOURCE not in docs or not any(d in docs for d in TARGETS)):
        raise ValueError("Полный пайплайн требует оригинал и хотя бы один перевод")
    execution = STAGES if stage == "all" else (stage,)
    for step in execution:
        print(f"\n{'=' * 20} {step.upper()} {'=' * 20}", flush=True)
        if step == "align":
            FUNCTIONS[step](root, docs, max_size, del_percentile_frac=del_percentile_frac)
        else:
            FUNCTIONS[step](root, docs, max_size)
    print("\nВсе выбранные этапы успешно завершены.", flush=True)
