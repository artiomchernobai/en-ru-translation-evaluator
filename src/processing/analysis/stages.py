"""Project adapter for the offline review-candidate stage."""
from pathlib import Path
from src.processing.pipeline.config import SOURCE, TARGETS, folder
from src.processing.pipeline.io import check_file
from .io import collect_files, load_settings


def stage_candidates(root: Path, docs: list[str], max_size: int) -> None:
    targets = [d for d in TARGETS if d in docs]
    if SOURCE not in docs or not targets:
        raise ValueError('Для поиска кандидатов выберите оригинал и хотя бы один перевод')
    settings = load_settings(root)
    for target in targets:
        collect_files(check_file(folder(root, 'word_alignment') / f'{SOURCE}_{target}.json'),
                      check_file(folder(root, 'translation_quality') / f'{SOURCE}_{target}.cometkiwi.json'),
                      folder(root, 'review_candidates') / f'{SOURCE}_{target}.candidates.json', **settings)
