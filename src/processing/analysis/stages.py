"""Project adapter for the offline review-candidate stage."""
from pathlib import Path
from src.processing.pipeline.config import SOURCE, TARGETS, folder
from src.processing.pipeline.io import check_file
from .io import collect_files, load_settings
from src.processing.evaluation.settings import read_settings, window_options, result_suffix


def stage_candidates(root: Path, docs: list[str], max_size: int) -> None:
    targets = [d for d in TARGETS if d in docs]
    if SOURCE not in docs or not targets:
        raise ValueError('Для поиска кандидатов выберите оригинал и хотя бы один перевод')
    settings = load_settings(root)
    strategy = settings.pop('window_strategy', window_options(read_settings(root))['strategy'])
    suffix = result_suffix(strategy)
    for target in targets:
        collect_files(check_file(folder(root, 'word_alignment') / f'{SOURCE}_{target}.json'),
                      check_file(folder(root, 'translation_quality') / f'{SOURCE}_{target}.cometkiwi{suffix}.json'),
                      folder(root, 'review_candidates') / f'{SOURCE}_{target}.candidates{suffix}.json', window_strategy=strategy, **settings)
