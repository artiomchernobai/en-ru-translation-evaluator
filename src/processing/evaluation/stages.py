"""Application adapter; one loaded model serves all selected translation pairs."""
from pathlib import Path
from src.processing.pipeline.config import SOURCE, TARGETS, folder
from src.processing.pipeline.io import check_file
from .io import configured_backend, evaluate_files


def stage_cometkiwi(root: Path, docs: list[str], max_size: int) -> None:
    targets = [doc for doc in TARGETS if doc in docs]
    if SOURCE not in docs or not targets:
        raise ValueError('Для COMETKiwi выберите оригинал и хотя бы один перевод')
    pairs = []
    for target in targets:
        pairs.append((check_file(folder(root, 'normalized') / f'{SOURCE}.json'),
                      check_file(folder(root, 'normalized') / f'{target}.json'),
                      check_file(folder(root, 'aligned') / f'{SOURCE}_{target}.align'),
                      folder(root, 'translation_quality') / f'{SOURCE}_{target}.cometkiwi.json'))
    backend = configured_backend(root)
    for source, target, alignment, output in pairs:
        evaluate_files(source, target, alignment, output, backend=backend)
