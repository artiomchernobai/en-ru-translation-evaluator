"""Application adapter; one loaded model serves all selected translation pairs."""
from pathlib import Path
import os
import subprocess
from src.processing.pipeline.config import SOURCE, TARGETS, folder
from src.processing.pipeline.io import check_file
from .io import configured_backend, evaluate_files
from .settings import read_settings


def selected_pairs(root: Path, docs: list[str]):
    targets = [doc for doc in TARGETS if doc in docs]
    if SOURCE not in docs or not targets:
        raise ValueError('Для COMETKiwi выберите оригинал и хотя бы один перевод')
    pairs = []
    for target in targets:
        pairs.append((check_file(folder(root, 'normalized') / f'{SOURCE}.json'),
                      check_file(folder(root, 'normalized') / f'{target}.json'),
                      check_file(folder(root, 'aligned') / f'{SOURCE}_{target}.align'),
                      folder(root, 'translation_quality') / f'{SOURCE}_{target}.cometkiwi.json'))
    return pairs


def stage_cometkiwi(root: Path, docs: list[str], max_size: int) -> None:
    pairs = selected_pairs(root, docs)
    targets = [doc for doc in TARGETS if doc in docs]
    settings = read_settings(root)
    interpreter = settings.get('python') or os.environ.get('COMETKIWI_PYTHON')
    if interpreter:
        python = Path(interpreter).expanduser()
        if not python.is_absolute():
            python = root / python
        if not python.is_file():
            raise FileNotFoundError(f'Python окружения COMETKiwi не найден: {python}')
        code_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONPATH'] = str(code_root) + os.pathsep + env.get('PYTHONPATH', '')
        command = [str(python.absolute()), '-u', '-m', 'src.processing.evaluation.worker',
                   '--root', str(root.resolve()), '--documents', SOURCE, *targets]
        print(f'[COMETKiwi] Отдельное окружение: {python}', flush=True)
        subprocess.run(command, cwd=code_root, env=env, check=True)
        return
    evaluate_pairs(root, pairs)


def evaluate_pairs(root, pairs):
    """Used inside the chosen environment; never launches another interpreter."""
    backend = configured_backend(root)
    for source, target, alignment, output in pairs:
        evaluate_files(source, target, alignment, output, backend=backend)
