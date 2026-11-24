"""Final pipeline adapter; selection of work filenames stays outside the judge."""
from pathlib import Path
from src.processing.pipeline.config import SOURCE,TARGETS,folder
from src.processing.pipeline.io import check_file
from src.processing.analysis.io import load_settings as candidate_settings
from src.processing.evaluation.settings import read_settings as comet_settings, result_suffix
from .io import read_settings, configured_judge, analyze_file


def stage_mqm(root: Path, docs: list[str], max_size: int) -> None:
    targets=[doc for doc in TARGETS if doc in docs]
    if SOURCE not in docs or not targets:raise ValueError('Для MQM выберите оригинал и хотя бы один перевод')
    strategy=candidate_settings(root).get('window_strategy',comet_settings(root).get('window_strategy','fixed'))
    suffix=result_suffix(strategy)
    pairs=[(check_file(folder(root,'review_candidates')/f'{SOURCE}_{target}.candidates{suffix}.json'),
            folder(root,'mqm_reports')/f'{SOURCE}_{target}.mqm{suffix}.json') for target in targets]
    settings=read_settings(root)
    backend=configured_judge(settings)
    partial=False
    for source,output in pairs:
        result=analyze_file(source,output,backend=backend,scope=settings.get('scope','candidates'),resume=settings.get('resume',True),window_strategy=strategy,max_fragments=settings.get('max_fragments',0),min_priority=settings.get('min_priority',0),
                            corpus_path=check_file(folder(root,'corpus')/(source.name.split('.candidates')[0]+'.json')))
        partial=partial or result['status']!='complete'
    if partial:raise RuntimeError('MQM завершён частично; результаты сохранены. Проверьте сервер/цитаты и повторите этап.')


def analyze_candidate(root, group_index, target, *, strategy=None):
    """Assess exactly the chosen window; never overwrite the batch report."""
    if target not in TARGETS or type(group_index) is not int or group_index<0:
        raise ValueError('Invalid selected candidate')
    strategy=strategy or candidate_settings(root).get('window_strategy',comet_settings(root).get('window_strategy','fixed'))
    suffix=result_suffix(strategy)
    pair=f'{SOURCE}_{target}'
    settings=read_settings(root)
    result=analyze_file(check_file(folder(root,'review_candidates')/f'{pair}.candidates{suffix}.json'),
        folder(root,'mqm_reports')/f'{pair}.mqm{suffix}.selected-{group_index}.json',
        backend=configured_judge(settings),scope='candidates',resume=settings.get('resume',True),
        window_strategy=strategy,group_indices=[group_index],max_fragments=1,
        corpus_path=check_file(folder(root,'corpus')/f'{pair}.json'))
    if result['status']!='complete':raise RuntimeError('Выбранный фрагмент обработан частично; отчёт и диагностика сохранены')
    return result
