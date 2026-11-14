"""Merge independent screening signals; retain every group for future semantics."""
from copy import deepcopy
import math
from .inputs import validate_inputs
from .locations import whole_sentences
from .rules import DEFAULT_GLOSSARY, lexical_signals


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.floor(fraction * (len(ordered)-1))]


def collect_candidates(words, comet, *, low_quantile=.1, high_quantile=.9, glossary=None):
    for value in (low_quantile, high_quantile):
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Quantiles must be between 0 and 1')
    if low_quantile >= high_quantile:
        raise ValueError('Low quantile must precede high quantile')
    validate_inputs(words, comet)
    glossary = DEFAULT_GLOSSARY if glossary is None else glossary
    if not isinstance(glossary, list) or any(not isinstance(entry, dict) or
            any(not isinstance(entry.get(key), str) or not entry[key] for key in ('id', 'source', 'target')) for entry in glossary):
        raise ValueError('Glossary requires id/source/target regex strings')
    if len({entry['id'] for entry in glossary}) != len(glossary):
        raise ValueError('Glossary concept IDs must be unique')
    blocks, windows = words['blocks'], comet['windows']
    low = percentile([w['verdict']['score'] for w in windows if w['verdict']['status'] == 'scored'], low_quantile)
    high = percentile([b['vecalign_cost'] for b in blocks if b['source']['sentence_indices'] and b['target']['sentence_indices']], high_quantile)
    candidates, tasks = [], []
    for block, window in zip(blocks, windows):
        index = block['block_id']
        signals = lexical_signals(words, block, glossary)
        def signal(code, message, weight):
            signals.append({'code': code, 'message': message, 'weight': weight,
                            'source_spans': whole_sentences(words['source'], block['source']['sentence_indices']),
                            'target_spans': whole_sentences(words['target'], block['target']['sentence_indices'])})
        if not block['source']['sentence_indices'] or not block['target']['sentence_indices']:
            signal('sentence_gap', 'Пустая сторона Vecalign: возможны отсутствие текста или ошибка выравнивания.', 40)
        elif high is not None and block['vecalign_cost'] > high:
            signal('high_vecalign_cost', 'Стоимость выше верхнего перцентиля этой пары.', 20)
        if window['verdict']['status'] != 'scored':
            signal('comet_unscored', 'COMETKiwi не оценил окно; нулевой балл не подставляется.', 10)
        elif low is not None and window['verdict']['score'] < low:
            signal('low_comet_window', 'Оценка всего окна ниже нижнего перцентиля этой пары; центральная ошибка не установлена.', 20)
        candidate_id = f'group-{index}'
        if signals:
            candidates.append({'id': candidate_id, 'group_index': index, 'status': 'needs_review',
                               'priority': min(100, sum(s['weight'] for s in signals)),
                               'priority_is_probability': False, 'signals': signals,
                               'source_spans': whole_sentences(words['source'], block['source']['sentence_indices']),
                               'target_spans': whole_sentences(words['target'], block['target']['sentence_indices']),
                               'context_group_indices': list(window['group_indices']),
                               'vecalign_cost': block['vecalign_cost'], 'comet_window': deepcopy(window['verdict'])})
        # High COMET and existing SimAlign links do not exclude local meaning errors.
        tasks.append({'id': f'semantic-{index}', 'center_group_index': index,
                      'status': 'not_analyzed', 'candidate_ids': [candidate_id] if signals else [],
                      'window': deepcopy(window), 'word_links': [
                          {'group_index': i, 'status': blocks[i]['word_alignment']['status'],
                           'method': words['selected_method'],
                           'links': deepcopy(blocks[i]['word_alignment'].get('methods', {}).get(words['selected_method'], []))}
                          for i in window['group_indices']]})
    candidates.sort(key=lambda item: (-item['priority'], item['group_index']))
    return {'schema_version': 1, 'analysis_version': 'candidate_rules_ru_en_v1',
            'semantic_analysis': 'not_run', 'source_document': words['source']['document_id'],
            'target_document': words['target']['document_id'],
            'offsets': deepcopy(words['offsets']), 'source': deepcopy(words['source']), 'target': deepcopy(words['target']),
            'selected_word_method': words['selected_method'],
            'configuration': {'low_comet_quantile': low_quantile, 'high_vecalign_quantile': high_quantile,
                              'low_comet_threshold': low, 'high_vecalign_threshold': high,
                              'glossary': deepcopy(glossary)},
            'summary': {'groups': len(blocks), 'candidate_groups': len(candidates),
                        'semantic_tasks': len(tasks), 'semantic_groups_checked': 0,
                        'signal_count': sum(len(c['signals']) for c in candidates)},
            'candidates': candidates, 'semantic_tasks': tasks}
