"""One review unit per identical pair of sentence windows, with all its signals."""
from copy import deepcopy
import hashlib
import json


def window_key(window):
    return tuple(tuple(window[side]['sentence_indices']) for side in ('source', 'target'))


def fragment_id(window):
    value = json.dumps(window_key(window), separators=(',', ':')).encode()
    return 'fragment-' + hashlib.sha256(value).hexdigest()[:16]


def review_fragments(data, *, scope='candidates', min_priority=0):
    candidates = data['candidates']
    by_group = {c['group_index']: c for c in candidates}
    selected = [c for c in candidates if c['priority'] >= min_priority]
    anchors = [c['group_index'] for c in selected] if scope == 'candidates' else range(len(data['semantic_tasks']))
    units = {}
    for index in anchors:
        window = data['semantic_tasks'][index]['window']
        key = window_key(window)
        if key not in units:
            unit = deepcopy(by_group.get(index, {
                'id': f'group-{index}', 'group_index': index, 'priority': 0,
                'source_spans': [], 'target_spans': []}))
            unit.update(fragment_id=fragment_id(window), anchor_group_indices=[], candidate_ids=[], signals=[])
            units[key] = unit
        units[key]['anchor_group_indices'].append(index)
        units[key]['priority'] = max(units[key]['priority'], by_group.get(index, {}).get('priority', 0))
    for unit in units.values():
        window = data['semantic_tasks'][unit['group_index']]['window']
        groups = set(window['group_indices'])
        seen = set()
        for candidate in candidates:
            if candidate['group_index'] not in groups:
                continue
            unit['candidate_ids'].append(candidate['id'])
            for signal in candidate['signals']:
                key = json.dumps(signal, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    unit['signals'].append(deepcopy(signal))
                    seen.add(key)
        unit['context_group_indices'] = list(window['group_indices'])
    return list(units.values())
