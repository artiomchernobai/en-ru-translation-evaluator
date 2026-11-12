"""Evaluate prepared windows with an injected, already loaded model adapter."""
import math
from collections import Counter
from copy import deepcopy


def evaluate_windows(windows, backend):
    output = deepcopy(windows)
    active = [w for w in output if w['source']['text'] and w['target']['text']]
    results = backend.predict([{'src': w['source']['text'], 'mt': w['target']['text']} for w in active]) if active else []
    if len(results) != len(active):
        raise ValueError('Backend result count does not match windows')
    for window, result in zip(active, results):
        if result.get('status') not in ('scored', 'too_long'):
            raise ValueError('Unknown evaluation status')
        if result['status'] == 'scored' and (not isinstance(result.get('score'), (int, float)) or not math.isfinite(result['score'])):
            raise ValueError('Invalid model score')
        window['verdict'] = {**result, 'scope': 'entire_window',
                             'interpretation': 'higher_is_better', 'is_error_probability': False}
    for window in output:
        if 'verdict' not in window:
            window['verdict'] = {'status': 'empty_side', 'score': None, 'scope': 'entire_window'}
    return {'schema_version': 1, 'model': deepcopy(backend.metadata),
            'summary': dict(Counter(w['verdict']['status'] for w in output)), 'windows': output}
