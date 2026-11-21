"""Evaluate prepared windows with an injected, already loaded model adapter."""
import math
from collections import Counter
from copy import deepcopy


def evaluate_windows(windows, backend):
    output = deepcopy(windows)
    active = [w for w in output if w['source']['text'] and w['target']['text']]
    unique = {(w['source']['text'], w['target']['text']): None for w in active}
    results = backend.predict([{'src': src, 'mt': mt} for src, mt in unique]) if unique else []
    if len(results) != len(unique):
        raise ValueError('Backend result count does not match windows')
    lookup = dict(zip(unique, results))
    for window in active:
        result = deepcopy(lookup[(window['source']['text'], window['target']['text'])])
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
            'inference': {'input_windows': len(active), 'unique_model_inputs': len(unique)},
            'summary': dict(Counter(w['verdict']['status'] for w in output)), 'windows': output}
