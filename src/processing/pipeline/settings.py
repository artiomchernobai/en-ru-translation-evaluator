"""Shared execution settings for CLI and GUI; defaults are identical."""
from copy import deepcopy
import json
from pathlib import Path
from .config import DOCUMENTS

DEFAULTS = {
    'documents': list(DOCUMENTS),
    'vecalign': {'max_size': 4, 'deletion_percentile': .1},
    'external_tools': {'backend': '', 'laser_python': '', 'vecalign_python': '',
                       'wsl_project_dir': '', 'vecalign_script': '', 'overlap_script': ''},
    'word_alignment': {'model': 'bert', 'device': 'cpu', 'layer': 8, 'method': 'itermax',
                       'min_similarity': .55, 'min_margin': .02, 'secondary_drop': .12},
    'segmentation': {'batch_size': 8, 'context_size': 200, 'max_length': 256, 'threshold': .5},
}


def read_settings(root):
    path = Path(root)/'data/pipeline_settings.json'
    raw = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    if not isinstance(raw, dict) or set(raw)-set(DEFAULTS): raise ValueError('Unknown pipeline settings')
    result = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if isinstance(result[key], dict):
            if not isinstance(value, dict) or set(value)-set(result[key]): raise ValueError(f'Unknown settings: {key}')
            result[key].update(value)
        else: result[key] = value
    external=result['external_tools']
    if any(not isinstance(v,str) for v in external.values()) or external['backend'] not in ('','native','wsl'):
        raise ValueError('Invalid external tool settings')
    v,w,s = (result[k] for k in ('vecalign','word_alignment','segmentation'))
    if not result['documents'] or any(d not in DOCUMENTS for d in result['documents']): raise ValueError('Invalid documents')
    if type(v['max_size']) is not int or not 2 <= v['max_size'] <= 8: raise ValueError('Vecalign max_size must be 2..8')
    from src.external_tools.vecalign_tools import validate_deletion_percentile
    v['deletion_percentile'] = validate_deletion_percentile(v['deletion_percentile'])
    if w['method'] not in ('inter','itermax','mwmf') or w['device'] not in ('cpu','cuda'): raise ValueError('Invalid word alignment method/device')
    if not isinstance(w['model'],str) or not w['model'].strip() or type(w['layer']) is not int or w['layer']<0: raise ValueError('Invalid encoder settings')
    for k in ('min_similarity','min_margin','secondary_drop'):
        if type(w[k]) not in (int,float) or not 0<=w[k]<=1: raise ValueError(f'Invalid {k}')
    if any(type(s[k]) is not int or s[k]<1 for k in ('batch_size','context_size','max_length')) or not 0<=s['threshold']<=1: raise ValueError('Invalid segmentation settings')
    return result
