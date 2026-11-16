"""Shared evaluation settings, without importing COMET or Streamlit."""
import json
from pathlib import Path


def read_settings(root):
    path = Path(root) / 'data/cometkiwi_settings.json'
    settings = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(settings, dict):
        raise ValueError('Настройки COMETKiwi должны быть JSON-объектом')
    return settings


def window_options(settings):
    strategy = settings.get('window_strategy', 'fixed')
    if strategy not in ('fixed', 'adaptive'):
        raise ValueError('Unknown window strategy')
    extra = settings.get('max_extra_groups', 6)
    if type(extra) is not int or not 0 <= extra <= 100:
        raise ValueError('max_extra_groups must be between 0 and 100')
    return {'strategy': strategy, 'max_extra_groups': extra}


def result_suffix(strategy):
    if strategy not in ('fixed', 'adaptive'):
        raise ValueError('Unknown window strategy')
    return '.adaptive' if strategy == 'adaptive' else ''
