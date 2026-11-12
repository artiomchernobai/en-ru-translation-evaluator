"""Shared evaluation settings, without importing COMET or Streamlit."""
import json
from pathlib import Path


def read_settings(root):
    path = Path(root) / 'data/cometkiwi_settings.json'
    settings = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(settings, dict):
        raise ValueError('Настройки COMETKiwi должны быть JSON-объектом')
    return settings
