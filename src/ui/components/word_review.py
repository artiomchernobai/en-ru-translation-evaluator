"""Local, build-free Streamlit component. Text is passed as data, never HTML."""
from pathlib import Path
import streamlit.components.v2 as components

ASSETS = Path(__file__).with_name('assets')
_component = components.component(
    'translation_word_review', html=(ASSETS/'word_review.html').read_text(encoding='utf-8'),
    css=(ASSETS/'word_review.css').read_text(encoding='utf-8'), js=(ASSETS/'word_review.js').read_text(encoding='utf-8'))


def show_word_review(payload: dict, key: str) -> None:
    _component(data=payload, key=key, height='content')
