"""Bidirectional candidate reader; candidate picks update the shared dropdown."""
from pathlib import Path
import streamlit as st
import streamlit.components.v2 as components

ASSETS = Path(__file__).with_name('assets')
_component = components.component('candidate_text_review',
    html='<div class="candidate-reader"><div class="candidate-panes"></div></div>',
    css=(ASSETS/'candidate_review.css').read_text(encoding='utf-8'),
    js=(ASSETS/'candidate_review.js').read_text(encoding='utf-8'))


def _pick(key, identity, ids):
    event = st.session_state.get(key, {}).get('pick')
    if event and event.get('identity') == identity and event.get('id') in ids:
        st.session_state['candidate_index'] = ids.index(event['id'])


def show_candidate_review(payload, *, ids, key='candidate_reader'):
    _component(data=payload, key=key, height='content', on_pick_change=lambda: _pick(key, payload['identity'], ids))
