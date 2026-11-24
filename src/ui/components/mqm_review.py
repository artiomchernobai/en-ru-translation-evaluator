"""Reuse the safe two-pane reader with MQM-specific selection events."""
import streamlit as st
from .candidate_review import _component


def pick_mqm(key, identity, picks):
    event = st.session_state.get(key, {}).get('pick')
    if event and event.get('identity') == identity and event.get('id') in picks:
        index, finding = picks[event['id']]
        st.session_state['mqm_fragment'] = index
        st.session_state['mqm_selected_finding'] = finding


def show_mqm_review(payload, picks, *, key='mqm_reader'):
    _component(data=payload, key=key, height='content', on_pick_change=lambda: pick_mqm(key, payload['identity'], picks))
