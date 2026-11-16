"""A dropdown and previous/next controls sharing one selected fragment."""
import streamlit as st


def _move(key, delta, count):
    st.session_state[key] = max(0, min(count-1, st.session_state.get(key, 0)+delta))


def fragment_selector(items, *, key, label, identity, format_func, indicator, scope, show_buttons=True):
    ids = [identity(item) for item in items]
    previous = st.session_state.get(key+'_ids', [])
    old = st.session_state.get(key, 0)
    same_scope = st.session_state.get(key+'_scope') == scope
    if previous != ids or not same_scope:
        selected_id = previous[old] if same_scope and 0 <= old < len(previous) else None
        st.session_state[key] = ids.index(selected_id) if selected_id in ids else 0
    st.session_state[key+'_ids'] = ids
    st.session_state[key+'_scope'] = scope
    selected = st.selectbox(label, list(range(len(items))), format_func=format_func, key=key)
    if show_buttons:
        fragment_navigation(items, selected=selected, key=key, indicator=indicator)
    return selected


def fragment_navigation(items, *, selected, key, indicator):
    left, middle, right = st.columns([2, 3, 2])
    left.button('Предыдущий фрагмент', key=key+'_previous', disabled=selected == 0,
                on_click=_move, args=(key, -1, len(items)), use_container_width=True)
    middle.caption(f'{selected+1}/{len(items)} · {indicator(items[selected])}')
    right.button('Следующий фрагмент', key=key+'_next', disabled=selected == len(items)-1,
                 on_click=_move, args=(key, 1, len(items)), use_container_width=True)
