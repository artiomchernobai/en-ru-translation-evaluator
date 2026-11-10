"""Streamlit entry point: python -m streamlit run src/ui/app.py."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.ui.views.files import render_files
from src.ui.views.review import render_review
from src.ui.views.settings import render_settings
from src.ui.views.sentence_alignment import render_alignment
from src.ui.components.jobs import render_controls, render_console
from src.ui.components.layout import fit_viewers

def configure_page() -> None:
    st.set_page_config(page_title="Text Pipeline · Workbench", page_icon="📄", layout="wide", initial_sidebar_state="collapsed")
    st.html("""
    <style>
    .block-container {padding:4.25rem 1.25rem 1rem;max-width:none;}
    [data-testid="stSidebar"] {display:none;}
    h1 {font-size:1.15rem !important;line-height:1.3 !important;padding:0 !important;margin:0 !important;}
    h3 {font-size:1rem !important;}
    [data-testid="stVerticalBlock"] {gap:0.65rem;}
    [data-testid="stRadio"] {margin-bottom:0.2rem;}
    [data-testid="stTextArea"] textarea {border-radius:4px;}
    [data-testid="stBottomBlockContainer"] {padding:0.35rem 1.25rem 0.65rem;}
    .st-key-right_actions {padding:0.5rem;}
    </style>
    """)


def main(root: Path = ROOT) -> None:
    configure_page()
    st.title("Text Pipeline · Workbench")
    page = st.radio("Навигация", ["Файлы", "Проверка", "Выравнивание", "Настройки"],
                    horizontal=True, label_visibility="collapsed", key="navigation")
    central, sidebar = st.columns([5.3,1.7],gap="medium")
    with central:
        if page=="Файлы": render_files(root)
        elif page=="Проверка": render_review(root)
        elif page=="Выравнивание": render_alignment(root)
        else: render_settings(root)
    with sidebar:
        render_controls(root)
    render_console(root)
    fit_viewers()


if __name__=="__main__":
    main()
