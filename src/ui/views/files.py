from pathlib import Path
import streamlit as st
from src.processing.pipeline.config import DOCUMENTS, SOURCE, STAGE_NAMES, TARGETS, folder
import json
from src.ui.file_browser import available_files, safe_file, file_details, parse_csv_preview, parse_json

def show_viewer(root: Path, key: str, options: list[str], default_index: int = 0) -> None:
    with st.container(border=False, key=f"viewer_{key}"):
        chosen = st.selectbox("Файл", ["— Выберите файл —"] + options,
                              key=f"file_{key}", index=default_index, label_visibility="collapsed")
        try:
            path = None if chosen.startswith("—") else safe_file(root, chosen)
            details = file_details(path) if path else None
            content = details["content"] if details else ""
            fmt = details["type"] if details else ".txt"
            mode = st.radio("Режим", ["Структура", "Исходный текст"], horizontal=True,
                            key=f"mode_{key}") if fmt == ".json" else None
            with st.container(key=f"viewer_body_{key}", height=500, border=False):
                if fmt == ".csv":
                    _, rows, truncated = parse_csv_preview(content)
                    if rows:
                        st.dataframe(rows, width="stretch", height="stretch", hide_index=True)
                    else:
                        st.text_area("CSV", content[:200_000], height="stretch", disabled=True,
                                     key=f"content_{key}", label_visibility="collapsed")
                elif fmt == ".json" and mode == "Структура":
                    st.json(parse_json(content), expanded=False)
                else:
                    st.text_area("Текст", content[:200_000], height="stretch", disabled=True,
                                 key=f"content_{key}_{chosen}_{path.stat().st_mtime_ns if path else 0}", label_visibility="collapsed")
            if path:
                with st.popover("Файл · сведения и скачивание", use_container_width=True):
                    st.caption(f"{details['lines']} строк · {details['chars']:,} символов · {details['bytes']:,} байт")
                    st.download_button("Скачать файл", data=path.read_bytes(), file_name=path.name,
                                       key=f"download_{key}", use_container_width=True)
                    if len(content)>200_000 and mode != 'Структура':
                        st.caption('Показаны первые 200 000 символов; скачивание содержит полный файл.')
                    if fmt == '.csv' and truncated:
                        st.caption('Показаны первые 500 строк CSV.')
        except (ValueError, OSError, UnicodeError) as exc:
            st.error(f"Не удалось открыть файл: {exc}")


def render_files(root: Path) -> None:
    files = [str(path.relative_to(root)) for path in available_files(root)]
    left, right = st.columns(2, gap="medium")
    with left:
        show_viewer(root,"a", files)
    with right:
        show_viewer(root,"b", files, default_index=min(2, len(files)))
