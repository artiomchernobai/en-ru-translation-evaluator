from pathlib import Path
import streamlit as st
from src.processing.pipeline.config import DOCUMENTS, SOURCE, STAGE_NAMES, TARGETS, folder
import importlib.util
from src.ui.file_browser import available_files,store_import,assign_to_raw

def render_settings(root: Path) -> None:
    ROOT = root
    st.subheader("Проект")
    st.code(str(ROOT), language=None)
    st.caption("В этой версии имена документов согласованы с твоим существующим пайплайном.")
    st.table([{"Документ": key, "Язык": value, "Вход": f"data/raw/{key}.txt"} for key, value in DOCUMENTS.items()])
    st.divider()
    st.subheader("Импорт файлов")
    upload = st.file_uploader("Добавить файл для просмотра", type=["txt", "csv", "json"])
    if st.button("Сохранить в data/imported", disabled=upload is None):
        try:
            stored = store_import(ROOT, upload.name, upload.getvalue())
            st.success(f"Сохранён: {str(stored.relative_to(ROOT))}")
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    imported = [path for path in available_files(ROOT) if "imported" in path.parts and path.suffix == ".txt"]
    if imported:
        st.caption("Чтобы использовать импортированный TXT как исходный текст пайплайна, явно назначьте ему документ. Существующий оригинал не перезаписывается.")
        file_sel = st.selectbox("Импортированный TXT", imported, format_func=lambda p:str(p.relative_to(ROOT)))
        destination = st.selectbox("Назначить документу", list(DOCUMENTS))
        if st.button("Копировать в data/raw"):
            try:
                new_path = assign_to_raw(ROOT, file_sel, destination)
                st.success(f"Создан: {str(new_path.relative_to(ROOT))}")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    st.divider()
    st.subheader("Подключённые компоненты")
    names = [
        "src.processing.preprocessing.cleaning", "src.processing.segmentation.postprocessor", "src.processing.preprocessing.normalization",
        "src.processing.segmentation.segmenter", "src.processing.segmentation.boundary_aware",
        "src.processing.preprocessing.export", "src.processing.alignment.corpus",
        "src.external_tools.vecalign_tools",
    ]
    for name in names:
        try:
            exists = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError):
            exists = False
        st.caption(f"{'✓' if exists else '—'} {name}")
    st.caption("Обнаружение модуля не гарантирует, что зависимости, модели и внешние исполняемые файлы установлены.")
