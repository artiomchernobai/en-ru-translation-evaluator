"""Streamlit workbench. Run from the project root:

    python -m streamlit run src/ui/app.py

All UI-only code lives in src/ui. Processing remains in the existing project.
"""
from __future__ import annotations

import csv
import io
import importlib.util
import json
from pathlib import Path
import sys

# Streamlit runs app.py as a script. Make sibling imports work independent of cwd.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.processing.alignment import validate_alignment
from src.ui.alignment_view import expand_alignment, parse_alignment, sentences
from src.ui.config import DOCUMENTS, SOURCE, STAGE_NAMES, TARGETS, folder
from src.ui.file_browser import (
    available_files, assign_to_raw, file_details, parse_csv_preview,
    parse_json, safe_file, store_import,
)
from src.ui.jobs import cancel_job, read_status, start_job, tail_log

st.set_page_config(page_title="Text Pipeline · Workbench", page_icon="📄", layout="wide", initial_sidebar_state="collapsed")

# CSS is exclusively cosmetic. The bottom dock uses Streamlit's native st.bottom.
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-left: 2rem; padding-right: 2rem; padding-bottom: 2rem; max-width: none;}
[data-testid="stSidebar"] {display: none;}
h1 {font-size: 1.65rem !important;}
h3 {font-size: 1.05rem !important;}
.st-key-right_actions {position: relative;}
</style>
""", unsafe_allow_html=True)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def show_viewer(title: str, key: str, options: list[str], default_index: int = 0) -> None:
    with st.container(border=True, key=f"viewer_{key}"):
        st.subheader(title)
        chosen = st.selectbox("Файл", ["— Выберите файл —"] + options,
                              key=f"file_{key}", index=default_index)
        try:
            path = None if chosen.startswith("—") else safe_file(ROOT, chosen)
            details = file_details(path) if path else None
            content = details["content"] if details else ""
            st.caption(f"{details['lines']} строк · {details['chars']:,} символов · {details['bytes']:,} байт"
                       if details else "Выберите файл из каталога data/.")
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
                st.download_button("Скачать файл", data=path.read_bytes(), file_name=path.name,
                                   key=f"download_{key}", use_container_width=True)
                if len(content) > 200_000 and fmt != ".csv" and mode != "Структура":
                    st.caption("Показаны первые 200 000 символов. Скачивание содержит полный файл.")
                if fmt == ".csv" and truncated:
                    st.caption("Показаны первые 500 строк. Скачивание содержит полный файл.")
        except (ValueError, OSError, UnicodeError) as exc:
            st.error(f"Не удалось открыть файл: {exc}")


def render_files() -> None:
    files = [relative(path) for path in available_files(ROOT)]
    left, right = st.columns(2, gap="medium")
    with left:
        show_viewer("Окно A", "a", files)
    with right:
        show_viewer("Окно B", "b", files, default_index=min(2, len(files)))


def render_alignment() -> None:
    st.subheader("Проверка результатов Vecalign")
    target = st.selectbox("Английский перевод", TARGETS, format_func=lambda x: x.replace("-translation-en", ""))
    align_path = folder(ROOT, "aligned") / f"{SOURCE}_{target}.align"
    source_path = folder(ROOT, "overlap_input") / f"{SOURCE}.txt"
    target_path = folder(ROOT, "overlap_input") / f"{target}.txt"
    absent = [relative(p) for p in (align_path, source_path, target_path) if not p.exists()]
    if absent:
        st.info("Чтобы увидеть таблицу, нужны три файла:\n\n" + "\n\n".join(f"`{item}`" for item in absent))
        return
    try:
        mapping = parse_alignment(align_path.read_text(encoding="utf-8-sig"))
        source, target_lines = sentences(source_path), sentences(target_path)
        validate_alignment([{"source_indices": row["ru_ids"], "target_indices": row["en_ids"],
                             "cost": row["cost"]} for row in mapping], len(source), len(target_lines))
        rows = expand_alignment(mapping, source, target_lines)
    except (ValueError, OSError) as exc:
        st.error(f"Ошибка разбора выравнивания: {exc}")
        return
    if not rows:
        st.warning("В файле выравнивания нет записей")
        return
    gaps = sum(row["Пропуск"] for row in rows)
    grouped = sum(bool("," in row["RU индексы"] or "," in row["EN индексы"]) for row in rows)
    invalid = sum(row["Индекс вне диапазона"] for row in rows)
    a, b, c, d = st.columns(4)
    a.metric("Соответствий", len(rows))
    b.metric("Пустых", gaps)
    c.metric("Групповых", grouped)
    d.metric("Некорректных индексов", invalid)
    filter_mode = st.radio("Фильтр", ["Все", "Только пропуски", "Только групповые", "Только ошибки индексов"], horizontal=True)
    if filter_mode == "Только пропуски":
        rows = [row for row in rows if row["Пропуск"]]
    elif filter_mode == "Только групповые":
        rows = [row for row in rows if "," in row["RU индексы"] or "," in row["EN индексы"]]
    elif filter_mode == "Только ошибки индексов":
        rows = [row for row in rows if row["Индекс вне диапазона"]]
    st.dataframe(rows, width="stretch", height=610, hide_index=True)
    if rows:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "Экспорт таблицы CSV", data="\ufeff" + output.getvalue(),
            file_name=f"review_{target}.csv", mime="text/csv", use_container_width=False,
        )
    st.caption("Стоимость Vecalign — значение целевой функции, а не вероятность корректного перевода.")


def render_settings() -> None:
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
            st.success(f"Сохранён: {relative(stored)}")
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    imported = [path for path in available_files(ROOT) if "imported" in path.parts and path.suffix == ".txt"]
    if imported:
        st.caption("Чтобы использовать импортированный TXT как исходный текст пайплайна, явно назначьте ему документ. Существующий оригинал не перезаписывается.")
        file_sel = st.selectbox("Импортированный TXT", imported, format_func=relative)
        destination = st.selectbox("Назначить документу", list(DOCUMENTS))
        if st.button("Копировать в data/raw"):
            try:
                new_path = assign_to_raw(ROOT, file_sel, destination)
                st.success(f"Создан: {relative(new_path)}")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    st.divider()
    st.subheader("Подключённые компоненты")
    names = [
        "src.processing.text_normalizer", "src.processing.postprocessor", "src.processing.normalizer",
        "src.processing.text_segmenter", "src.processing.boundary_aware_segmenter",
        "src.processing.text_json_txt_exporter", "src.processing.corpus_builder",
        "src.external_tools.vecalign_tools",
    ]
    for name in names:
        try:
            exists = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError):
            exists = False
        st.caption(f"{'✓' if exists else '—'} {name}")
    st.caption("Обнаружение модуля не гарантирует, что зависимости, модели и внешние исполняемые файлы установлены.")


@st.fragment(run_every="1s")
def render_controls() -> None:
    with st.container(border=True, key="right_actions"):
        st.subheader("Этапы обработки")
        st.caption("Выберите документы и выполните отдельный этап или весь пайплайн.")
        choices = st.multiselect(
            "Документы", options=list(DOCUMENTS), default=list(DOCUMENTS), key="ui_docs",
            format_func=lambda x: x.replace("-translation-en", "")
        )
        max_size = st.selectbox("Размер группы Vecalign", [2, 3, 4], index=0)
        current = st.session_state.get("ui_job")
        running = bool(current and current["process"].poll() is None)
        if current and not running and not current.get("completion_rendered"):
            current["completion_rendered"] = True
            st.rerun(scope="app")
        if current:
            status = read_status(ROOT, current["id"])
            st.caption(f"Последняя задача: {current['stage']} · {status.get('state', 'unknown')}")
        if st.button("▶ Запустить весь пайплайн", key="all", disabled=running or not choices, use_container_width=True, type="primary"):
            submit("all", choices, max_size)
        st.divider()
        for stage, title in STAGE_NAMES.items():
            if st.button(title, key=f"stage_{stage}", disabled=running or not choices, use_container_width=True):
                submit(stage, choices, max_size)
        if running:
            st.divider()
            if st.button("■ Остановить задачу", use_container_width=True):
                cancel_job(ROOT, current)
                st.warning("Отправлен сигнал завершения основного процесса. Внешние дочерние процессы могут требовать отдельной остановки.")
                st.rerun()
        st.caption("Ошибки модулей, пути и вывод внешних инструментов отображаются в нижнем журнале.")


def submit(stage: str, docs: list[str], max_size: int) -> None:
    try:
        st.session_state["ui_job"] = start_job(ROOT, stage, docs, max_size)
        st.session_state["ui_console_open"] = True
        st.rerun()
    except (OSError, ValueError) as exc:
        st.error(f"Ошибка запуска: {exc}")


@st.fragment(run_every="2s")
def live_console(job_id: str | None) -> None:
    if not job_id:
        st.caption("Журнал пуст: сначала запустите этап обработки.")
        return
    status = read_status(ROOT, job_id)
    state = status.get("state", "unknown")
    st.caption(f"Задача {job_id} · состояние: {state}")
    st.text_area(
        "stdout / stderr", value=tail_log(ROOT, job_id), height=245,
        disabled=True, label_visibility="collapsed",
    )
    if state == "error":
        st.error("Этап завершился с ошибкой — проверьте traceback выше.")
    elif state == "done":
        st.success("Этап завершён успешно.")


def render_console() -> None:
    if "ui_console_open" not in st.session_state:
        st.session_state["ui_console_open"] = False
    last = st.session_state.get("ui_job", {}).get("id")
    if not last:
        statuses = sorted(folder(ROOT, "jobs").glob("*.status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if statuses:
            last = statuses[0].name.removesuffix(".status.json")
    with st.bottom:
        a, b = st.columns([4, 1])
        with a:
            if st.button("▴ Журнал выполнения" if not st.session_state["ui_console_open"] else "▾ Скрыть журнал", use_container_width=True):
                st.session_state["ui_console_open"] = not st.session_state["ui_console_open"]
                st.rerun()
        with b:
            st.caption("stdout / stderr")
        if st.session_state["ui_console_open"]:
            live_console(last)


def fit_viewers() -> None:
    # Observe the actual dock and panel positions rather than assuming a screen size.
    st.html(r"""
    <style>
    [class*="st-key-viewer_body_"] {height: var(--viewer-height, 50dvh) !important; min-height: 160px !important; max-height: var(--viewer-height, 50dvh) !important; flex: 0 0 var(--viewer-height, 50dvh) !important;}
    div:has(> [class*="st-key-viewer_body_"]) {height: var(--viewer-height, 50dvh) !important; flex: 0 0 var(--viewer-height, 50dvh) !important;}
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] label {color: var(--reading-color);}
    .st-key-right_actions {overflow-y: auto; max-height: var(--controls-height, none);}
    [class*="st-key-viewer_body_"] textarea:disabled,
    [data-testid="stBottom"] textarea:disabled {
        color: var(--reading-color) !important;
        -webkit-text-fill-color: var(--reading-color) !important;
        opacity: 1 !important;
    }
    </style>
    <script>
    if (window.__fitViewersCleanup) window.__fitViewersCleanup();
    let scheduled = false;
    const fit = () => {
      scheduled = false;
      const app = document.querySelector('[data-testid="stApp"]');
      if (!app) return;
      const rgb = getComputedStyle(app).backgroundColor.match(/\d+/g);
      const color = rgb && (Number(rgb[0])*.299 + Number(rgb[1])*.587 + Number(rgb[2])*.114) < 128 ? '#ffffff' : '#000000';
      if (document.documentElement.style.getPropertyValue('--reading-color') !== color)
        document.documentElement.style.setProperty('--reading-color', color);
      const dock = document.querySelector('[data-testid="stBottom"]');
      const bottom = Math.min(window.innerHeight, dock ? dock.getBoundingClientRect().top : window.innerHeight);
      const controls = document.querySelector('.st-key-right_actions');
      if (controls) {
        const h = Math.max(200, bottom - controls.getBoundingClientRect().top - 16) + 'px';
        if (controls.style.getPropertyValue('--controls-height') !== h) controls.style.setProperty('--controls-height',h);
      }
      document.querySelectorAll('[class*="st-key-viewer_body_"]').forEach(el => {
        const h = Math.max(160, Math.floor(bottom - el.getBoundingClientRect().top - 82)) + 'px';
        for (const box of [el, el.parentElement]) {
          if (box.style.getPropertyValue('--viewer-height') !== h) box.style.setProperty('--viewer-height', h);
        }
      });
    };
    const schedule = () => { if (!scheduled) {scheduled = true; requestAnimationFrame(fit);} };
    const observer = new MutationObserver(schedule);
    observer.observe(document.body, {subtree:true, childList:true, attributes:true, attributeFilter:['style','class']});
    window.addEventListener('resize', schedule);
    window.__fitViewersCleanup = () => {observer.disconnect();window.removeEventListener('resize',schedule);};
    schedule();
    </script>
    """, unsafe_allow_javascript=True)


def main() -> None:
    header, version = st.columns([4, 1], vertical_alignment="center")
    with header:
        st.title("Text Pipeline · Workbench")
    with version:
        st.caption("Локальный интерфейс · Streamlit")
    page = st.radio(
        "Навигация", ["Файлы", "Выравнивание", "Настройки"],
        horizontal=True, label_visibility="collapsed"
    )
    st.divider()
    central, sidebar = st.columns([5.3, 1.7], gap="large")
    with central:
        if page == "Файлы":
            render_files()
        elif page == "Выравнивание":
            render_alignment()
        else:
            render_settings()
    with sidebar:
        render_controls()
    render_console()
    fit_viewers()


if __name__ == "__main__":
    main()
