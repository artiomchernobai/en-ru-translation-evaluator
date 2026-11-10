from pathlib import Path
import streamlit as st
from src.ui.config import DOCUMENTS, SOURCE, STAGE_NAMES, TARGETS, folder
from src.ui.jobs import cancel_job,read_status,tail_log
from src.ui import jobs as job_manager
from src.external_tools.vecalign_tools import DEFAULT_DELETION_PERCENTILE

@st.fragment(run_every="1s")
def render_controls(root: Path) -> None:
    ROOT = root
    with st.container(border=True, key="right_actions"):
        st.subheader("Этапы обработки")
        st.caption("Выберите документы и выполните отдельный этап или весь пайплайн.")
        choices = st.multiselect(
            "Документы", options=list(DOCUMENTS), default=list(DOCUMENTS), key="ui_docs",
            format_func=lambda x: x.replace("-translation-en", "")
        )
        max_size = st.selectbox("Размер группы Vecalign", [2, 3, 4], index=0,
                                help="Суммарное число сегментов с обеих сторон: 3 допускает 1:2 и 2:1; 4 — также 1:3, 3:1 и 2:2.")
        del_percentile_frac = st.number_input(
            "Штраф пропуска Vecalign (перцентиль)", min_value=0.0, max_value=1.0,
            value=DEFAULT_DELETION_PERCENTILE, step=0.05, format="%.2f", key="del_percentile_frac",
            help="По умолчанию 0.20. Меньше — пропуск дешевле и чаще выбирается вместо объединения. "
                 "Слишком низкое значение может создать ложные пропуски. Применяется к выравниванию и полному запуску.")
        current = st.session_state.get("ui_job")
        running = bool(current and current["process"].poll() is None)
        if current and not running and not current.get("completion_rendered"):
            current["completion_rendered"] = True
            st.rerun(scope="app")
        if current:
            status = read_status(ROOT, current["id"])
            st.caption(f"Последняя задача: {current['stage']} · {status.get('state', 'unknown')}")
        if st.button("▶ Запустить весь пайплайн", key="all", disabled=running or not choices, use_container_width=True, type="primary"):
            submit(ROOT,"all", choices, max_size, del_percentile_frac)
        st.divider()
        for stage, title in STAGE_NAMES.items():
            if st.button(title, key=f"stage_{stage}", disabled=running or not choices, use_container_width=True):
                submit(ROOT,stage, choices, max_size, del_percentile_frac)
        if running:
            st.divider()
            if st.button("■ Остановить задачу", use_container_width=True):
                cancel_job(ROOT, current)
                st.warning("Отправлен сигнал завершения основного процесса. Внешние дочерние процессы могут требовать отдельной остановки.")
                st.rerun()
        st.caption("Ошибки модулей, пути и вывод внешних инструментов отображаются в нижнем журнале.")


def submit(ROOT: Path, stage: str, docs: list[str], max_size: int,
           del_percentile_frac: float = DEFAULT_DELETION_PERCENTILE) -> None:
    try:
        st.session_state["ui_job"] = job_manager.start_job(ROOT, stage, docs, max_size, del_percentile_frac=del_percentile_frac)
        st.session_state["ui_console_open"] = True
        st.rerun()
    except (OSError, ValueError) as exc:
        st.error(f"Ошибка запуска: {exc}")


@st.fragment(run_every="2s")
def live_console(ROOT: Path, job_id: str | None) -> None:
    if not job_id:
        st.caption("Журнал пуст: сначала запустите этап обработки.")
        return
    status = read_status(ROOT, job_id)
    state = status.get("state", "unknown")
    st.caption(f"Задача {job_id} · состояние: {state}")
    st.text_area(
        "stdout / stderr", value=tail_log(ROOT, job_id), height=140,
        disabled=True, label_visibility="collapsed",
    )
    if state == "error":
        st.error("Этап завершился с ошибкой — проверьте traceback выше.")
    elif state == "done":
        st.success("Этап завершён успешно.")


def render_console(ROOT: Path) -> None:
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
            live_console(ROOT,last)
