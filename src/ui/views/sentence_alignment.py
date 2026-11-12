from pathlib import Path
import streamlit as st
from src.processing.pipeline.config import DOCUMENTS, SOURCE, STAGE_NAMES, TARGETS, folder
import csv
import io
from src.ui.alignment_view import expand_alignment,parse_alignment,sentences
from src.processing.alignment.sentences import validate_alignment

def render_alignment(root: Path) -> None:
    ROOT = root
    relative = lambda p: str(p.relative_to(ROOT))
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
