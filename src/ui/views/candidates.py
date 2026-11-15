"""Read-only evidence browser: exact normalized locations, no MQM verdicts."""
import html
import json
from pathlib import Path
import streamlit as st
from src.processing.pipeline.config import DOCUMENTS, folder
from src.ui.components.jobs import submit


def highlight(text, spans):
    """Escape user text and merge overlapping half-open sentence ranges."""
    ranges = []
    for span in sorted(spans, key=lambda s: (s['start'], s['end'])):
        start, end = span['start'], span['end']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != span['quote']:
            raise ValueError('Цитата и координаты не совпадают')
        if ranges and start <= ranges[-1][1]:
            ranges[-1][1] = max(end, ranges[-1][1])
        else:
            ranges.append([start, end])
    parts, cursor = [], 0
    for start, end in ranges:
        parts.extend((html.escape(text[cursor:start]), '<mark>', html.escape(text[start:end]), '</mark>'))
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return '<div style="white-space:pre-wrap;overflow-wrap:anywhere;color:inherit">' + ''.join(parts) + '</div>'


def render_candidates(root: Path):
    st.subheader('Кандидаты на проверку')
    st.caption('Эвристические сигналы требуют проверки. MQM и смысловая модель пока не запускались. '
               'Координаты относятся к нормализованному тексту, индексы начинаются с нуля.')
    current = st.session_state.get('ui_job')
    running = bool(current and current['process'].poll() is None)
    if st.button('Найти кандидатов', key='candidate_run', disabled=running):
        submit(root, 'candidates', st.session_state.get('ui_docs', list(DOCUMENTS)), 3)
    files = sorted(folder(root, 'review_candidates').glob('*.candidates.json'))
    if not files:
        st.info('Сначала выполните SimAlign и COMETKiwi, затем поиск кандидатов.')
        return
    path = st.selectbox('Пара текстов', files, format_func=lambda p: p.name, key='candidate_file')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or 'candidates' not in data:
            raise ValueError('Неподдерживаемый формат отчёта')
        st.caption(f"Групп: {data['summary']['groups']} · кандидатов: {data['summary']['candidate_groups']} · "
                   f"подготовлено задач для будущего анализа: {data['summary']['semantic_tasks']}")
        st.download_button('Скачать отчёт JSON', path.read_bytes(), file_name=path.name, mime='application/json')
        codes = sorted({s['code'] for c in data['candidates'] for s in c['signals']})
        selected_codes = st.multiselect('Сигналы', codes, key='candidate_signals')
        items = [c for c in data['candidates'] if not selected_codes or any(s['code'] in selected_codes for s in c['signals'])]
        if not items:
            st.info('Кандидатов по выбранному фильтру нет. Это не доказывает отсутствие смысловых ошибок.')
            return
        index = st.selectbox('Группа · приоритет проверки', list(range(len(items))),
                             format_func=lambda i: f"№{items[i]['group_index']} · {items[i]['priority']}", key='candidate_index')
        candidate = items[index]
        task = data['semantic_tasks'][candidate['group_index']]
        st.caption('Приоритет — сумма весов эвристик с ограничением 100, не вероятность ошибки.')
        names = ['Все сигналы'] + list(dict.fromkeys(s['code'] for s in candidate['signals']))
        selected = st.selectbox('Подсветка основания', names, key='candidate_highlight')
        signals = [s for s in candidate['signals'] if selected == 'Все сигналы' or s['code'] == selected]
        for signal in signals:
            st.write(f"{signal['code']}: {signal['message']}")
        columns = st.columns(2)
        for column, side, label in zip(columns, ('source', 'target'), ('Оригинал', 'Перевод')):
            with column:
                st.caption(label)
                with st.container(height=420, border=True):
                    for record in task['window'][side]['sentences']:
                        spans = [s for signal in signals for s in signal[side+'_spans'] if s['sentence_index'] == record['index']]
                        st.caption(f"[{record['index']}] · ID {record['sentence_id']}")
                        st.html(highlight(record['text'], spans))
        with st.expander('Координаты, токены и основания'):
            st.json(signals)
        st.caption('Записи Vecalign и контекст окна')
        st.code('\n'.join(g['vecalign_line'] for g in task['window']['alignments']), language=None)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        st.error(f'Не удалось открыть отчёт: {error}')
