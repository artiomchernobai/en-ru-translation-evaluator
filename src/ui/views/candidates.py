"""Read-only evidence browser: exact normalized locations, no MQM verdicts."""
import html
import json
from pathlib import Path
import streamlit as st
from src.processing.pipeline.config import DOCUMENTS, folder
from src.ui.components.jobs import submit
from src.ui.components.fragments import fragment_selector, fragment_navigation
from src.processing.analysis.io import load_settings
from src.processing.evaluation.settings import read_settings
from src.ui.candidate_model import candidate_payload
from src.ui.components.candidate_review import show_candidate_review
from src.processing.analysis.fragments import review_fragments


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
    st.caption('Кандидаты требуют проверки · нормализованный текст · вердикты модели доступны в отчётах MQM.')
    view_column, run_column = st.columns([3,1])
    view_mode = view_column.selectbox('Область просмотра', ['Фрагмент', 'Весь текст'], key='candidate_view_mode')
    current = st.session_state.get('ui_job')
    running = bool(current and current['process'].poll() is None)
    if run_column.button('Найти кандидатов', key='candidate_run', disabled=running):
        submit(root, 'candidates', st.session_state.get('ui_docs', list(DOCUMENTS)))
    files = sorted(folder(root, 'review_candidates').glob('*.candidates*.json'))
    if not files:
        st.info('Сначала выполните SimAlign и COMETKiwi, затем поиск кандидатов.')
        return
    file_column, filter_column, download_column = st.columns([3, 3, 1], vertical_alignment='bottom')
    path = file_column.selectbox('Пара текстов', files, format_func=lambda p: p.name, key='candidate_file')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or 'candidates' not in data:
            raise ValueError('Неподдерживаемый формат отчёта')
        st.caption(f"Групп: {data['summary']['groups']} · кандидатов: {data['summary']['candidate_groups']} · "
                   f"подготовлено задач для будущего анализа: {data['summary']['semantic_tasks']}")
        download_column.download_button('Скачать отчёт JSON', path.read_bytes(), file_name=path.name, mime='application/json')
        codes = sorted({s['code'] for c in data['candidates'] for s in c['signals']})
        selected_codes = filter_column.multiselect('Сигналы', codes, key='candidate_signals')
        items = [c for c in review_fragments(data) if not selected_codes or any(s['code'] in selected_codes for s in c['signals'])]
        if not items:
            st.info('Кандидатов по выбранному фильтру нет. Это не доказывает отсутствие смысловых ошибок.')
            return
        selection_column, highlight_column = st.columns([3, 2])
        with selection_column:
            index = fragment_selector(items, key='candidate_index', label='Группа · приоритет проверки',
                                  identity=lambda c: c['group_index'],
                                  format_func=lambda i: f"№{items[i]['group_index']} · {items[i]['priority']}",
                                  indicator=lambda c: f"№{c['group_index']} · приоритет {c['priority']}",
                                  scope=(str(path), path.stat().st_mtime_ns), show_buttons=False)
        candidate = items[index]
        task = data['semantic_tasks'][candidate['group_index']]
        if st.button('Оценить выбранный фрагмент через MQM',key='candidate_mqm',disabled=running):
            from src.ui.jobs import start_job
            st.session_state['ui_job']=start_job(root,'mqm',[data['source_document'],data['target_document']],
                selection={'group_index':candidate['group_index'],'target':data['target_document'],'strategy':data.get('window_config',{}).get('strategy','fixed')})
            st.session_state['ui_console_open']=True
            st.rerun()
        st.caption('MQM проверит весь контекст выбранного кандидата и сохранит отдельный отчёт; общая очередь не запускается.')
        st.caption(f"Контекст: {data.get('window_config', {}).get('strategy', 'fixed')} · группы {candidate['context_group_indices']} · якоря {candidate['anchor_group_indices']}")
        names = ['Все сигналы'] + list(dict.fromkeys(s['code'] for s in candidate['signals']))
        selected = highlight_column.selectbox('Подсветка основания', names, key='candidate_highlight')
        signals = [s for s in candidate['signals'] if selected == 'Все сигналы' or s['code'] == selected]
        with st.expander('Основания и значение приоритета'):
            st.caption('Приоритет группы — сумма весов эвристик с ограничением 100. Для объединённого окна берётся максимум приоритетов его якорей, а не повторная сумма сигналов. Это не вероятность ошибки.')
            for signal in signals:
                st.write(f"{signal['code']}: {signal['message']}")
        fragment_navigation(items, selected=index, key='candidate_index',
                            indicator=lambda c: f"№{c['group_index']} · приоритет {c['priority']}")
        payload = candidate_payload(data, candidate, items, signals, mode=view_mode,
                                    identity=f'{path}:{path.stat().st_mtime_ns}')
        st.caption('Ярко — фрагмент выбранного сигнала · пунктир — область проверки без точной локализации · голубой фон — контекст · приглушённые отметки — остальные кандидаты. Нажмите на отметку для выбора ближайшего кандидата текущего фильтра.')
        show_candidate_review(payload, ids=[item['id'] for item in items])
        with st.expander('Координаты, токены и основания'):
            st.json(signals)
        st.caption('Записи Vecalign и контекст окна')
        st.code('\n'.join(g['vecalign_line'] for g in task['window']['alignments']), language=None)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        st.error(f'Не удалось открыть отчёт: {error}')
