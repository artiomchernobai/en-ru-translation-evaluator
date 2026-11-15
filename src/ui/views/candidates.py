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


def save_strategy(root):
    settings = load_settings(root)
    strategy = st.session_state['candidate_strategy']
    settings['window_strategy'] = strategy
    path = root / 'data/candidate_settings.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    st.session_state['candidate_strategy_saved'] = strategy


def render_candidates(root: Path):
    st.subheader('Кандидаты на проверку')
    st.caption('Кандидаты требуют проверки · нормализованный текст · MQM пока не запускался.')
    mode_column, run_column = st.columns([3, 1], vertical_alignment='bottom')
    try:
        settings = load_settings(root)
        default_strategy = settings.get('window_strategy', read_settings(root).get('window_strategy', 'fixed'))
        if st.session_state.get('candidate_strategy_saved') != default_strategy:
            st.session_state['candidate_strategy'] = default_strategy
            st.session_state['candidate_strategy_saved'] = default_strategy
        mode_column.selectbox('Контекст кандидатов', ['fixed', 'adaptive'],
                     index=1 if default_strategy == 'adaptive' else 0, key='candidate_strategy',
                     on_change=save_strategy, args=(root,),
                     format_func=lambda value: 'Адаптивный' if value == 'adaptive' else 'Фиксированный')
    except (OSError, ValueError) as error:
        st.error(f'Не удалось прочитать настройки: {error}')
        return
    current = st.session_state.get('ui_job')
    running = bool(current and current['process'].poll() is None)
    if run_column.button('Найти кандидатов', key='candidate_run', disabled=running):
        submit(root, 'candidates', st.session_state.get('ui_docs', list(DOCUMENTS)), 3)
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
        items = [c for c in data['candidates'] if not selected_codes or any(s['code'] in selected_codes for s in c['signals'])]
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
        st.caption(f"Контекст: {data.get('window_config', {}).get('strategy', 'fixed')} · группы {candidate['context_group_indices']}")
        names = ['Все сигналы'] + list(dict.fromkeys(s['code'] for s in candidate['signals']))
        selected = highlight_column.selectbox('Подсветка основания', names, key='candidate_highlight')
        signals = [s for s in candidate['signals'] if selected == 'Все сигналы' or s['code'] == selected]
        with st.expander('Основания и значение приоритета'):
            st.caption('Приоритет — сумма весов эвристик с ограничением 100, не вероятность ошибки.')
            for signal in signals:
                st.write(f"{signal['code']}: {signal['message']}")
        fragment_navigation(items, selected=index, key='candidate_index',
                            indicator=lambda c: f"№{c['group_index']} · приоритет {c['priority']}")
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
