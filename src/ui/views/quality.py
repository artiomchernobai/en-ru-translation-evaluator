"""Inspect the exact source/translation windows sent to COMETKiwi."""
import json
from pathlib import Path
import streamlit as st
from src.processing.pipeline.config import folder
from src.processing.evaluation.cometkiwi import DEFAULT_MODEL
from src.ui.components.jobs import submit
from src.ui.components.fragments import fragment_selector, fragment_navigation


def render_quality(root: Path):
    st.subheader('Оценка перевода · COMETKiwi')
    st.caption('Фиксированные или адаптивные окна вокруг центральной группы Vecalign. '
               'Оценивается весь фрагмент; балл не является вероятностью ошибки.')
    with st.expander('Модель и параметры запуска'):
        settings_path = root / 'data/cometkiwi_settings.json'
        try:
            settings = json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.exists() else {}
            if not isinstance(settings, dict):
                raise ValueError('Настройки должны быть JSON-объектом')
        except (OSError, ValueError) as error:
            st.error(f'Не удалось прочитать настройки: {error}')
            settings = {}
        with st.form('cometkiwi_settings'):
            model = st.text_input('Имя модели Hugging Face или путь к checkpoint', value=settings.get('model', DEFAULT_MODEL))
            python = st.text_input('Python отдельного окружения COMETKiwi', value=settings.get('python', ''),
                                   help=r'Например D:\piton\.venv-comet12\Scripts\python.exe. COMET устанавливается отдельно от Streamlit.')
            batch = st.number_input('Размер батча', min_value=1, max_value=128, value=int(settings.get('batch_size', 8)))
            strategy = st.selectbox('Сборка контекста', ['fixed', 'adaptive'],
                                    index=1 if settings.get('window_strategy') == 'adaptive' else 0,
                                    format_func=lambda value: 'Адаптивная' if value == 'adaptive' else 'Фиксированная · три группы')
            extra = st.number_input('Максимум дополнительных групп', min_value=0, max_value=100,
                                    value=int(settings.get('max_extra_groups', 6)))
            limit = st.number_input('Верхний лимит токенов на сторону', min_value=1, max_value=8192,
                                    value=int(settings.get('max_tokens', 510)),
                                    help='Фактический предел дополнительно ограничивается моделью и её токенизатором.')
            gpu = st.checkbox('Использовать GPU', value=bool(settings.get('gpus', 0)))
            st.caption('Модель загружается один раз на задачу и используется для всех выбранных переводов. '
                       'Для первой загрузки могут потребоваться доступ к Hugging Face и принятие лицензии модели. '
                       'При превышении длины окно отмечается без оценки; текст не обрезается.')
            if st.form_submit_button('Сохранить параметры'):
                if not model.strip():
                    st.error('Укажите модель или checkpoint')
                else:
                    settings_path.parent.mkdir(parents=True, exist_ok=True)
                    settings_path.write_text(json.dumps({**settings, 'model': model.strip(), 'batch_size': int(batch), 'gpus': int(gpu), 'python': python.strip(),
                                                         'window_strategy': strategy, 'max_extra_groups': int(extra), 'max_tokens': int(limit)}, indent=2), encoding='utf-8')
                    st.success('Параметры сохранены для следующей задачи')
    current = st.session_state.get('ui_job')
    running = bool(current and current['process'].poll() is None)
    if st.button('Оценить выбранные переводы', disabled=running, key='quality_run'):
        submit(root, 'cometkiwi', st.session_state.get('ui_docs', ['original-rus', 'glenny-translation-en', 'pevear-translation-en']), 3)
    files = sorted(folder(root, 'translation_quality').glob('*.cometkiwi*.json'))
    if not files:
        st.info('Результатов пока нет. Нужны нормализованные предложения и выравнивание Vecalign. Запустите этап COMETKiwi.')
        return
    path = st.selectbox('Пара оригинал — перевод', files, format_func=lambda p: p.name, key='quality_file')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        windows = data['windows']
        if data.get('schema_version') != 1 or not isinstance(windows, list):
            raise ValueError('Неподдерживаемый формат результата')
        st.caption(f"Модель: {data['model']['name']} · {data['summary']}")
        st.download_button('Скачать JSON', path.read_bytes(), file_name=path.name, mime='application/json')
        if not windows:
            st.info('В выравнивании нет записей')
            return
        mode = st.selectbox('Порядок просмотра', ['Порядок текста', 'Сначала низкие оценки', 'Без оценки'], key='quality_sort')
        if mode == 'Сначала низкие оценки':
            windows = sorted(windows, key=lambda w: (w['verdict']['score'] is None, w['verdict']['score'] or 0))
        elif mode == 'Без оценки':
            windows = [w for w in windows if w['verdict']['status'] != 'scored']
        if not windows:
            st.info('Нет окон для выбранного фильтра')
            return
        selected = fragment_selector(windows, key='quality_window', label='Центральная группа Vecalign',
                                     identity=lambda w: w['center_group_index'],
                                     format_func=lambda i: f"№{windows[i]['center_group_index']} · {windows[i]['verdict']['score']}",
                                     indicator=lambda w: f"№{w['center_group_index']} · оценка окна {w['verdict']['score']}",
                                     scope=(str(path), path.stat().st_mtime_ns), show_buttons=False)
        window = windows[selected]
        verdict = window['verdict']
        st.caption(f"Контекст: {data.get('window_config', {}).get('strategy', 'fixed')} · группы {window['group_indices']}")
        if 'context' in window:
            with st.expander('Сборка окна и токены'):
                st.json(window['context'])
        if verdict['status'] == 'scored':
            st.metric('Оценка всего окна · выше — лучше', f"{verdict['score']:.4f}")
        elif verdict['status'] == 'too_long':
            st.warning('Окно превышает ограничение модели. Оценка не вычислена; текст не обрезался.')
        else:
            st.warning('Одна из сторон окна пуста. Оценка не вычислена.')
        if window['alignment_gap_present']:
            st.warning('В окне есть группа с пустой стороной. Требуется проверка выравнивания.')
        st.caption('Центральная запись')
        st.code(window['center_alignment']['vecalign_line'], language=None)
        fragment_navigation(windows, selected=selected, key='quality_window',
                            indicator=lambda w: f"№{w['center_group_index']} · оценка окна {w['verdict']['score']}")
        left, right = st.columns(2)
        for column, key, title in [(left, 'source', 'Оригинал'), (right, 'target', 'Перевод')]:
            with column:
                st.caption(f"{title} · индексы {window[key]['sentence_indices']}")
                center_ids = window['center_alignment'][key+'_indices']
                with st.container(height=420, border=True):
                    for sentence in window[key]['sentences']:
                        central = sentence['index'] in center_ids
                        st.caption(f"[{sentence['index']}]" + (' · центр' if central else ' · контекст'))
                        # Plain text prevents markup in uploaded books from executing.
                        st.text(sentence['text'])
        st.caption('Все записи Vecalign, использованные для оценки окна')
        st.code('\n'.join(g['vecalign_line'] for g in window['alignments']), language=None)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        st.error(f'Не удалось открыть результат COMETKiwi: {error}')
