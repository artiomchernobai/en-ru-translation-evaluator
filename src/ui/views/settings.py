from pathlib import Path
import streamlit as st
from src.processing.pipeline.config import DOCUMENTS, SOURCE, STAGE_NAMES, TARGETS, folder
import importlib.util
from src.ui.file_browser import available_files,store_import,assign_to_raw

def render_settings(root: Path) -> None:
    ROOT = root
    try: render_pipeline_settings(root)
    except (ValueError,OSError) as error: st.error(f"Не удалось сохранить настройки: {error}")
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


def render_pipeline_settings(root):
    import json
    from src.processing.pipeline.settings import read_settings
    from src.processing.mqm.io import atomic_json
    settings=read_settings(root)
    st.subheader('Общие параметры CLI и интерфейса')
    with st.form('pipeline_settings'):
        settings['documents']=st.multiselect('Документы пайплайна',list(DOCUMENTS),default=settings['documents'])
        st.caption('Vecalign: размер — сумма N+M. 4 разрешает 2:2, 1:3 и 3:1.')
        v=settings['vecalign']
        v['max_size']=st.number_input('Максимальный размер группы Vecalign',min_value=2,max_value=8,value=v['max_size'])
        v['deletion_percentile']=st.number_input('Перцентиль пропуска Vecalign',min_value=0.,max_value=1.,value=float(v['deletion_percentile']),step=.05)
        st.caption('Выравнивание слов. Порог — косинусное сходство, не вероятность верности перевода.')
        w=settings['word_alignment']
        w['model']=st.text_input('Энкодер SimAlign',value=w['model'])
        w['device']=st.selectbox('Устройство SimAlign',['cpu','cuda'],index=int(w['device']=='cuda'))
        w['method']=st.selectbox('Метод для пайплайна',['itermax','inter','mwmf'],index=['itermax','inter','mwmf'].index(w['method']))
        w['layer']=st.number_input('Слой энкодера',min_value=0,value=w['layer'])
        w['min_similarity']=st.number_input('Минимальное сходство связи',min_value=0.,max_value=1.,value=float(w['min_similarity']),step=.05)
        w['min_margin']=st.number_input('Минимальный отрыв от альтернативы',min_value=0.,max_value=1.,value=float(w['min_margin']),step=.01)
        st.caption('Сегментация RuBERT')
        s=settings['segmentation']
        for key,label in [('batch_size','Батч RuBERT'),('context_size','Контекст RuBERT, символов'),('max_length','Лимит токенов RuBERT')]:
            s[key]=st.number_input(label,min_value=1,value=s[key])
        s['threshold']=st.number_input('Порог разделения RuBERT',min_value=0.,max_value=1.,value=float(s['threshold']))
        if st.form_submit_button('Сохранить общие настройки'):
            # Validate a complete payload before replacing the active configuration.
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as directory:
                temporary=Path(directory);atomic_json(temporary/'data/pipeline_settings.json',settings)
                read_settings(temporary)
            atomic_json(root/'data/pipeline_settings.json',settings)
            st.success('Сохранено для CLI и интерфейса')
    with st.expander('Отбор кандидатов'):
        from src.processing.analysis.io import load_settings
        from src.processing.evaluation.settings import read_settings as comet_settings
        c=load_settings(root)
        with st.form('candidate_parameters'):
            st.caption('Контекст общий с COMETKiwi; настройте его в разделе оценки.')
            c['window_strategy']=comet_settings(root).get('window_strategy','fixed')
            c['low_quantile']=st.number_input('Нижний квантиль COMET',min_value=0.,max_value=1.,value=float(c.get('low_quantile',.1)))
            c['high_quantile']=st.number_input('Верхний квантиль Vecalign',min_value=0.,max_value=1.,value=float(c.get('high_quantile',.9)))
            glossary=st.text_area('Дополнительные понятия: JSON-список',value=json.dumps(c.get('glossary',__import__('src.processing.analysis.rules',fromlist=['DEFAULT_GLOSSARY']).DEFAULT_GLOSSARY),ensure_ascii=False,indent=2))
            if st.form_submit_button('Сохранить отбор'):
                c['glossary']=json.loads(glossary)
                if c['low_quantile']>=c['high_quantile']:raise ValueError('Нижний квантиль должен быть меньше верхнего')
                if not isinstance(c['glossary'],list) or any(not isinstance(e,dict) or any(not isinstance(e.get(k),str) or not e[k] for k in ('id','source','target')) for e in c['glossary']):raise ValueError('Некорректные понятия')
                atomic_json(root/'data/candidate_settings.json',c)
    from src.ui.views.quality import render_quality_settings
    from src.ui.views.mqm import render_mqm_settings
    render_quality_settings(root)
    render_mqm_settings(root)
