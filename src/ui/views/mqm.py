"""Configure the local judge and inspect proposal reports with explicit coverage."""
import json
from pathlib import Path
import streamlit as st
from src.processing.pipeline.config import DOCUMENTS, folder
from src.processing.mqm.io import read_settings, atomic_json
from src.processing.mqm.backend import LocalMQMBackend
from src.ui.components.jobs import submit
from src.ui.components.fragments import fragment_selector, fragment_navigation
from src.ui.mqm_model import load_review, review_items, mqm_payload
from src.ui.components.mqm_review import show_mqm_review


def render_mqm(root: Path):
    st.subheader('MQM · локальный анализ перевода')
    st.caption('GEMBA-MQM-подобная схема: модель предлагает замечания, программа проверяет цитаты. '
               'Метрики описывают найденные предложения ошибок и фактический охват проверки.')
    current=st.session_state.get('ui_job')
    running=bool(current and current['process'].poll() is None)
    if st.button('Анализировать перевод локальной моделью',key='mqm_run',disabled=running):
        submit(root,'mqm',st.session_state.get('ui_docs',list(DOCUMENTS)))
    paths=sorted(p for p in folder(root,'mqm_reports').glob('*.mqm*.json') if not p.name.endswith('.diagnostics.json'))
    if not paths:
        st.info('Сначала сформируйте кандидатов, настройте локальную модель и запустите MQM.');return
    file_column,mode_column,download_column=st.columns([3,2,1],vertical_alignment='bottom')
    path=file_column.selectbox('Отчёт оригинал — перевод',paths,format_func=lambda p:p.name,key='mqm_file')
    try:
        raw=json.loads(path.read_text(encoding='utf-8'))
        if raw.get('schema_version') not in (1,2):raise ValueError('Неподдерживаемый формат отчёта')
        report,documents,candidates,warning=load_review(raw,root)
        if warning:st.warning(warning)
        metrics=report['metrics'];coverage=metrics['coverage']
        st.caption(f"{report['model']['name']} · статус {report['status']} · окон {coverage['analyzed_groups']} · охвачено {coverage.get('covered_groups',coverage['analyzed_groups'])}/{coverage['total_groups']} групп")
        download_column.download_button('Скачать MQM JSON',json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8'),file_name=path.name,mime='application/json')
        with st.expander('Сводные метрики и ограничения'):
            if report['status']!='complete':st.warning('Отчёт неполный. Сохранённые замечания доступны; диагностика находится в отдельном файле.')
            if not coverage['entire_text_fully_analyzed']:
                st.warning('Проверена часть текста. Частота по документу описывает найденные предложения ошибок без экстраполяции.')
            a,b,c=st.columns(3)
            a.metric('Уникальных замечаний модели',metrics['unique_error_proposals'])
            b.metric('Найдено на 1000 слов оригинала',f"{metrics['observed_errors_per_1000_document_source_words']:.2f}" if metrics['source_words'] else '—')
            c.metric('Охват слов оригинала',f"{100*coverage['source_word_fraction']:.1f}%" if coverage['source_word_fraction'] is not None else '—')
            st.json(metrics)
        mode=mode_column.selectbox('Область просмотра MQM',['Фрагмент','Весь текст'],key='mqm_view_mode')
        if mode=='Весь текст' and documents is None:
            st.info('Для просмотра всего текста нужен корпус или файл кандидатов той же версии. Сохранённые фрагменты доступны автономно.')
            mode='Фрагмент'
        fragments=review_items(report,candidates) if mode=='Весь текст' else report['fragments']
        if not fragments:st.info('Нет выбранных фрагментов для анализа.');return
        fragment_column,finding_column=st.columns([2,2])
        with fragment_column:
            selected=fragment_selector(fragments,key='mqm_fragment',label='Отчёт фрагмента',identity=lambda f:f['fragment_id'],
                format_func=lambda i:f"№{fragments[i]['group_index']} · {fragments[i]['status']}",
                indicator=lambda f:f"№{f['group_index']} · {f['status']}",scope=(str(path),path.stat().st_mtime_ns,mode),show_buttons=False)
        fragment=fragments[selected]
        if fragment.get('error'):st.error(fragment['error'])
        findings=fragment['findings'];records=findings+fragment['observations']
        by_id={r['id']:r for r in records}
        if st.session_state.get('mqm_selected_finding') not in ['__all__',*by_id]:
            st.session_state['mqm_selected_finding']='__all__'
        chosen=finding_column.selectbox('Замечание модели',['__all__',*by_id],key='mqm_selected_finding',
            format_func=lambda x:'Все замечания фрагмента' if x=='__all__' else f"{by_id[x]['marker']} · {by_id[x].get('severity',by_id[x].get('kind'))} · {by_id[x].get('category','наблюдение')}")
        fragment_navigation(fragments,selected=selected,key='mqm_fragment',
                            indicator=lambda f:f"№{f['group_index']} · {f['status']} · якоря {f['anchor_group_indices']}")
        if fragment['status']=='not_analyzed':st.info('Для этого подозрительного фрагмента вердикт модели пока отсутствует.')
        elif fragment['status']=='completed' and not findings:st.info('Модель не предложила подтверждаемых цитатами ошибок; это не гарантия правильности перевода.')
        shown=records if chosen=='__all__' else [by_id[chosen]]
        with st.container(height=110 if len(shown)>2 else 'content',border=False):
            for record in shown:
                st.write(f"{record['marker']} · {record.get('category',record.get('kind'))} · {record.get('severity','наблюдение')}: {record['explanation']}")
        st.caption('Приглушённые отметки — подозрительные фрагменты; яркие — выбранные замечания. Нажмите на отметку, чтобы открыть вердикт, маркер и серьёзность.')
        payload,picks=mqm_payload(report,fragments,selected,documents=documents,candidates=candidates,mode=mode,
                                  selected_finding=chosen,identity=f'{path}:{path.stat().st_mtime_ns}:{mode}')
        show_mqm_review(payload,picks)
        with st.expander('Входной фрагмент, вердикты и относительные координаты'):st.json(fragment)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:st.error(f'Не удалось открыть MQM-отчёт: {error}')


def render_mqm_settings(root: Path):
    with st.expander('Локальная модель и параметры'):
        try:settings=read_settings(root)
        except (ValueError,OSError) as error:
            st.error(str(error));return
        with st.form('mqm_settings'):
            url=st.text_input('OpenAI-совместимый адрес сервера',value=settings.get('base_url','http://127.0.0.1:1234/v1'),
                              help='LM Studio: http://127.0.0.1:1234/v1 · Ollama: http://127.0.0.1:11434/v1')
            model=st.text_input('Имя загруженной instruction-модели',value=settings.get('model',''))
            revision=st.text_input('Версия/квантизация модели для воспроизводимости',value=settings.get('revision','unspecified'))
            scope=st.selectbox('Область анализа',['candidates','all'],index=1 if settings.get('scope')=='all' else 0,
                               format_func=lambda x:'Все группы текста' if x=='all' else 'Только кандидаты')
            timeout=st.number_input('Ожидание ответа, секунд',min_value=1,max_value=3600,value=int(settings.get('timeout',120)))
            tokens=st.number_input('Максимум токенов ответа',min_value=128,max_value=32768,value=int(settings.get('max_tokens',2048)))
            json_mode=st.checkbox('Запрашивать режим JSON у сервера',value=bool(settings.get('json_mode',True)))
            limit=st.number_input('Максимум фрагментов за запуск (0 — все)',min_value=0,value=int(settings.get('max_fragments',0)))
            priority=st.number_input('Минимальный приоритет кандидата',min_value=0,max_value=100,value=int(settings.get('min_priority',0)))
            links=st.checkbox('Передавать компактные связи подозрительных слов (до 48)',value=bool(settings.get('include_word_links',False)))
            retries=st.number_input('Повторных попыток исправления ответа',min_value=0,max_value=2,value=int(settings.get('max_retries',1)))
            resume=st.checkbox('Продолжать с сохранённых фрагментов',value=bool(settings.get('resume',True)))
            st.caption('Предварительно загрузите модель в локальном сервере. Длинные или обрезанные ответы '
                       'не принимаются как успешный анализ. Изменив веса модели, укажите новую версию либо отключите продолжение.')
            if st.form_submit_button('Сохранить настройки MQM'):
                try:
                    LocalMQMBackend(model=model,base_url=url,timeout=timeout,max_tokens=int(tokens),json_mode=json_mode)
                    atomic_json(root/'data/mqm_settings.json',{'model':model,'base_url':url,'revision':revision,'scope':scope,
                                'timeout':timeout,'max_tokens':int(tokens),'json_mode':json_mode,'resume':resume,'max_retries':int(retries),'include_word_links':links,'max_fragments':int(limit),'min_priority':int(priority)})
                    st.success('Настройки сохранены')
                except ValueError as error:st.error(str(error))
