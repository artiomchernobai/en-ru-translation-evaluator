"""Configure the local judge and inspect proposal reports with explicit coverage."""
import json
from pathlib import Path
import streamlit as st
from src.processing.pipeline.config import DOCUMENTS, folder
from src.processing.mqm.io import read_settings, atomic_json
from src.processing.mqm.backend import LocalMQMBackend
from src.ui.components.jobs import submit
from src.ui.components.fragments import fragment_selector
from src.ui.views.candidates import highlight


def render_mqm(root: Path):
    st.subheader('MQM · локальный анализ перевода')
    st.caption('GEMBA-MQM-подобная схема: модель предлагает замечания, программа проверяет цитаты. '
               'Метрики описывают найденные предложения ошибок и фактический охват проверки.')
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
            links=st.checkbox('Передавать подробные связи слов (увеличивает запрос)',value=bool(settings.get('include_word_links',False)))
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
    current=st.session_state.get('ui_job')
    running=bool(current and current['process'].poll() is None)
    if st.button('Анализировать перевод локальной моделью',key='mqm_run',disabled=running):
        submit(root,'mqm',st.session_state.get('ui_docs',list(DOCUMENTS)),3)
    paths=sorted(folder(root,'mqm_reports').glob('*.mqm*.json'))
    if not paths:
        st.info('Сначала сформируйте кандидатов, настройте локальную модель и запустите MQM.');return
    path=st.selectbox('Отчёт оригинал — перевод',paths,format_func=lambda p:p.name,key='mqm_file')
    try:
        report=json.loads(path.read_text(encoding='utf-8'))
        if report.get('schema_version')!=1:raise ValueError('Неподдерживаемый формат отчёта')
        metrics=report['metrics'];coverage=metrics['coverage']
        st.caption(f"{report['model']['name']} · статус {report['status']} · проверено {coverage['analyzed_groups']}/{coverage['total_groups']} групп")
        if report['status']!='complete':st.warning('Отчёт неполный. Проверьте ошибки фрагментов; повторный запуск продолжит работу.')
        if not coverage['entire_text_fully_analyzed']:
            st.warning('Проверена часть текста. Частота по всему документу — наблюдаемое число предложенных ошибок на объём текста, без экстраполяции.')
        a,b,c=st.columns(3)
        a.metric('Уникальных замечаний модели',metrics['unique_error_proposals'])
        b.metric('Найдено на 1000 слов оригинала',f"{metrics['observed_errors_per_1000_document_source_words']:.2f}" if metrics['source_words'] else '—')
        c.metric('Охват слов оригинала',f"{100*coverage['source_word_fraction']:.1f}%" if coverage['source_word_fraction'] is not None else '—')
        st.download_button('Скачать MQM JSON',path.read_bytes(),file_name=path.name,mime='application/json')
        with st.expander('Сводные метрики и ограничения'):st.json(metrics)
        fragments=report['fragments']
        if not fragments:st.info('Нет выбранных фрагментов для анализа.');return
        selected=fragment_selector(fragments,key='mqm_fragment',label='Отчёт фрагмента',identity=lambda f:f['group_index'],
            format_func=lambda i:f"№{fragments[i]['group_index']} · {fragments[i]['status']}",
            indicator=lambda f:f"№{f['group_index']} · {f['status']}",scope=(str(path),path.stat().st_mtime_ns))
        fragment=fragments[selected]
        if fragment.get('error'):st.error(fragment['error'])
        findings=fragment['findings']
        if fragment['status']=='completed' and not findings:st.info('Модель не предложила подтверждаемых цитатами ошибок; это не гарантия правильности перевода.')
        for finding in findings:
            st.write(f"{finding['category']} · {finding['severity']}: {finding['explanation']}")
        columns=st.columns(2)
        for col,side,label in zip(columns,('source','target'),('Оригинал','Перевод')):
            with col:
                st.caption(label)
                with st.container(height=320,border=True):
                    for sentence in fragment.get('context',{}).get(side,[]):
                        spans=[s for f in findings for s in f[side+'_spans'] if s['sentence_index']==sentence['index']]
                        st.caption(f"[{sentence['index']}]")
                        st.html(highlight(sentence['text'],spans))
        with st.expander('Наблюдения, отклонённые цитаты и координаты'):
            st.json({'observations':fragment['observations'],'rejected':fragment['rejected'],'findings':findings})
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:st.error(f'Не удалось открыть MQM-отчёт: {error}')
