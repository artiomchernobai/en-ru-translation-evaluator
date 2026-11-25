"""Review aligned text without loading ML dependencies or rerunning inference."""
from pathlib import Path
import json
import streamlit as st
from src.processing.pipeline.config import folder
from src.ui.file_browser import safe_file
from src.ui.review_model import review_payload, validate_review
from src.ui.components.word_review import show_word_review


@st.cache_data(show_spinner=False, max_entries=1)
def _read(path: str, mtime: int, size: int) -> dict:
    with Path(path).open(encoding='utf-8-sig') as stream:
        data=json.load(stream)
    validate_review(data)
    return data


def render_review(root: Path) -> None:
    paths=sorted(p for p in folder(root,'word_alignment').glob('*.json') if p.resolve().is_relative_to((root/'data').resolve()))
    if not paths:
        st.info('Для проверки запустите этап «SimAlign — выравнивание слов». Результаты появятся в data/word_alignment/.')
        return
    a,b,c=st.columns([5,2,2],gap='small')
    chosen=a.selectbox('Пара текстов',paths,format_func=lambda p:p.stem,label_visibility='collapsed',key='review_file')
    method=b.selectbox('Метод просмотра',['itermax','inter','mwmf'],label_visibility='collapsed',key='review_method')
    mode=c.selectbox('Область текста',['Весь текст','Фрагмент'],label_visibility='collapsed',key='review_mode')
    try:
        # Generated token maps can exceed the generic text/JSON preview limit.
        path=safe_file(root,str(chosen.relative_to(root)),max_bytes=None)
        stat=path.stat()
        with st.spinner('Загрузка и проверка выравнивания…'):
            data=_read(str(path),stat.st_mtime_ns,stat.st_size)
        if not data['blocks']:
            st.info('В файле нет фрагментов для проверки.');return
        a,b=st.columns([2,5],gap='small')
        number=a.number_input('Фрагмент',min_value=1,max_value=len(data['blocks']),value=1,step=1,label_visibility='collapsed',key=f'review_block_{path.name}_{path.stat().st_mtime_ns}')
        block_id=int(number)-1
        block=data['blocks'][block_id]
        b.caption(f"Фрагмент {number}/{len(data['blocks'])} · Vecalign {block['vecalign_cost']:.3f} · {block['word_alignment']['status']}")
        with st.expander('Покрытие группы и альтернативы токенов'):
            st.caption('Покрытие — доля лексических токенов с принятой связью; не оценка верности перевода. Пустая сторона не считается ошибкой вычисления.')
            links=block['word_alignment'].get('methods',{}).get(method,[])
            for side in ('source','target'):
                ids=[i for i in block[side]['token_ids'] if not data[side]['tokens'][i]['is_punctuation']]
                covered={l[side+'_token_id'] for l in links}
                st.write(f"{side}: {sum(i in covered for i in ids)}/{len(ids)} токенов имеют связь")
            from copy import deepcopy
            token_map=deepcopy(block['word_alignment'].get('token_map',{}))
            for side,entries in token_map.items():
                other='target' if side=='source' else 'source'
                for entry in entries:
                    entry['matches']=[l[other+'_token_id'] for l in links if l[side+'_token_id']==entry['token_id']]
                    if block['word_alignment']['status']=='computed':entry['status']='linked' if entry['matches'] else 'no_reliable_match'
            st.caption(f"Просмотр: {method}. Для кандидатов сохранён метод {data['selected_method']}; измените его на странице настроек и пересчитайте этап.")
            st.json(token_map)
        raw={}
        for side in ('source','target'):
            candidate=folder(root,'raw')/f"{data[side]['document_id']}.txt"
            try:
                if not candidate.resolve().is_relative_to(folder(root,'raw').resolve()):
                    raise ValueError('Некорректный путь исходного текста')
                raw[side]=safe_file(root,str(candidate.relative_to(root))).read_text(encoding='utf-8-sig')
            except (ValueError,OSError):
                raw[side]=None
        payload=review_payload(data,method=method,raw_texts=raw,
                               block_id=block_id if mode=='Фрагмент' else None,focus_block=block_id)
        if any('исходник отличается' in pane['label'] for pane in payload['panes'].values()):
            st.warning('Исходный файл отличается от данных выравнивания. Для него показан нормализованный текст с проверенными смещениями.')
        show_word_review(payload,key='word_review')
    except (ValueError,OSError,KeyError,TypeError) as error:
        st.error(f'Не удалось открыть выравнивание: {error}')
