"""Review aligned text without loading ML dependencies or rerunning inference."""
from pathlib import Path
import json
import streamlit as st
from src.ui.config import folder
from src.ui.file_browser import safe_file
from src.ui.review_model import review_payload, validate_review
from src.ui.components.word_review import show_word_review


@st.cache_data(show_spinner=False, max_entries=6)
def _read(path: str, mtime: int, size: int) -> dict:
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    validate_review(data)
    return data


def render_review(root: Path) -> None:
    paths=sorted(p for p in folder(root,'word_alignment').glob('*.json') if p.resolve().is_relative_to((root/'data').resolve()))
    if not paths:
        st.info('Для проверки запустите этап «SimAlign — выравнивание слов». Результаты появятся в data/word_alignment/.')
        return
    a,b,c=st.columns([5,2,2],gap='small')
    chosen=a.selectbox('Пара текстов',paths,format_func=lambda p:p.stem,label_visibility='collapsed',key='review_file')
    method=b.selectbox('Метод связей',['itermax','inter','mwmf'],label_visibility='collapsed',key='review_method')
    mode=c.selectbox('Область текста',['Весь текст','Фрагмент'],label_visibility='collapsed',key='review_mode')
    try:
        path=safe_file(root,str(chosen.relative_to(root)))
        data=_read(str(path),path.stat().st_mtime_ns,path.stat().st_size)
        if not data['blocks']:
            st.info('В файле нет фрагментов для проверки.');return
        a,b=st.columns([2,5],gap='small')
        number=a.number_input('Фрагмент',min_value=1,max_value=len(data['blocks']),value=1,step=1,label_visibility='collapsed',key=f'review_block_{path.name}_{path.stat().st_mtime_ns}')
        block_id=int(number)-1
        block=data['blocks'][block_id]
        b.caption(f"Фрагмент {number}/{len(data['blocks'])} · Vecalign {block['vecalign_cost']:.3f} · {block['word_alignment']['status']}")
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
