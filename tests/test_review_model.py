from copy import deepcopy
from pathlib import Path
import importlib
import json
from unittest.mock import Mock, patch
import pytest
from src.processing.alignment.words import WordAligner
from src.ui.review_model import review_payload, validate_review


class Backend:
    metadata = {'name':'test'}
    def align_tokens(self, s, t):
        return {'itermax':[(0,0),(1,1)],'inter':[(0,0)],'mwmf':[(0,0),(1,1)]}


def sample():
    def doc(name,lang,texts):
        return {'document':name,'language':lang,'sentences':[{'sentence_id':i,'normalized_text':t} for i,t in enumerate(texts)]}
    return WordAligner(Backend()).align(doc('book','ru',['Он читает книгу.','Он читает книгу.','😀 Мир!']),
        doc('translation','en',['He reads books.','He reads books.','😀 World!']),
        [{'source_indices':[i],'target_indices':[i],'cost':.2} for i in range(3)])


def test_original_text_projection_preserves_paragraphs_and_repeated_words():
    data=sample();original='Он читает книгу.\n\nОн читает книгу.\n\n😀 Мир!'
    payload=review_payload(data,raw_texts={'source':original})
    pane=payload['panes']['source']
    assert pane['text']==original
    assert pane['label']=='Исходный текст'
    assert pane['tokens'][1]['partners']==[1]
    assert pane['tokens'][5]['partners']==[5]
    assert pane['tokens'][2]['state']=='unlinked'
    for token in pane['tokens']:
        assert original[token['start']:token['end']]==token['text']


def test_modified_original_uses_verified_normalized_offsets():
    payload=review_payload(sample(),raw_texts={'source':'Совсем другая книга.'})
    assert 'исходник отличается' in payload['panes']['source']['label']
    assert payload['panes']['source']['text']==sample()['source']['text']


def test_method_switch_recomputes_links_not_default_token_references():
    payload=review_payload(sample(),method='inter')
    assert payload['panes']['source']['tokens'][1]['partners']==[]
    assert payload['panes']['source']['tokens'][1]['state']=='unlinked'
    assert payload['panes']['target']['tokens'][0]['partners']==[0]


def test_fragment_keeps_global_ids_but_rebases_offsets():
    payload=review_payload(sample(),block_id=1)
    pane=payload['panes']['source']
    assert pane['text']=='Он читает книгу.'
    assert pane['tokens'][0]['id']==4
    assert pane['tokens'][0]['start']==0
    assert pane['tokens'][1]['partners']==[5]


@pytest.mark.parametrize('mutate',[
    lambda d:d.update(schema_version='9'),
    lambda d:d['source']['tokens'][0].update(document_char_start=99),
    lambda d:d['source']['tokens'][0].update(token_id=True),
    lambda d:d['blocks'][0]['word_alignment']['methods']['itermax'][0].update(target_token_id=999),
    lambda d:d['blocks'][0]['source']['token_ids'].append(999),
    lambda d:d['blocks'][0]['source']['token_ids'].clear(),
])
def test_invalid_saved_json_is_rejected(mutate):
    data=sample();mutate(data)
    with pytest.raises(ValueError):validate_review(data)


def test_quotes_and_removed_numeric_references_can_map_to_original():
    data=sample()
    raw='Он читает книгу.[12]\n\nОн читает книгу.\n😀 Мир!'
    pane=review_payload(data,raw_texts={'source':raw})['panes']['source']
    assert pane['label']=='Исходный текст'
    assert pane['text']==raw
    assert pane['text'][pane['tokens'][4]['start']:pane['tokens'][4]['end']]=='Он'


def test_ui_all_pages_have_no_exceptions_and_no_window_titles():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/ui/app.py'), default_timeout=10).run()
    for page in ['Файлы','Проверка','Выравнивание','Настройки']:
        app.radio(key='navigation').set_value(page).run()
        assert not app.exception
    assert all(h.value not in ('Окно A','Окно B') for h in app.subheader)


def test_review_page_with_saved_json(tmp_path):
    from streamlit.testing.v1 import AppTest
    from src.ui.views import review
    path=tmp_path/'data/word_alignment/pair.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(sample(),ensure_ascii=False))
    with patch('src.ui.views.review.show_word_review') as display:
        app=AppTest.from_string('from pathlib import Path\nfrom src.ui.views.review import render_review\nrender_review(Path('+repr(str(tmp_path))+'))').run()
        assert not app.exception
        assert display.call_args.args[0]['panes']['source']['tokens'][1]['partners']==[1]
        app.selectbox(key='review_method').set_value('inter').run()
        assert display.call_args.args[0]['panes']['source']['tokens'][1]['partners']==[]


def test_all_processing_imports_and_external_embedding_command(tmp_path):
    for module in ['preprocessing.cleaning','preprocessing.normalization','preprocessing.export',
        'segmentation.segmenter','segmentation.ru_classifier','segmentation.boundary_aware','segmentation.english_dialogue',
        'datasets.boundary_dataset_builder','alignment.sentences','alignment.words','alignment.io','alignment.corpus','main']:
        importlib.import_module('src.processing.'+module)
    from src.external_tools.vecalign_tools import ExternalToolsRunner
    runner=ExternalToolsRunner(tmp_path,backend='native')
    with patch.object(runner,'run_command') as run:
        runner.generate_embeddings(tmp_path/'input',tmp_path/'out','ru')
    assert run.call_args.args[0][1:3]==['-m','src.processing.alignment.embeddings']


@pytest.mark.parametrize('kind', ['sentence_gap', 'too_long'])
def test_unprocessed_tokens_are_distinct_from_computed_unlinked_tokens(kind):
    from src.processing.alignment.words import BlockTooLongError
    def doc(name, texts):
        return {'document':name,'language':'ru','sentences':[{'sentence_id':i,'normalized_text':t} for i,t in enumerate(texts)]}
    backend=Backend()
    if kind=='too_long':
        backend.align_tokens=Mock(side_effect=BlockTooLongError('limit'))
    data=WordAligner(backend).align(doc('s',['Текст']),doc('t',[] if kind=='sentence_gap' else ['Text']),
        [{'source_indices':[0],'target_indices':[] if kind=='sentence_gap' else [0],'cost':0}])
    payload=review_payload(data,block_id=0)
    assert payload['panes']['source']['tokens'][0]['state']==('sentence_unaligned' if kind=='sentence_gap' else 'not_processed')
    assert payload['panes']['source']['tokens'][0]['partners']==[]
    if kind=='sentence_gap':assert payload['panes']['target']['text']==''
