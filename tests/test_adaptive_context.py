"""Adaptive context, token budgets and reproducible candidate windows."""
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from src.processing.evaluation.windows import build_windows
from src.processing.evaluation.cometkiwi import CometKiwiBackend
from src.processing.evaluation.io import evaluate_files
from src.processing.analysis.candidates import collect_candidates
from src.processing.alignment.words import WordAligner


def docs(texts, target=None):
    def document(name, lang, strings):
        return {'document': name, 'language': lang, 'sentences': [
            {'sentence_id': i, 'normalized_text': text} for i, text in enumerate(strings)]}
    source = document('source', 'ru', texts)
    target = document('target', 'en', target if target is not None else texts)
    groups = [{'source_indices': [i], 'target_indices': [i], 'cost': .2} for i in range(len(texts))]
    return source, target, groups


def counter(text):
    return len(text.split()) + 2


def adaptive(source, target, groups, **kwargs):
    return build_windows(source, target, groups, strategy='adaptive', token_counter=counter, **{'max_tokens': 100, **kwargs})


def test_short_reply_adds_previous_question_outside_initial_three():
    source,target,groups = docs(['Are you invited?', 'He asked.', 'She nodded.', 'Yes.', 'They waited.'])
    fixed = build_windows(source,target,groups)
    result = adaptive(source,target,groups)
    assert fixed[3]['group_indices'] == [2,3,4]
    assert result[3]['group_indices'] == [0,1,2,3,4]
    assert all(s['direction'] == 'left' for s in result[3]['context']['expansions'])
    assert result[3]['center_alignment'] == fixed[3]['center_alignment']
    assert 'source:short_reply_needs_question' in result[3]['context']['expansions'][0]['reasons']


def test_both_directions_and_minimum_preserved():
    source,target,groups = docs(['A beginning.', 'and continuation.', 'Center.', 'He said:', 'A conclusion.'])
    window = adaptive(source,target,groups)[2]
    assert window['group_indices'] == [0,1,2,3,4]
    assert [s['direction'] for s in window['context']['expansions']] == ['left','right']
    assert window['context']['base_group_indices'] == [1,2,3]
    assert len(build_windows(source,target,groups)[2]['group_indices']) == 3


def test_token_limit_checked_for_target_and_source_before_expansion():
    for target_is_long in [True,False]:
        short = ['Start.', 'Center.', 'He said:', 'End.']
        long = short[:3] + ['These many additional tokens cannot fit into this tiny budget.']
        source,target,groups = docs(short if target_is_long else long, long if target_is_long else short)
        window = adaptive(source,target,groups,max_tokens=8)[1]
        assert window['group_indices'] == [0,1,2]
        assert 'right:token_limit' in window['context']['stop_reasons']
        assert all(n<=8 for n in window['context']['token_counts'].values())


def test_minimum_over_budget_not_shrunk_and_not_expanded():
    source,target,groups = docs(['Start.', 'Center.', 'He said:', 'End.'])
    window = adaptive(source,target,groups,max_tokens=2)[1]
    assert window['group_indices'] == [0,1,2]
    assert window['context']['expansions'] == []
    assert window['context']['stop_reasons'] == ['minimum_exceeds_token_limit']


def test_incomplete_tail_expands_right_until_complete():
    source,target,groups = docs(['Start.', 'Center.', 'He said:', '"One more', 'word."', 'Unrelated.'])
    window = adaptive(source,target,groups)[1]
    assert window['group_indices'] == [0,1,2,3,4]
    assert all(s['direction']=='right' for s in window['context']['expansions'])


def test_expansion_bound_and_document_edges():
    source,target,groups = docs(['first,','second,','third,','fourth,','fifth,'])
    windows=adaptive(source,target,groups,max_extra_groups=1)
    assert windows[2]['group_indices'] == [0,1,2,3]
    assert 'expansion_limit' in windows[2]['context']['stop_reasons']
    assert windows[0]['context']['base_group_indices'] == [0,1]
    source,target,groups=docs(['Done.'])
    assert adaptive(source,target,groups)[0]['group_indices']==[0]


@pytest.mark.parametrize('kwargs', [{'radius':0}, {'max_tokens':None}, {'max_extra_groups':-1}])
def test_invalid_adaptive_configuration_rejected(kwargs):
    source,target,groups=docs(['Done.'])
    with pytest.raises(ValueError): adaptive(source,target,groups,**kwargs)


class AdaptiveLinks:
    metadata={'name':'test-links'}
    def align_tokens(self, source, target):
        return {'mwmf': []}


def make_backend(limit=100):
    tokenizer=SimpleNamespace(model_max_length=512, encode=lambda text, **kwargs:list(range(counter(text))))
    predict=Mock(side_effect=lambda pairs, **kwargs: SimpleNamespace(scores=[.8]*len(pairs)))
    return CometKiwiBackend(SimpleNamespace(encoder=SimpleNamespace(tokenizer=tokenizer,max_positions=512), predict=predict),max_tokens=limit)


def files(tmp_path):
    source,target,groups=docs(['Start.', 'and continuation.', 'Center.', 'He said:', 'A conclusion.'])
    source_path,target_path,align_path=[tmp_path/name for name in ('source.json','target.json','pair.align')]
    source_path.write_text(json.dumps(source));target_path.write_text(json.dumps(target))
    align_path.write_text('\n'.join(f'{g["source_indices"]}:{g["target_indices"]}:0.200000' for g in groups))
    return source,target,groups,source_path,target_path,align_path


def test_evaluated_context_is_reused_by_candidates_without_tokenizer(tmp_path):
    source,target,groups,sp,tp,ap=files(tmp_path)
    backend=make_backend()
    comet=evaluate_files(sp,tp,ap,tmp_path/'adaptive.json',backend=backend,strategy='adaptive')
    words=WordAligner(AdaptiveLinks(),selected_method='mwmf').align(source,target,groups)
    result=collect_candidates(words,comet)
    assert result['window_config']['strategy']=='adaptive'
    assert result['semantic_tasks'][2]['window']==comet['windows'][2]
    assert result['semantic_tasks'][2]['word_links'][0]['group_index']==0
    assert comet['windows'][2]['context']['token_counts']['src']==counter(comet['windows'][2]['source']['text'])
    assert backend.model.predict.call_count==1
    assert result['candidates'][0]['context_group_indices']


@pytest.mark.parametrize('corruption',['skip','trace','limit','counts'])
def test_candidate_validation_rejects_forged_adaptive_context(tmp_path,corruption):
    source,target,groups,sp,tp,ap=files(tmp_path)
    comet=evaluate_files(sp,tp,ap,tmp_path/'adaptive.json',backend=make_backend(),strategy='adaptive')
    words=WordAligner(AdaptiveLinks(),selected_method='mwmf').align(source,target,groups)
    window=comet['windows'][2]
    if corruption=='skip': window['group_indices']=[0,2,3,4]
    if corruption=='trace': window['context']['expansions'][0]['group_index']=4
    if corruption=='limit': comet['window_config']['max_extra_groups']=0
    if corruption=='counts': window['context']['token_counts']['src']=101
    with pytest.raises(ValueError):collect_candidates(words,comet)


def test_over_budget_minimum_gets_no_score_and_still_yields_candidate(tmp_path):
    source,target,groups,sp,tp,ap=files(tmp_path)
    backend=make_backend(2)
    comet=evaluate_files(sp,tp,ap,tmp_path/'adaptive.json',backend=backend,strategy='adaptive')
    assert all(w['verdict']['status']=='too_long' for w in comet['windows'])
    backend.model.predict.assert_not_called()
    words=WordAligner(AdaptiveLinks(),selected_method='mwmf').align(source,target,groups)
    report=collect_candidates(words,comet)
    assert any(s['code']=='comet_unscored' for c in report['candidates'] for s in c['signals'])


def test_stage_uses_separate_paths_for_both_modes(tmp_path,monkeypatch):
    from src.processing.evaluation import stages
    from src.processing.analysis import stages as candidates
    from src.processing.pipeline.config import SOURCE,TARGETS,folder
    for name in [SOURCE,TARGETS[0]]:
        path=folder(tmp_path,'normalized')/f'{name}.json'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    path=folder(tmp_path,'aligned')/f'{SOURCE}_{TARGETS[0]}.align'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('[0]:[0]:0.1')
    settings_path=tmp_path/'data/cometkiwi_settings.json'
    settings_path.write_text(json.dumps({'window_strategy':'adaptive'}))
    pair=stages.selected_pairs(tmp_path,[SOURCE,TARGETS[0]])[0]
    assert pair[-1].name.endswith('.cometkiwi.adaptive.json')
    for category,name in [('word_alignment',f'{SOURCE}_{TARGETS[0]}.json'),('translation_quality',pair[-1].name)]:
        path=folder(tmp_path,category)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    collect=Mock();monkeypatch.setattr(candidates,'collect_files',collect)
    candidates.stage_candidates(tmp_path,[SOURCE,TARGETS[0]],3)
    assert collect.call_args.args[-1].name.endswith('.candidates.adaptive.json')
    assert collect.call_args.kwargs['window_strategy']=='adaptive'
    settings_path.write_text('{}')
    assert stages.selected_pairs(tmp_path,[SOURCE,TARGETS[0]])[0][-1].name.endswith('.cometkiwi.json')


def test_fragment_buttons_and_dropdown_share_selection_across_filters():
    from streamlit.testing.v1 import AppTest
    script = '''
import streamlit as st
from src.ui.components.fragments import fragment_selector
scope = st.selectbox('File', ['a', 'b'])
filtered = st.checkbox('Filter')
items = [{'id': n, 'priority': n*10} for n in ([2,3] if filtered else [1,2,3])]
selected = fragment_selector(items,key='fragment',label='Dropdown', identity=lambda c:c['id'],
                            format_func=lambda i:str(items[i]['id']),
                            indicator=lambda c:f"№{c['id']} · приоритет {c['priority']}",scope=scope)
st.text(f"selected={items[selected]['id']}")
'''
    app=AppTest.from_string(script).run()
    assert not app.exception
    assert app.button(key='fragment_previous').disabled
    app.button(key='fragment_next').click().run()
    assert app.selectbox(key='fragment').value==1
    assert app.text[0].value=='selected=2'
    assert any('приоритет 20' in item.value for item in app.caption)
    app.checkbox[0].check().run()
    assert app.selectbox(key='fragment').value==0  # same group 2 after filtering
    assert app.text[0].value=='selected=2'
    app.selectbox(key='fragment').select(1).run()
    assert app.button(key='fragment_next').disabled
    app.button(key='fragment_previous').click().run()
    assert app.text[0].value=='selected=2'
    app.selectbox[0].select('b').run()
    assert not app.exception
    assert app.selectbox(key='fragment').value==0


def test_saved_adaptive_settings_reach_isolated_worker(tmp_path,monkeypatch):
    from src.processing.evaluation import stages
    from src.processing.pipeline.config import SOURCE,TARGETS,folder
    import sys
    fake=tmp_path/'fake';fake.mkdir()
    (fake/'comet.py').write_text('''
from types import SimpleNamespace
class Model:
    encoder=SimpleNamespace(tokenizer=SimpleNamespace(model_max_length=512,encode=lambda text,**kw:text.split()+['S','E']),max_positions=512)
    def eval(self):pass
    def requires_references(self):return False
    def predict(self,pairs,**kw):return SimpleNamespace(scores=[.85]*len(pairs))
def download_model(name):return 'test.ckpt'
def load_from_checkpoint(path):return Model()
''')
    monkeypatch.setenv('PYTHONPATH',str(fake))
    monkeypatch.setenv('COMETKIWI_PYTHON',sys.executable)
    source,target,groups=docs(['Are you invited?', 'He asked.', 'She nodded.', 'Yes.', 'They waited.'])
    for name,document in [(SOURCE,source),(TARGETS[0],target)]:
        path=folder(tmp_path,'normalized')/f'{name}.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(document))
    path=folder(tmp_path,'aligned')/f'{SOURCE}_{TARGETS[0]}.align';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(f'{g["source_indices"]}:{g["target_indices"]}:0.2' for g in groups))
    (tmp_path/'data/cometkiwi_settings.json').write_text(json.dumps({'window_strategy':'adaptive','max_tokens':100,'max_extra_groups':6}))
    stages.stage_cometkiwi(tmp_path,[SOURCE,TARGETS[0]],3)
    result=json.loads((folder(tmp_path,'translation_quality')/f'{SOURCE}_{TARGETS[0]}.cometkiwi.adaptive.json').read_text())
    assert result['window_config']['strategy']=='adaptive'
    assert result['windows'][3]['group_indices']==[0,1,2,3,4]
    assert result['windows'][3]['verdict']['score']==.85


def test_candidate_strategy_mismatch_does_not_overwrite_report(tmp_path):
    from src.processing.analysis.io import collect_files
    source,target,groups,sp,tp,ap=files(tmp_path)
    comet_path=tmp_path/'adaptive.json'
    evaluate_files(sp,tp,ap,comet_path,backend=make_backend(),strategy='adaptive')
    words=WordAligner(AdaptiveLinks(),selected_method='mwmf').align(source,target,groups)
    word_path=tmp_path/'words.json';word_path.write_text(json.dumps(words))
    output=tmp_path/'candidates.json';output.write_text('previous result')
    with pytest.raises(ValueError,match='strategy'):
        collect_files(word_path,comet_path,output,window_strategy='fixed')
    assert output.read_text()=='previous result'


def test_candidate_mode_change_preserves_other_settings(tmp_path):
    from streamlit.testing.v1 import AppTest
    settings_path=tmp_path/'data/candidate_settings.json'
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({'low_quantile':.2,'glossary':[]}))
    app=AppTest.from_string(f'from pathlib import Path\nfrom src.ui.views.candidates import render_candidates\nrender_candidates(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    # Merely opening the page must not persist a mode override.
    assert 'window_strategy' not in json.loads(settings_path.read_text())
    assert not any(s.key=='candidate_strategy' for s in app.selectbox)
    settings=json.loads(settings_path.read_text())
    assert 'window_strategy' not in settings
    assert settings['low_quantile']==.2 and settings['glossary']==[]
