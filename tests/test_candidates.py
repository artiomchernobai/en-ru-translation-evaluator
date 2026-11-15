"""Candidate provenance and locations, independent of external models."""
import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from src.processing.alignment.words import WordAligner
from src.processing.evaluation.windows import build_windows
from src.processing.evaluation.engine import evaluate_windows
from src.processing.analysis.candidates import collect_candidates
from src.processing.analysis.io import collect_files


class AllLinks:
    metadata = {'name': 'test-SimAlign'}
    def align_tokens(self, source, target):
        links = [(i, i % len(target)) for i in range(len(source))]
        links += [(i % len(source), i) for i in range(len(target))]
        return {'mwmf': links}


class HighScores:
    metadata = {'name': 'test-QE'}
    def predict(self, pairs):
        return [{'status': 'scored', 'score': .9} for _ in pairs]


def doc(name, language, texts):
    return {'document': name, 'language': language, 'sentences': [
        {'sentence_id': f'{name}-{i}', 'normalized_text': text} for i, text in enumerate(texts)]}


def pair(source=None, target=None, groups=None):
    source = source or ['Саркома лёгкого.', 'Десятого века.', 'Он не пришёл.', '1000 м.']
    target = target or ['A slight heart attack.', 'Ninth-century.', 'He came.', '1 km.']
    src, tgt = doc('source', 'ru', source), doc('target', 'en', target)
    groups = groups if groups is not None else [
        {'source_indices': [i], 'target_indices': [i], 'cost': .2} for i in range(len(source))]
    words = WordAligner(AllLinks(), selected_method='mwmf').align(src, tgt, groups)
    comet = evaluate_windows(build_windows(src, tgt, groups), HighScores())
    comet.update(source_document='source', target_document='target', radius=1)
    return words, comet


def signal_for(result, index, code):
    return next(s for c in result['candidates'] if c['group_index'] == index for s in c['signals'] if s['code'] == code)


def test_semantic_candidates_survive_high_comet_and_complete_word_links():
    words, comet = pair()
    result = collect_candidates(words, comet)
    term = signal_for(result, 0, 'terminology_mismatch')
    assert term['source_spans'][0]['quote'] == 'Саркома лёгкого'
    assert term['target_spans'][0]['quote'] == 'heart attack'
    assert not any(s['code'] == 'unlinked_content' for c in result['candidates'] for s in c['signals'])
    number = signal_for(result, 1, 'quantity_mismatch')
    assert number['source_spans'][0]['quote'] == 'Десятого века'
    assert number['target_spans'][0]['quote'] == 'Ninth-century'
    assert signal_for(result, 2, 'negation_mismatch')
    assert not any(c['group_index'] == 3 for c in result['candidates'])  # 1000 m == 1 km
    assert len(result['semantic_tasks']) == 4  # even groups without heuristic flags
    assert result['summary']['semantic_groups_checked'] == 0
    assert result['semantic_analysis'] == 'not_run'
    assert words['source']['tokens'][0]['alignment_status'] == 'linked'


def test_century_same_value_and_mass_conversion_not_flagged():
    w, c = pair(['Девятого века. 1000 г.'], ['Ninth-century. 1 kg.'])
    assert not any(s['code'] == 'quantity_mismatch' for item in collect_candidates(w,c)['candidates'] for s in item['signals'])


def test_gaps_zero_cost_not_treated_as_confident_alignment():
    groups = [{'source_indices': [0], 'target_indices': [], 'cost': 0},
              {'source_indices': [1], 'target_indices': [0], 'cost': .3}]
    words, comet = pair(['Пропущено.', 'Дальше.'], ['Next.'], groups)
    result = collect_candidates(words, comet)
    assert signal_for(result, 0, 'sentence_gap')
    assert not any(s['code'] == 'high_vecalign_cost' for s in next(c for c in result['candidates'] if c['group_index'] == 0)['signals'])
    assert result['semantic_tasks'][0]['window']['group_indices'] == [0,1]


def test_empty_comet_is_distinct_from_low_score():
    groups = [{'source_indices': [0], 'target_indices': [], 'cost': 0}]
    src, tgt = doc('source','ru',['Нет перевода.']), doc('target','en',[])
    words = WordAligner(AllLinks(), selected_method='mwmf').align(src,tgt,groups)
    comet = evaluate_windows(build_windows(src,tgt,groups), HighScores())
    comet.update(source_document='source',target_document='target',radius=1)
    result = collect_candidates(words,comet)
    assert signal_for(result,0,'comet_unscored')
    assert result['configuration']['low_comet_threshold'] is None
    assert result['candidates'][0]['comet_window']['score'] is None


@pytest.mark.parametrize('change', ['text','cost','offset','fingerprint','order','quote','token_member'])
def test_stale_or_inconsistent_inputs_rejected(change):
    words, comet = pair()
    if change == 'text': comet['windows'][0]['source']['text'] = 'Other edition'
    if change == 'cost': comet['windows'][0]['alignments'][0]['cost'] = .9
    if change == 'offset': words['source']['tokens'][0]['document_char_start'] += 1
    if change == 'fingerprint':
        words['inputs'] = {'source': {'sha256': 'a'}}
        comet['inputs'] = {'source': {'sha256': 'b'}}
    if change == 'order': comet['windows'][0]['group_indices'] = [1,0]
    if change == 'quote': comet['windows'][0]['source']['sentences'][0]['text'] = 'Other'
    if change == 'token_member': words['blocks'][0]['source']['token_ids'] = []
    with pytest.raises(ValueError): collect_candidates(words,comet)


def test_exact_coordinates_and_repeated_token_locations():
    words, comet = pair(['Он не пришёл, не позвонил.'], ['He came and called.'])
    result = collect_candidates(words,comet)
    spans = signal_for(result,0,'negation_mismatch')['source_spans']
    assert len(spans) == 2 and spans[0]['start'] != spans[1]['start']
    for candidate in result['candidates']:
        for signal in candidate['signals']:
            for side in ('source','target'):
                for span in signal[side+'_spans']:
                    sentence = words[side]['sentences'][span['sentence_index']]
                    assert sentence['normalized_text'][span['start']:span['end']] == span['quote']
                    assert words[side]['text'][span['document_start']:span['document_end']] == span['quote']
                    assert span['token_ids']


def test_configurable_name_glossary():
    w,c = pair(['Берлиоз пришёл.'], ['Ivan came.'])
    glossary = [{'id':'Berlioz', 'source':r'Берлиоз\w*', 'target':r'Berlioz'}]
    assert signal_for(collect_candidates(w,c,glossary=glossary),0,'terminology_mismatch')


def test_bad_configuration_rejected():
    w,c = pair()
    for options in [{'low_quantile':float('nan')}, {'high_quantile':False}, {'low_quantile':.9}, {'glossary':[{}]}]:
        with pytest.raises(ValueError): collect_candidates(w,c,**options)


def test_files_write_provenance_and_preserve_result_on_failure(tmp_path):
    w,c = pair()
    wp,cp,out = [tmp_path/name for name in ('words.json','comet.json','out.json')]
    wp.write_text(json.dumps(w)); cp.write_text(json.dumps(c))
    result = collect_files(wp,cp,out)
    assert json.loads(out.read_text()) == result
    assert len(result['inputs']['word_alignment']['sha256']) == 64
    assert w == json.loads(wp.read_text())
    old = out.read_bytes()
    c['windows'][0]['source']['text'] = 'stale';cp.write_text(json.dumps(c))
    with pytest.raises(ValueError): collect_files(wp,cp,out)
    assert out.read_bytes() == old
    with pytest.raises(ValueError, match='overwrite'): collect_files(wp,cp,wp)


def test_main_standalone_does_not_load_models(monkeypatch):
    from src.processing.pipeline import cli
    from src.processing.analysis import stages
    run = Mock()
    monkeypatch.setattr(stages,'stage_candidates',run)
    monkeypatch.setattr(cli,'SentenceSegmenter',Mock(side_effect=AssertionError('no segmentation')))
    cli.main(['--stage','candidates'])
    run.assert_called_once_with(cli.BASE_DIR,list(cli.DOCUMENTS),3)


def test_ui_highlighting_escapes_book_markup():
    from src.ui.views.candidates import highlight
    text = '<script>alert(1)</script> саркома'
    start = text.index('саркома')
    result = highlight(text,[{'start':start,'end':start+len('саркома'),'quote':'саркома'}])
    assert '<script>' not in result
    assert '&lt;script&gt;' in result
    assert '<mark>саркома</mark>' in result
    with pytest.raises(ValueError): highlight(text,[{'start':0,'end':1,'quote':'X'}])


def test_candidate_screen_filters_and_locations(tmp_path):
    from streamlit.testing.v1 import AppTest
    w,c = pair()
    report = collect_candidates(w,c)
    path = tmp_path/'data/review_candidates/test.candidates.json'
    path.parent.mkdir(parents=True);path.write_text(json.dumps(report))
    app = AppTest.from_string(f'from pathlib import Path\nfrom src.ui.views.candidates import render_candidates\nrender_candidates(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    app.multiselect(key='candidate_signals').set_value(['terminology_mismatch']).run()
    assert not app.exception
    assert len(app.selectbox(key='candidate_index').options) == 1
    assert any('terminology_mismatch' in item.value for item in app.markdown)


def test_scores_rank_windows_but_do_not_locate_central_errors():
    w,c = pair()
    c['windows'][1]['verdict']['score'] = .1
    for window in c['windows']:
        for entry in window['alignments']:
            if entry['group_index'] == 2:
                entry['cost'] = .95
                entry['vecalign_line'] = '[2]:[2]:0.95'
        window['center_alignment'] = deepcopy(window['alignments'][window['group_indices'].index(window['center_group_index'])])
    w['blocks'][2]['vecalign_cost'] = .95
    result = collect_candidates(w,c,low_quantile=.5)
    assert signal_for(result,1,'low_comet_window')
    assert signal_for(result,2,'high_vecalign_cost')
    assert all(candidate['priority_is_probability'] is False for candidate in result['candidates'])


def test_unlinked_content_and_skipped_blocks_have_separate_signals():
    from src.processing.alignment.words import BlockTooLongError
    class NoLinks:
        metadata = {'name': 'empty'}
        def align_tokens(self,source,target): return {'mwmf': []}
    class Skipped:
        metadata = {'name': 'too-long'}
        def align_tokens(self,source,target): raise BlockTooLongError('too long')
    src,tgt = doc('source','ru',['Он читает.']),doc('target','en',['He reads.'])
    groups=[{'source_indices':[0],'target_indices':[0],'cost':.1}]
    comet=evaluate_windows(build_windows(src,tgt,groups),HighScores())
    comet.update(source_document='source',target_document='target',radius=1)
    for backend,code in [(NoLinks(),'unlinked_content'),(Skipped(),'word_alignment_unavailable')]:
        words=WordAligner(backend,selected_method='mwmf').align(src,tgt,groups)
        signal = signal_for(collect_candidates(words,comet),0,code)
        if code == 'unlinked_content':
            assert [s['quote'] for s in signal['source_spans']] == ['читает']
            assert [s['quote'] for s in signal['target_spans']] == ['reads']


@pytest.mark.parametrize('ru,en', [('Двадцать первого века.','Twenty-first century.'), ('9-го века.','Ninth-century.')])
def test_equivalent_centuries_do_not_trigger_number_signal(ru,en):
    w,c=pair([ru],[en])
    result=collect_candidates(w,c)
    assert not any(s['code']=='quantity_mismatch' for item in result['candidates'] for s in item['signals'])
