"""Fragment grouping, portable MQM reports and full-text projections."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import Mock
import pytest
from test_mqm import report, TestJudge, finding
from src.processing.analysis.fragments import review_fragments
from src.processing.mqm.contracts import make_request, validate_judgment
from src.processing.mqm.engine import analyze
from src.processing.mqm.io import analyze_file
from src.processing.mqm.prompt import messages
from src.processing.mqm.reports import compact_report, absolute_record, restore_report
from src.ui.mqm_model import load_review, review_items, mqm_payload


def test_identical_windows_collect_all_candidates_and_run_once_before_limit():
    data = report(['Саркома лёгкого.', 'Он не пришёл.'], ['A heart attack.', 'He came.'])
    assert len(data['candidates']) == 2
    assert len(data['review_fragments']) == 1
    unit = review_fragments(data)[0]
    assert unit['anchor_group_indices'] == [0, 1]
    assert {'terminology_mismatch', 'negation_mismatch'} <= {s['code'] for s in unit['signals']}
    backend = TestJudge()
    result = analyze(data, backend, max_fragments=1)
    backend.judge.assert_called_once()
    assert result['fragments'][0]['anchor_group_indices'] == [0, 1]
    assert result['metrics']['coverage']['requested_anchors'] == 2
    assert result['metrics']['coverage']['source_word_fraction'] == 1
    assert {'terminology_mismatch', 'negation_mismatch'} <= {s['code'] for s in backend.judge.call_args.args[0]['hints']}
    resumed = TestJudge()
    analyze(data, resumed, previous=result)
    resumed.judge.assert_not_called()
    regenerated = deepcopy(data)
    regenerated['created_at'] = 'a later bookkeeping timestamp'
    regenerated['inputs'] = {'cometkiwi': {'sha256': 'different file formatting'}}
    resumed = TestJudge()
    analyze(regenerated, resumed, previous=result)
    resumed.judge.assert_not_called()


def test_portable_report_has_only_excerpts_and_relative_spans_and_resumes(tmp_path):
    data = report(['😀 Саркома лёгкого.', 'Он не пришёл.'], ['😀 A heart attack.', 'He came.'])
    backend = TestJudge()
    backend.judge = Mock(return_value={'findings': [finding(), finding(1, 'не', 'came')]})
    path, output = tmp_path/'candidates.json', tmp_path/'pair.mqm.json'
    path.write_text(json.dumps(data))
    result = analyze_file(path, output, backend=backend)
    assert result == json.loads(output.read_text())
    assert result['schema_version'] == 2
    assert 'source' not in result and 'target' not in result and 'findings' not in result
    assert 'tokens' not in json.dumps(result)
    fragment = result['fragments'][0]
    assert fragment['source']['text'] == '😀 Саркома лёгкого.\nОн не пришёл.'
    assert 'attempts' not in fragment and 'rejected' not in fragment
    assert fragment['findings'][1]['source_spans'][0]['start'] > len('😀 Саркома лёгкого.')
    for record in fragment['findings']:
        for side in ('source', 'target'):
            for span in record[side+'_spans']:
                assert fragment[side]['text'][span['start']:span['end']] == span['quote']
                assert set(span) == {'start', 'end', 'quote'}
        projected = absolute_record(record, fragment, data)
        assert projected['source_spans'][0]['token_ids']
    assert output.with_suffix('.diagnostics.json').exists()
    resumed = TestJudge()
    analyze_file(path, output, backend=resumed)
    resumed.judge.assert_not_called()


def test_wrong_sentence_ref_is_recovered_only_by_unique_in_window_quote():
    data = report()
    req = make_request(data, 0)
    raw = finding(source_spans=[{'ref': 'S999', 'quote': 'Саркома'}],
                  target_spans=[{'ref': 'T0', 'quote': 'heart attack'}])
    result = validate_judgment({'findings': [raw]}, data, req)
    assert not result['rejected']
    assert result['findings'][0]['source_spans'][0]['index_resolution'] == 'unique_exact_quote_in_fragment'
    raw['source_spans'][0]['ref'] = 'T0'
    assert validate_judgment({'findings': [raw]}, data, req)['rejected']
    duplicate = report(['Да.', 'Да.'], ['Yes.', 'Yes.'])
    raw = finding(source_spans=[{'ref': 'S999', 'quote': 'Да'}], target_spans=[{'ref': 'T0', 'quote': 'Yes'}])
    assert validate_judgment({'findings': [raw]}, duplicate, make_request(duplicate, 0))['rejected']


def test_word_hints_are_bounded_and_do_not_send_whole_token_tables():
    data = report(['Саркома лёгкого. ' + ' '.join(f'слово{i}' for i in range(100))],
                  ['heart attack. ' + ' '.join(f'word{i}' for i in range(100))])
    req = make_request(data, 0)
    for side in ('source', 'target'):
        req['hints'][0][side+'_spans'][0]['token_ids'] = list(range(len(data[side]['tokens'])))
    wire = json.loads(messages(req, include_word_links=True)[1]['content'])
    assert len(wire['word_correspondence_hypotheses']) == 48
    assert wire['word_correspondences_omitted'] > 0
    assert 'tokens' not in wire and 'word_links' not in wire
    assert wire['source']['role'] == 'original' and wire['target']['role'] == 'translation'
    assert wire['source']['sentences'][0]['ref'] == 'S0'


def test_full_text_projection_and_unreviewed_candidates_require_matching_inputs(tmp_path):
    data = report()
    internal = analyze(data, TestJudge(), max_fragments=1)
    compact = compact_report(internal)
    corpus = tmp_path/'data/corpus/pair.json'
    corpus.parent.mkdir(parents=True)
    corpus.write_text(json.dumps({'documents': {s: data[s] for s in ('source', 'target')}}))
    cp = tmp_path/'data/review_candidates/pair.json';cp.parent.mkdir(parents=True)
    cp.write_text(json.dumps(data))
    compact['inputs'] = {kind: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                         for kind, path in [('corpus', corpus), ('candidates', cp)]}
    loaded, docs, candidates, warning = load_review(compact, tmp_path)
    assert docs and candidates and not warning
    items = review_items(loaded, candidates)
    assert len(items) == 3 and items[-1]['status'] == 'not_analyzed'
    payload, picks = mqm_payload(loaded, items, 0, documents=docs, candidates=candidates,
                                mode='Весь текст', selected_finding='__all__', identity='test')
    assert len(payload['panes']['source']['sentences']) == 4
    pieces = [p for r in payload['panes']['source']['sentences'] for p in r['pieces']]
    error = next(p for p in pieces if p['marker'] == 'E1')
    assert error['severity'] == 'major' and picks[error['pick']][0] == 0
    assert any(p['suspect'] and not p['marker'] and p['pick'] for p in pieces)
    corpus.write_text('{}');cp.write_text('{}')
    _, docs, _, warning = load_review(compact, tmp_path)
    assert docs is None and warning


def test_portable_excerpt_works_without_corpus_and_tampered_spans_are_rejected():
    data = report()
    compact = compact_report(analyze(data, TestJudge(), max_fragments=1))
    payload, _ = mqm_payload(compact, compact['fragments'], 0, documents=None, candidates=None,
                             mode='Фрагмент', selected_finding='__all__', identity='test')
    assert any(p['active'] for r in payload['panes']['source']['sentences'] for p in r['pieces'])
    broken = deepcopy(compact)
    broken['fragments'][0]['findings'][0]['source_spans'][0]['start'] += 1
    with pytest.raises(ValueError): restore_report(broken, data)


def test_comet_evaluates_identical_text_pair_once_but_keeps_anchor_windows():
    from src.processing.evaluation.engine import evaluate_windows
    from types import SimpleNamespace
    windows = [deepcopy(t['window']) for t in report(['Он не пришёл.', 'Саркома лёгкого.'], ['He came.', 'Heart attack.'])['semantic_tasks']]
    backend = SimpleNamespace(metadata={'name':'QE'}, predict=Mock(return_value=[{'status':'scored','score':.7}]))
    result = evaluate_windows(windows, backend)
    assert len(backend.predict.call_args.args[0]) == 1
    assert len(result['windows']) == 2
    assert result['inference'] == {'input_windows':2,'unique_model_inputs':1}
    assert all(w['verdict']['score'] == .7 for w in result['windows'])


def test_portable_report_can_be_viewed_without_corpus(tmp_path):
    from streamlit.testing.v1 import AppTest
    compact = compact_report(analyze(report(), TestJudge(), max_fragments=1))
    path = tmp_path/'data/mqm_reports/pair.mqm.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(compact))
    path.with_suffix('.diagnostics.json').write_text('{}')
    app = AppTest.from_string(f'from pathlib import Path\nfrom src.ui.views.mqm import render_mqm\nrender_mqm(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    app.selectbox(key='mqm_view_mode').select('Весь текст').run()
    assert not app.exception
    assert any('нужен корпус' in info.value for info in app.info)


def test_fragment_spans_cannot_be_projected_to_changed_corpus():
    data = report()
    compact = compact_report(analyze(data, TestJudge(), max_fragments=1))
    changed = deepcopy(data)
    changed['source']['sentences'][0]['normalized_text'] = 'Несовместимый текст'
    with pytest.raises(ValueError):
        absolute_record(compact['fragments'][0]['findings'][0], compact['fragments'][0], changed)
