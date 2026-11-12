"""Exercise window coverage, model adapter and file/UI orchestration without model downloads."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.processing.evaluation.windows import build_windows
from src.processing.evaluation.engine import evaluate_windows
from src.processing.evaluation.io import evaluate_files
from src.processing.evaluation.cometkiwi import CometKiwiBackend


def document(name, count):
    return {'document': name, 'language': 'ru' if name == 'source' else 'en',
            'sentences': [{'sentence_id': f'{name}-{i}', 'normalized_text': f'{name} sentence {i}.'} for i in range(count)]}


def groups():
    return [{'source_indices': [0, 1], 'target_indices': [0], 'cost': .1},
            {'source_indices': [2], 'target_indices': [1], 'cost': .2},
            {'source_indices': [3], 'target_indices': [2, 3], 'cost': .3},
            {'source_indices': [4], 'target_indices': [4], 'cost': .4}]


class RecordingBackend:
    metadata = {'name': 'test-model', 'score_scope': 'entire_window'}
    def __init__(self):
        self.calls = []
    def predict(self, pairs):
        self.calls.append(pairs)
        return [{'status': 'scored', 'score': .75} for _ in pairs]


def test_sliding_windows_are_groups_not_sentence_counts():
    windows = build_windows(document('source', 5), document('target', 5), groups())
    middle = windows[1]
    assert middle['group_indices'] == [0, 1, 2]
    assert middle['source']['sentence_indices'] == [0, 1, 2, 3]
    assert middle['target']['sentence_indices'] == [0, 1, 2, 3]
    assert middle['center_alignment']['source_indices'] == [2]
    assert windows[2]['group_indices'] == [1, 2, 3]
    assert windows[0]['group_indices'] == [0, 1]
    assert windows[-1]['group_indices'] == [2, 3]
    backend = RecordingBackend()
    result = evaluate_windows(windows, backend)
    assert backend.calls[0][1]['src'] == middle['source']['text']
    assert result['windows'][1]['verdict']['scope'] == 'entire_window'
    assert 'verdict' not in windows[1]


def test_gap_retained_and_empty_window_not_sent_to_model():
    alignment = [{'source_indices': [0], 'target_indices': [], 'cost': 0},
                 {'source_indices': [1], 'target_indices': [0], 'cost': .3}]
    source, target = document('source', 2), document('target', 1)
    backend = RecordingBackend()
    result = evaluate_windows(build_windows(source, target, alignment, radius=0), backend)
    assert len(backend.calls[0]) == 1
    assert result['windows'][0]['verdict']['status'] == 'empty_side'
    wider = build_windows(source, target, alignment)
    assert wider[0]['alignment_gap_present']
    assert wider[0]['source']['sentence_indices'] == [0, 1]


@pytest.mark.parametrize('alignment', [
    [{'source_indices': [1], 'target_indices': [0], 'cost': .1}],
    [{'source_indices': [0, 0], 'target_indices': [0], 'cost': .1}],
    [{'source_indices': [0], 'target_indices': [0], 'cost': float('nan')}],
])
def test_invalid_coverage_rejected(alignment):
    with pytest.raises(ValueError):
        build_windows(document('source', 1), document('target', 1), alignment)


def test_preloaded_adapter_never_truncates_and_keeps_result_positions():
    class Tokenizer:
        model_max_length = 512
        def encode(self, text, **kwargs):
            assert kwargs['truncation'] is False
            return list(text)
    model = SimpleNamespace(encoder=SimpleNamespace(tokenizer=Tokenizer(), max_positions=512),
                            predict=Mock(return_value=SimpleNamespace(scores=[.8, .3])))
    backend = CometKiwiBackend(model, max_tokens=10)
    result = backend.predict([{'src': 'short', 'mt': 'small'},
                              {'src': 'x'*11, 'mt': 'small'},
                              {'src': 'last', 'mt': 'last'}])
    assert [r['status'] for r in result] == ['scored', 'too_long', 'scored']
    assert result[2]['score'] == .3
    assert len(model.predict.call_args.args[0]) == 2


def test_files_preserve_exact_alignment_lines_and_provenance(tmp_path):
    source, target, alignment, output = [tmp_path / name for name in ('s.json', 't.json', 'pair.align', 'output.json')]
    source.write_text(json.dumps(document('source', 5)))
    target.write_text(json.dumps(document('target', 5)))
    lines = ['[0, 1]:[0]:0.100000', '[2]:[1]:0.200000', '[3]:[2, 3]:0.300000', '[4]:[4]:0.400000']
    alignment.write_text('\n\n'.join(lines))
    result = evaluate_files(source, target, alignment, output, backend=RecordingBackend())
    assert result == json.loads(output.read_text())
    assert result['windows'][1]['alignments'][2]['vecalign_line'] == lines[2]
    assert result['windows'][1]['source']['sentences'][0]['sentence_id'] == 'source-0'
    assert len(result['inputs']['alignment']['sha256']) == 64
    assert result['summary'] == {'scored': 4}
    with pytest.raises(ValueError, match='overwrite'):
        evaluate_files(source, target, alignment, source, backend=RecordingBackend())


def test_stage_loads_once_for_all_pairs(tmp_path, monkeypatch):
    from src.processing.evaluation import stages
    from src.processing.pipeline.config import SOURCE, TARGETS, folder
    for name in (SOURCE, *TARGETS):
        path = folder(tmp_path, 'normalized') / f'{name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    for target in TARGETS:
        path = folder(tmp_path, 'aligned') / f'{SOURCE}_{target}.align'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('[0]:[0]:0.1')
    load, evaluate = Mock(return_value=RecordingBackend()), Mock()
    monkeypatch.setattr(stages, 'configured_backend', load)
    monkeypatch.setattr(stages, 'evaluate_files', evaluate)
    stages.stage_cometkiwi(tmp_path, [SOURCE, *TARGETS], 3)
    load.assert_called_once_with(tmp_path)
    assert evaluate.call_count == 2
    assert all(call.kwargs['backend'] is load.return_value for call in evaluate.call_args_list)


def test_main_standalone_does_not_run_segmentation(monkeypatch):
    from src.processing.pipeline import cli
    from src.processing.evaluation import stages
    run = Mock()
    monkeypatch.setattr(stages, 'stage_cometkiwi', run)
    monkeypatch.setattr(cli, 'SentenceSegmenter', Mock(side_effect=AssertionError('must not segment')))
    cli.main(['--stage', 'cometkiwi'])
    run.assert_called_once_with(cli.BASE_DIR, list(cli.DOCUMENTS), 3)


def test_quality_screen_displays_window_and_score(tmp_path):
    from streamlit.testing.v1 import AppTest
    windows = build_windows(document('source', 5), document('target', 5), groups())
    result = evaluate_windows(windows, RecordingBackend())
    result.update({'source_document': 'source', 'target_document': 'target'})
    path = tmp_path / 'data/translation_quality/test.cometkiwi.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result))
    app = AppTest.from_string(f'from pathlib import Path\nfrom src.ui.views.quality import render_quality\nrender_quality(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    assert app.metric[0].value == '0.7500'
    assert any('source sentence 0.' in t.value for t in app.text)
    app.selectbox(key='quality_window').select(1).run()
    assert not app.exception
    assert any('[2]:[1]:0.2' in code.value for code in app.code)
    app.selectbox(key='quality_sort').select('Без оценки').run()
    assert not app.exception
    assert any('Нет окон' in info.value for info in app.info)


def test_backend_failure_preserves_existing_output(tmp_path):
    source, target, alignment, output = [tmp_path / name for name in ('s.json', 't.json', 'pair.align', 'output.json')]
    source.write_text(json.dumps(document('source', 1)))
    target.write_text(json.dumps(document('target', 1)))
    alignment.write_text('[0]:[0]:0.1')
    output.write_text('previous successful result')
    backend = RecordingBackend()
    backend.predict = Mock(side_effect=RuntimeError('model failed'))
    with pytest.raises(RuntimeError, match='model failed'):
        evaluate_files(source, target, alignment, output, backend=backend)
    assert output.read_text() == 'previous successful result'
