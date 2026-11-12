from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.processing.alignment.words import WordAligner, BlockTooLongError
from src.processing.alignment.io import align_word_files
from src.processing.alignment.simalign_backend import SimAlignBackend


class Backend:
    metadata = {'name': 'test-backend'}

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def align_tokens(self, source, target):
        self.calls.append((source, target))
        return self.result if self.result is not None else {'itermax': [(0, 0)], 'inter': [(0, 0)]}


def document(name, texts, language='ru'):
    return {'document': name, 'language': language, 'sentence_count': len(texts),
            'sentences': [{'sentence_id': 100 + i * 7, 'normalized_text': text} for i, text in enumerate(texts)]}


def group(source, target, cost=.2):
    return {'source_indices': source, 'target_indices': target, 'cost': cost}


def test_grouped_tokens_links_offsets_and_input_immutability():
    src = document('another-book', ['— Привет, мир!', 'Не повторяй: мир.'])
    tgt = document('any-translation', ['Hello, world!', 'World.'], 'en')
    groups = [group([0, 1], [0, 1])]
    before = deepcopy((src, tgt, groups))
    backend = Backend({'itermax': [(1, 0), (3, 2), (3, 3), (8, 4)], 'inter': [(3, 2)]})
    result = WordAligner(backend).align(src, tgt, groups)
    assert (src, tgt, groups) == before
    assert backend.calls[0][0] == ['—', 'Привет', ',', 'мир', '!', 'Не', 'повторяй', ':', 'мир', '.']
    assert result['blocks'][0]['source']['sentence_ids'] == [100, 107]
    assert result['source']['tokens'][3]['aligned_token_ids'] == [2, 3]
    assert result['target']['tokens'][2]['aligned_token_ids'] == [3]
    assert result['source']['tokens'][0]['alignment_status'] == 'unlinked'
    for label in ('source', 'target'):
        doc = result[label]
        for token in doc['tokens']:
            sentence = doc['sentences'][token['sentence_index']]['normalized_text']
            assert sentence[token['sentence_char_start']:token['sentence_char_end']] == token['text']
            assert doc['text'][token['document_char_start']:token['document_char_end']] == token['text']
        side = result['blocks'][0][label]
        for span in side['token_spans']:
            assert side['text'][span['char_start']:span['char_end']] == doc['tokens'][span['token_id']]['text']


def test_gaps_never_call_model_and_keep_tokens():
    backend = Backend()
    result = WordAligner(backend).align(document('s', ['Да.']), document('t', ['Yes.'], 'en'),
                                        [group([0], []), group([], [0])])
    assert not backend.calls
    assert result['summary']['sentence_gap_blocks'] == 2
    assert all(t['alignment_status'] == 'sentence_unaligned' for t in result['source']['tokens'])
    assert all(not b['word_alignment']['methods'] for b in result['blocks'])


def test_empty_documents_need_no_model():
    backend = Backend()
    result = WordAligner(backend).align(document('s', []), document('t', [], 'en'), [])
    assert not backend.calls
    assert result['summary']['block_count'] == 0


@pytest.mark.parametrize('groups', [[group([0], [0])], [group([0], [0]), group([0, 1], [1])],
    [group([1, 0], [0, 1])], [group([False, 1], [0, 1])]])
def test_invalid_sentence_coverage_rejected_before_inference(groups):
    backend = Backend()
    with pytest.raises(ValueError):
        WordAligner(backend).align(document('s', ['Один', 'Два']), document('t', ['One', 'Two'], 'en'), groups)
    assert not backend.calls


@pytest.mark.parametrize('mutate', [lambda d: d.update(sentence_count=9),
    lambda d: d['sentences'][0].update(normalized_text=''),
    lambda d: d['sentences'][0].update(normalized_text='one\ntwo'),
    lambda d: d['sentences'][0].pop('sentence_id'),
    lambda d: d.update(language=''),
    lambda d: d['sentences'][1].update(sentence_id=100)])
def test_invalid_normalized_documents_rejected(mutate):
    src = document('s', ['Один', 'Два'])
    mutate(src)
    with pytest.raises(ValueError):
        WordAligner(Backend()).align(src, document('t', ['One'], 'en'), [group([0, 1], [0])])


@pytest.mark.parametrize('result', [{'itermax': [(999, 0)]}, {'itermax': [(-1, 0)]},
    {'itermax': [(True, 0)]}, {'itermax': [(0, '0')]}, {'itermax': [(0,)]}, {'inter': []}])
def test_invalid_backend_output_is_not_saved_as_success(result):
    with pytest.raises(ValueError):
        WordAligner(Backend(result)).align(document('s', ['Да']), document('t', ['Yes'], 'en'), [group([0], [0])])


def test_long_blocks_marked_unprocessed_not_as_omissions():
    backend = Backend()
    backend.align_tokens = Mock(side_effect=BlockTooLongError('limit exceeded'))
    result = WordAligner(backend).align(document('s', ['Длинный текст']), document('t', ['Long text'], 'en'), [group([0], [0])])
    assert result['blocks'][0]['word_alignment']['status'] == 'skipped_too_long'
    assert result['summary']['skipped_too_long_blocks'] == 1
    assert result['source']['tokens'][0]['alignment_status'] == 'not_processed'
    assert not result['blocks'][0]['word_alignment']['source_unlinked_token_ids']


def test_word_indices_can_link_many_to_one_and_duplicate_links_are_deduplicated():
    result = WordAligner(Backend({'itermax': [(0, 0), (1, 0), (1, 0)]})).align(
        document('s', ['два слова']), document('t', ['word'], 'en'), [group([0], [0])])
    assert result['target']['tokens'][0]['aligned_token_ids'] == [0, 1]
    assert len(result['blocks'][0]['word_alignment']['methods']['itermax']) == 2


def write_pair(tmp_path):
    src, tgt, alignment = (tmp_path / name for name in ['Книга source.json', 'Other translation.json', 'pairs.align'])
    src.write_text(json.dumps(document('other-work', ['Привет!']), ensure_ascii=False), encoding='utf-8-sig')
    tgt.write_text(json.dumps(document('new-translator', ['Hello!'], 'en')), encoding='utf-8')
    alignment.write_text('[0]:[0]:.2\n', encoding='utf-8-sig')
    return src, tgt, alignment


def test_file_adapter_arbitrary_names_atomic_failure_and_repeatability(tmp_path):
    src, tgt, alignment = write_pair(tmp_path)
    output = tmp_path / 'any directory' / 'any name.json'
    aligner = WordAligner(Backend())
    align_word_files(src, tgt, alignment, output, aligner=aligner)
    before = output.read_bytes()
    result = json.loads(before)
    assert result['source']['document_id'] == 'other-work'
    assert len(result['inputs']['source']['sha256']) == 64
    align_word_files(src, tgt, alignment, output, aligner=aligner)
    assert output.read_bytes() == before
    with pytest.raises(ValueError):
        align_word_files(src, tgt, alignment, output, aligner=WordAligner(Backend({'itermax': [(999, 0)]})))
    assert output.read_bytes() == before
    with pytest.raises(ValueError):
        align_word_files(src, tgt, alignment, src, aligner=aligner)


def test_stage_and_main_write_each_pair_and_reuse_backend(tmp_path, monkeypatch):
    from src.processing.pipeline.config import SOURCE, TARGETS, folder
    from src.processing.pipeline.runner import run_stage
    import src.processing.pipeline.cli as processing_main
    backend = Backend()
    factory = Mock(return_value=WordAligner(backend))
    monkeypatch.setattr('src.processing.alignment.io.configured_word_aligner', factory)
    for name in (SOURCE, *TARGETS):
        path = folder(tmp_path, 'normalized') / f'{name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document(name, ['Text'], 'ru' if name == SOURCE else 'en')))
    for name in TARGETS:
        path = folder(tmp_path, 'aligned') / f'{SOURCE}_{name}.align'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('[0]:[0]:.2\n')
    run_stage(tmp_path, 'word_align', [SOURCE, *TARGETS], 3)
    assert len(backend.calls) == 2
    factory.assert_called_once()
    for name in TARGETS:
        assert (folder(tmp_path, 'word_alignment') / f'{SOURCE}_{name}.json').is_file()
    monkeypatch.setattr(processing_main, 'NORMALIZED_DIR', folder(tmp_path, 'normalized'))
    monkeypatch.setattr(processing_main, 'ALIGNED_DIR', folder(tmp_path, 'aligned'))
    monkeypatch.setattr(processing_main, 'WORD_ALIGNMENT_DIR', folder(tmp_path, 'word_alignment'))
    monkeypatch.setattr(processing_main, 'create_directories', Mock())
    monkeypatch.setattr(processing_main, 'SentenceSegmenter', Mock(side_effect=AssertionError('must not segment')))
    processing_main.main(['--stage', 'word_align'])
    assert len(backend.calls) == 4
    from src.ui.jobs import write_json, read_status
    from src.ui.worker import run_job
    request = folder(tmp_path, 'jobs') / 'word-test.request.json'
    write_json(request, {'id': 'word-test', 'project_root': str(tmp_path), 'stage': 'word_align',
                         'documents': [SOURCE, *TARGETS], 'max_size': 3})
    assert run_job(request) == 0
    assert read_status(tmp_path, 'word-test')['state'] == 'done'
    assert len(backend.calls) == 6


def test_simalign_adapter_checks_limit_before_inference():
    class Tokenizer:
        model_max_length = 5
        def tokenize(self, word): return [word]
        def __call__(self, tokens, **kwargs):
            assert kwargs['truncation'] is False
            return {'input_ids': [101, *range(len(tokens)), 102]}
    aligner = Mock()
    aligner.embed_loader = SimpleNamespace(tokenizer=Tokenizer(), emb_model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=5)))
    backend = SimAlignBackend()
    backend._aligner = aligner
    with pytest.raises(BlockTooLongError): backend.align_tokens(['a'] * 4, ['b'])
    aligner.get_word_aligns.assert_not_called()
    backend.align_tokens(['a'], ['b'])
    aligner.get_word_aligns.assert_called_once_with(['a'], ['b'])


def test_interface_exposes_word_alignment_stage():
    from streamlit.testing.v1 import AppTest
    process = Mock()
    process.poll.return_value = None
    from unittest.mock import patch
    with patch('src.ui.jobs.start_job', return_value={'id': 'words', 'stage': 'word_align', 'process': process}) as start:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/ui/app.py'), default_timeout=10).run()
        assert not app.exception
        app.button(key='stage_word_align').click().run()
        assert not app.exception
        assert start.call_args.args[1] == 'word_align'


def test_selected_method_controls_reverse_links_and_unlinked_tokens():
    backend = Backend({'itermax': [(0, 0)], 'inter': [(1, 0)]})
    result = WordAligner(backend, selected_method='inter').align(
        document('s', ['один два']), document('t', ['two'], 'en'), [group([0], [0])])
    assert result['selected_method'] == 'inter'
    assert result['source']['tokens'][0]['alignment_status'] == 'unlinked'
    assert result['source']['tokens'][1]['alignment_status'] == 'linked'
    assert result['target']['tokens'][0]['aligned_token_ids'] == [1]
    assert result['blocks'][0]['word_alignment']['source_unlinked_token_ids'] == [0]
    assert result['blocks'][0]['word_alignment']['methods']['itermax'][0]['source_token_id'] == 0
