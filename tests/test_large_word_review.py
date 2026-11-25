"""Large generated alignments belong in the dedicated word-review screen."""
import json
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from src.ui.file_browser import MAX_BYTES, safe_file, store_import
from src.ui.views.review import _read
from test_review_model import sample


def large_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False)
        # Valid JSON whitespace crosses the byte limit without inventing token IDs.
        stream.write('\n' + ' ' * MAX_BYTES)
    assert path.stat().st_size > MAX_BYTES


def test_large_alignment_opens_with_links_and_fragment_navigation(tmp_path):
    path = tmp_path/'data/word_alignment/book.json'
    large_json(path, sample())
    app = AppTest.from_string(f'from pathlib import Path\nfrom src.ui.views.review import render_review\nrender_review(Path({str(tmp_path)!r}))', default_timeout=15)
    with patch('src.ui.views.review.show_word_review') as display:
        app.run()
        assert not app.exception
        assert not app.error
        assert display.call_args.args[0]['panes']['source']['tokens'][1]['partners'] == [1]
        app.selectbox(key='review_mode').set_value('Фрагмент').run()
        assert not app.exception and not app.error
        assert display.call_args.args[0]['panes']['source']['text'] == 'Он читает книгу.'
        app.selectbox(key='review_method').set_value('inter').run()
        assert display.call_args.args[0]['panes']['source']['tokens'][1]['partners'] == []


def test_large_invalid_alignment_is_still_rejected(tmp_path):
    path = tmp_path/'data/word_alignment/book.json'
    data = sample()
    data['source']['tokens'][0]['document_char_start'] = 100
    large_json(path, data)
    stat = path.stat()
    with pytest.raises(ValueError, match='Текст и смещения токена'):
        _read(str(path), stat.st_mtime_ns, stat.st_size)


def test_review_cache_refreshes_after_replacing_file(tmp_path):
    path = tmp_path/'data/word_alignment/book.json'
    large_json(path, sample())
    stat = path.stat()
    assert _read(str(path), stat.st_mtime_ns, stat.st_size)['blocks']
    path.write_text('{}', encoding='utf-8')
    stat = path.stat()
    with pytest.raises(ValueError, match='версии 1.0'):
        _read(str(path), stat.st_mtime_ns, stat.st_size)


def test_preview_import_and_path_restrictions_remain(tmp_path):
    path = tmp_path/'data/word_alignment/book.json'
    large_json(path, sample())
    with pytest.raises(ValueError, match='15 МБ'):
        safe_file(tmp_path, 'data/word_alignment/book.json')
    assert safe_file(tmp_path, 'data/word_alignment/book.json', max_bytes=None) == path
    with pytest.raises(ValueError, match='15 МБ'):
        store_import(tmp_path, 'book.json', b' ' * (MAX_BYTES + 1))
    outside = tmp_path/'outside.json'
    outside.write_text('{}')
    with pytest.raises(ValueError, match='внутри data'):
        safe_file(tmp_path, 'outside.json', max_bytes=None)
    link = tmp_path/'data/word_alignment/link.json'
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='внутри data'):
        safe_file(tmp_path, 'data/word_alignment/link.json', max_bytes=None)
