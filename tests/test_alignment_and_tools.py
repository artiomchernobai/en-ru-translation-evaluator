import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from src.processing.alignment.sentences import parse_alignment, validate_alignment
from src.processing.alignment.corpus import CorpusBuilder
from src.ui.alignment_view import parse_alignment as ui_parse
from src.external_tools.vecalign_tools import ExternalToolsRunner


@pytest.mark.parametrize('text', ['[0,0]:[0]:1', '[]:[]:1', '[-1]:[0]:1',
    '[0,]:[0]:1', '[0]:[0]:1\n[0]:[1]:1', '[1]:[0]:1', '[0]:[0]:1e999'])
def test_both_parsers_reject_corruption(text, tmp_path):
    path = tmp_path/'bad.align'
    path.write_text(text)
    with pytest.raises(ValueError): ui_parse(text)
    with pytest.raises(ValueError): CorpusBuilder().parse_alignments(path)


def test_complete_groups_and_gaps():
    rows = parse_alignment('[0,1]:[0]:.2\n[2]:[]:0\n[]:[1]:1')
    validate_alignment(rows, 3, 2)
    with pytest.raises(ValueError): validate_alignment(rows, 4, 2)
    with pytest.raises(ValueError): validate_alignment(rows, 2, 2)


def test_incomplete_corpus_preserves_previous_result(tmp_path):
    for lang in ('ru','en'):
        (tmp_path/f'{lang}.json').write_text(json.dumps({'sentences':[
            {'sentence_id':i,'normalized_text':str(i)} for i in range(2)]}))
    (tmp_path/'pair.align').write_text('[0]:[0]:1')
    out = tmp_path/'out.json'
    out.write_text('previous')
    with pytest.raises(ValueError):
        CorpusBuilder().build_corpus(tmp_path/'ru.json',tmp_path/'en.json',tmp_path/'pair.align',out,'ru','en')
    assert out.read_text() == 'previous'


def test_native_arguments_and_atomic_output(tmp_path, monkeypatch):
    root = tmp_path/'project with spaces'
    root.mkdir()
    script = root/'fake align.py'
    script.write_text("import sys\nassert '--src_embed' in sys.argv\nprint('[0]:[0]:0.1')")
    monkeypatch.setenv('VECALIGN_SCRIPT',str(script))
    runner = ExternalToolsRunner(root, vecalign_python=sys.executable)
    out=root/'out.align'
    runner.align(*([root/'a b.txt']*6),out)
    assert out.read_text().strip()=='[0]:[0]:0.1'
    script.write_text('raise SystemExit(3)')
    with pytest.raises(subprocess.CalledProcessError): runner.align(*([root/'a b.txt']*6),out)
    assert out.read_text().strip()=='[0]:[0]:0.1'


def test_wsl_quotes_arguments(tmp_path):
    runner=ExternalToolsRunner(tmp_path,backend='wsl',wsl_project_dir='/mnt/c/my project',vecalign_python='/tools/python')
    with patch('src.external_tools.vecalign_tools.subprocess.run') as run:
        runner.generate_overlaps(tmp_path/'a b.txt',tmp_path/'out.txt')
    command=run.call_args.args[0]
    assert command[:3]==['wsl.exe','bash','-lc']
    assert "cd '/mnt/c/my project'" in command[3]
    assert "'/mnt/c/my project/a b.txt'" in command[3]


def test_corpus_keeps_every_sentence_including_gaps(tmp_path):
    for lang in ('ru','en'):
        (tmp_path/f'{lang}.json').write_text(json.dumps({'sentences':[
            {'sentence_id':i,'normalized_text':f'{lang}{i}'} for i in range(3)]}))
    (tmp_path/'pair.align').write_text('[0,1]:[0]:.1\n[2]:[]:0\n[]:[1,2]:0')
    out=tmp_path/'out.json'
    CorpusBuilder().build_corpus(tmp_path/'ru.json',tmp_path/'en.json',tmp_path/'pair.align',out,'ru','en')
    corpus=json.loads(out.read_text())
    assert corpus['segment_count']==3
    assert [s['alignment_status'] for s in corpus['segments']]==['aligned','unaligned','unaligned']
    for side in ('source','target'):
        assert [i for s in corpus['segments'] for i in s[side]['sentence_ids']]==[0,1,2]
