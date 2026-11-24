"""Overlap is pure text processing; Vecalign itself retains an explicit environment."""
import sys,json,subprocess
from unittest.mock import patch
import pytest
from src.processing.alignment.overlaps import overlap_vocabulary,write_overlaps
from src.external_tools.vecalign_tools import ExternalToolsRunner


def test_overlaps_match_vecalign_front_padding_and_unique_sorted_vocabulary(tmp_path):
    source=tmp_path/'source.txt';source.write_text('\ufeff A \n B\nA\n',encoding='utf-8')
    output=tmp_path/'overlaps.txt'
    with patch('src.external_tools.vecalign_tools.subprocess.run',side_effect=AssertionError('must not launch Vecalign')):
        ExternalToolsRunner(tmp_path).generate_overlaps(source,output,4)
    assert output.read_text().splitlines()==['A','A B','A B A','B','B A','PAD']
    assert overlap_vocabulary(['\n'],2)==['BLANK_LINE','PAD']
    assert overlap_vocabulary(['X'],1)==['X']


def test_overlap_limits_and_validation_preserve_previous_file(tmp_path):
    assert overlap_vocabulary(['X'*10001],1)==['X'*10000]
    p=tmp_path/'source.txt';p.write_text('A\n');out=tmp_path/'out';out.write_text('previous')
    for size in [0,-1,True]:
        with pytest.raises(ValueError):write_overlaps(p,out,size)
        assert out.read_text()=='previous'
    with pytest.raises(ValueError):write_overlaps(p,p)
    p.write_text('')
    with pytest.raises(ValueError,match='No input sentences'):write_overlaps(p,out)
    assert out.read_text()=='previous'


def test_missing_vecalign_fails_with_selected_python_and_action(tmp_path):
    runner=ExternalToolsRunner(tmp_path,vecalign_python=sys.executable)
    with patch.object(runner,'run_command',side_effect=subprocess.CalledProcessError(1,[],stderr=b'No module named vecalign')):
        with pytest.raises(RuntimeError,match='No module named vecalign') as error:runner.check_component('vecalign')
    assert sys.executable in str(error.value)
    assert 'Настройки' in str(error.value)


def test_shared_external_settings_reach_runner(tmp_path):
    from src.processing.alignment.stages import _tools
    p=tmp_path/'data/pipeline_settings.json';p.parent.mkdir()
    p.write_text(json.dumps({'external_tools':{'vecalign_python':'/some/python','laser_python':'/laser/python','backend':'native'}}))
    runner=_tools(tmp_path)
    assert runner.vecalign_python=='/some/python'
    assert runner.laser_python=='/laser/python'


def test_worker_overlap_does_not_need_vecalign(tmp_path, monkeypatch):
    from src.ui.worker import run_job
    from src.processing.pipeline.config import SOURCE
    # Reproduce an application environment whose selected Vecalign Python is missing.
    monkeypatch.delenv('VECALIGN_OVERLAP_SCRIPT', raising=False)
    monkeypatch.setenv('VECALIGN_PYTHON', str(tmp_path/'missing-python'))
    monkeypatch.setenv('EXTERNAL_TOOLS_BACKEND', 'native')
    source = tmp_path/f'data/overlaps/input/{SOURCE}.txt'
    source.parent.mkdir(parents=True)
    source.write_text('Первое.\nВторое.\n', encoding='utf-8')
    jobs = tmp_path/'data/ui_jobs'
    jobs.mkdir()
    request = jobs/'test.request.json'
    request.write_text(json.dumps({'id':'test', 'project_root':str(tmp_path),
                                  'stage':'overlap', 'documents':[SOURCE], 'max_size':4}))
    assert run_job(request)==0
    assert json.loads((jobs/'test.status.json').read_text())['state']=='done'
    assert (tmp_path/f'data/overlaps/output/{SOURCE}.txt').read_text(encoding='utf-8').splitlines()==[
        'PAD', 'Второе.', 'Первое.', 'Первое. Второе.']


def test_check_source_entrypoint_reports_missing_dependency(tmp_path):
    script = tmp_path/'vecalign script.py'
    script.write_text('import missing_vecalign_dependency_for_test\n')
    runner = ExternalToolsRunner(tmp_path, backend='native', vecalign_python=sys.executable,
                                 vecalign_script=str(script))
    with pytest.raises(RuntimeError,match='missing_vecalign_dependency_for_test'):
        runner.check_component('vecalign')
    script.write_text("import sys\nassert sys.argv[1:] == ['--help']\n")
    runner.check_component('vecalign')


def test_settings_form_saves_external_environment_for_cli(tmp_path):
    from streamlit.testing.v1 import AppTest
    from src.processing.alignment.stages import _tools
    app=AppTest.from_string(f'from pathlib import Path\nfrom src.ui.views.settings import render_settings\nrender_settings(Path({str(tmp_path)!r}))').run()
    next(n for n in app.text_input if n.label=='Python с Vecalign').set_value(sys.executable)
    next(b for b in app.button if b.label=='Сохранить внешние инструменты').click().run()
    assert not app.exception
    assert _tools(tmp_path).vecalign_python==sys.executable
    assert json.loads((tmp_path/'data/pipeline_settings.json').read_text())['vecalign']['max_size']==4
