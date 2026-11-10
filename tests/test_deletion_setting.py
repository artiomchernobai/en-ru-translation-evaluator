import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch, Mock

import pytest
from src.external_tools.vecalign_tools import ExternalToolsRunner, validate_deletion_percentile
from src.ui.jobs import start_job, read_status, tail_log
from src.ui.pipeline import run_stage


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), float('inf'), True])
def test_invalid_penalty_rejected_before_start(tmp_path, value):
    with patch('src.ui.jobs.subprocess.Popen') as popen:
        with pytest.raises(ValueError): start_job(tmp_path,'align',['original-rus'],3,value)
        popen.assert_not_called()
    assert not (tmp_path/'data').exists()


@pytest.mark.parametrize('value', [0.0,0.05,0.1,0.2,1.0])
def test_native_and_wsl_receive_selected_penalty(tmp_path, value):
    for backend in ['native','wsl']:
        runner=ExternalToolsRunner(tmp_path,backend=backend,wsl_project_dir='/mnt/d/project')
        with patch('src.external_tools.vecalign_tools.subprocess.run') as run:
            runner.align(*([tmp_path/'input']*6),tmp_path/'output',del_percentile_frac=value)
        args=run.call_args.args[0]
        command=args if backend=='native' else shlex.split(args[-1].split(' && ',1)[1])
        assert command[command.index('--del_percentile_frac')+1]==str(value)


def test_full_pipeline_forwards_penalty_only_to_alignment(tmp_path):
    stages=['clean','segment','normalize','overlap','embed','align','word_align','corpus']
    with patch.dict('src.ui.pipeline.FUNCTIONS',{name:Mock() for name in stages}):
        from src.ui.pipeline import FUNCTIONS
        run_stage(tmp_path,'all',['original-rus','glenny-translation-en'],3,0.05)
        FUNCTIONS['align'].assert_called_once_with(tmp_path,['original-rus','glenny-translation-en'],3,del_percentile_frac=0.05)
        FUNCTIONS['clean'].assert_called_once_with(tmp_path,['original-rus','glenny-translation-en'],3)


@pytest.mark.parametrize('legacy', [False,True])
def test_real_worker_alignment_receives_request_setting(tmp_path, monkeypatch, legacy):
    (tmp_path/'src').symlink_to(Path(__file__).resolve().parents[1]/'src',target_is_directory=True)
    for category,suffix in [('overlaps/input','txt'),('overlaps/output','txt'),('embeddings','emb')]:
        directory=tmp_path/'data'/category;directory.mkdir(parents=True)
        for doc in ['original-rus','glenny-translation-en']:(directory/f'{doc}.{suffix}').write_text('x\n')
    script=tmp_path/'fake align.py'
    expected=.2 if legacy else .1
    script.write_text(f"import sys\nassert sys.argv[sys.argv.index('--del_percentile_frac')+1] == '{expected}'\nprint('[0]:[0]:0.1')\n")
    monkeypatch.setenv('EXTERNAL_TOOLS_BACKEND','native')
    monkeypatch.setenv('VECALIGN_PYTHON',sys.executable)
    monkeypatch.setenv('VECALIGN_SCRIPT',str(script))
    if legacy:
        from src.ui.jobs import job_paths,write_json
        job_id='legacy';request_path,_,_=job_paths(tmp_path,job_id)
        write_json(request_path,{'id':job_id,'stage':'align','documents':['original-rus','glenny-translation-en'],'max_size':3,'project_root':str(tmp_path)})
        env=os.environ.copy();env['PYTHONPATH']=str(tmp_path)
        result=subprocess.run([sys.executable,'-m','src.ui.worker','--job-file',str(request_path)],cwd=tmp_path,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    else:
        job=start_job(tmp_path,'align',['original-rus','glenny-translation-en'],3,expected)
        job_id=job['id']
        assert job['process'].wait(timeout=15)==0,tail_log(tmp_path,job_id)
        request=json.loads((tmp_path/f'data/ui_jobs/{job_id}.request.json').read_text())
        assert request['del_percentile_frac']==expected
        assert '--del_percentile_frac 0.1' in tail_log(tmp_path,job_id)
    assert read_status(tmp_path,job_id)['state']=='done'
    assert (tmp_path/'data/aligned/original-rus_glenny-translation-en.align').read_text().strip()=='[0]:[0]:0.1'


@pytest.mark.parametrize('stage', ['align','all'])
def test_interface_submits_chosen_penalty(stage):
    from streamlit.testing.v1 import AppTest
    process=Mock()
    process.poll.return_value=None
    with patch('src.ui.jobs.start_job',return_value={'id':'ui-setting-test','stage':stage,'process':process}) as start:
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/ui/app.py'), default_timeout=10).run()
        assert not app.exception
        app.number_input(key='del_percentile_frac').set_value(0.1).run()
        app.button(key='stage_align' if stage=='align' else 'all').click().run()
        assert not app.exception
        assert start.call_args.args[1]==stage
        assert start.call_args.kwargs['del_percentile_frac']==0.1
