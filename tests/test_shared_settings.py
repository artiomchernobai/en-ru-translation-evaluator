"""Common configuration, selected review and abstaining token matching."""
import json
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from src.processing.pipeline.settings import read_settings
from src.processing.pipeline import cli,runner
from src.processing.alignment.simalign_backend import SimAlignBackend
from src.processing.alignment.words import WordAligner
from test_mqm import report,TestJudge
from src.processing.mqm.engine import analyze


def test_cli_and_runner_share_saved_options(tmp_path,monkeypatch):
    path=tmp_path/'data/pipeline_settings.json';path.parent.mkdir()
    path.write_text(json.dumps({'vecalign':{'max_size':4,'deletion_percentile':.1}}))
    stage=Mock();monkeypatch.setitem(runner.FUNCTIONS,'align',stage)
    cli.main(['--root',str(tmp_path),'--stage','align'])
    stage.assert_called_once_with(tmp_path,read_settings(tmp_path)['documents'],4,del_percentile_frac=.1)
    stage.reset_mock()
    cli.main(['--root',str(tmp_path),'--stage','align','--max-size','3','--deletion-percentile','.2'])
    stage.assert_called_once_with(tmp_path,read_settings(tmp_path)['documents'],3,del_percentile_frac=.2)
    assert read_settings(tmp_path)['vecalign']['max_size']==4


def test_token_map_scores_and_null_matches_use_one_encoder_pass():
    class Tokenizer:
        model_max_length=512
        def tokenize(self,word):return [word]
        def __call__(self,tokens,**kwargs):return {'input_ids':[101,*range(len(tokens)),102]}
    # Returned SimAlign scores are scaled cosine; one token has no adequate counterpart.
    cosine=np.array([[.9,.1],[.2,.3]])
    aligner=SimpleNamespace(embed_loader=SimpleNamespace(tokenizer=Tokenizer(),emb_model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))),model='test')
    aligner.get_similarity=lambda a,b:(cosine+1)/2
    def infer(a,b):
        aligner.get_similarity(None,None)
        return {m:[(0,0),(1,1)] for m in ('itermax','inter','mwmf')}
    aligner.get_word_aligns=Mock(side_effect=infer)
    backend=SimAlignBackend();backend._aligner=aligner
    doc=lambda name,lang,text:{'document':name,'language':lang,'sentences':[{'sentence_id':0,'normalized_text':text}]}
    result=WordAligner(backend).align(doc('s','ru','один два'),doc('t','en','one two'),[{'source_indices':[0],'target_indices':[0],'cost':.1}])
    aligner.get_word_aligns.assert_called_once()
    word=result['blocks'][0]['word_alignment']
    assert word['token_map']['source'][1]['status']=='filter_rejected'
    assert word['link_evidence']['itermax'][1]['reasons']==['low_similarity']
    assert word['token_map']['source'][1]['matches']==[]
    assert word['coverage']['symmetric']==.5
    assert word['methods']['itermax'][0]['similarity']>.89
    assert len(word['token_map']['source'][1]['alternatives'])==2


def test_selected_candidate_runs_only_one_window():
    backend=TestJudge()
    result=analyze(report(),backend,group_indices=[1],max_fragments=1)
    backend.judge.assert_called_once()
    assert result['fragments'][0]['group_index']==1
    assert result['selection']['max_fragments']==1


def test_selected_stage_uses_separate_output(tmp_path,monkeypatch):
    from src.processing.mqm import stages
    from src.processing.pipeline.config import SOURCE,TARGETS
    target=TARGETS[0];pair=f'{SOURCE}_{target}'
    for directory,name in [('review_candidates',f'{pair}.candidates.json'),('corpus',f'{pair}.json')]:
        p=tmp_path/'data'/directory/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    judge=Mock();monkeypatch.setattr(stages,'configured_judge',Mock(return_value=judge))
    run=Mock(return_value={'status':'complete'});monkeypatch.setattr(stages,'analyze_file',run)
    stages.analyze_candidate(tmp_path,7,target)
    assert run.call_args.kwargs['group_indices']==[7]
    assert run.call_args.args[1].name.endswith('.selected-7.json')


def test_all_run_settings_are_on_settings_page(tmp_path):
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string(f'from pathlib import Path\nfrom src.ui.app import main\nmain(Path({str(tmp_path)!r}))').run()
    app.button(key='nav_Настройки').click().run()
    assert not app.exception
    labels=[n.label for n in app.number_input]
    assert 'Минимальное сходство связи' in labels
    assert 'Перцентиль пропуска Vecalign' in labels
    app.button(key='nav_MQM').click().run()
    assert not app.exception
    assert not app.number_input


def test_candidate_button_submits_exact_group_without_batch(tmp_path):
    from streamlit.testing.v1 import AppTest
    data=report();path=tmp_path/'data/review_candidates/test.candidates.json'
    path.parent.mkdir(parents=True);path.write_text(json.dumps(data))
    process=Mock();process.poll.return_value=None
    with patch('src.ui.jobs.start_job',return_value={'id':'selected','stage':'mqm','process':process}) as start:
        app=AppTest.from_string(f'from pathlib import Path\nfrom src.ui.views.candidates import render_candidates\nrender_candidates(Path({str(tmp_path)!r}))').run()
        app.button(key='candidate_mqm').click().run()
        assert not app.exception
        start.assert_called_once()
        assert start.call_args.kwargs['selection']=={'group_index':0,'target':'target','strategy':'fixed'}


def test_settings_form_persists_common_options(tmp_path):
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string(f'from pathlib import Path\nfrom src.ui.views.settings import render_settings\nrender_settings(Path({str(tmp_path)!r}))').run()
    next(n for n in app.number_input if n.label=='Максимальный размер группы Vecalign').set_value(4)
    next(b for b in app.button if b.label=='Сохранить общие настройки').click().run()
    assert not app.exception
    assert read_settings(tmp_path)['vecalign']=={'max_size':4,'deletion_percentile':.1}
