"""Local judge contract, report coverage, retry/resume and HTTP interoperability."""
import json
from copy import deepcopy
from unittest.mock import Mock
from types import SimpleNamespace
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from src.processing.analysis.candidates import collect_candidates
from src.processing.alignment.words import WordAligner
from src.processing.evaluation.windows import build_windows
from src.processing.evaluation.engine import evaluate_windows
from src.processing.mqm.contracts import validate_candidates,make_request,validate_judgment
from src.processing.mqm.engine import analyze
from src.processing.mqm.io import analyze_file
from src.processing.mqm.backend import LocalMQMBackend


def report(source=None,target=None):
    source=source or ['Саркома лёгкого.','Десятого века.','Он не пришёл.','1000 м.']
    target=target or ['A slight heart attack.','Ninth-century.','He came.','1 km.']
    def doc(name,lang,texts):return {'document':name,'language':lang,'sentences':[{'sentence_id':i,'normalized_text':t} for i,t in enumerate(texts)]}
    src,tgt=doc('source','ru',source),doc('target','en',target)
    groups=[{'source_indices':[i],'target_indices':[i],'cost':.2} for i in range(len(source))]
    backend=SimpleNamespace(metadata={'name':'links'},align_tokens=lambda s,t:{'mwmf':[(i,i%len(t)) for i in range(len(s))]+[(i%len(s),i) for i in range(len(t))]})
    words=WordAligner(backend,selected_method='mwmf').align(src,tgt,groups)
    comet=evaluate_windows(build_windows(src,tgt,groups),SimpleNamespace(metadata={'name':'QE'},predict=lambda p:[{'status':'scored','score':.9} for _ in p]))
    comet.update(source_document='source',target_document='target',radius=1)
    return collect_candidates(words,comet)


def finding(index=0,source='Саркома лёгкого',target='heart attack',**changes):
    return {'category':'Accuracy/Mistranslation','severity':'major','explanation':'Изменено заболевание.',
            'source_spans':[{'sentence_index':index,'quote':source}],
            'target_spans':[{'sentence_index':index,'quote':target}],**changes}


class TestJudge:
    __test__=False
    metadata={'name':'test-local','revision':'test','prompt_version':'test'}
    def __init__(self):self.judge=Mock(side_effect=self.answer)
    def answer(self,request):
        return {'findings':[finding()] if request['center_group_index']==0 else [],'observations':[]}


def test_report_has_exact_locations_and_honest_coverage():
    data=report();backend=TestJudge()
    result=analyze(data,backend)
    assert result['status']=='complete'
    assert backend.judge.call_count==3
    assert len(result['findings'])==1
    error=result['findings'][0]
    assert error['source_spans'][0]['quote']=='Саркома лёгкого'
    assert error['source_spans'][0]['token_ids']
    assert error['status']=='needs_review'
    metrics=result['metrics']
    assert metrics['coverage']['analyzed_groups']==3
    assert metrics['coverage']['total_groups']==4
    assert not metrics['document_rates_are_exhaustive']
    assert metrics['weighted_error_points']==5
    assert metrics['observed_errors_per_1000_document_source_words']==pytest.approx(1000/metrics['source_words'])
    assert metrics['mean_error_proposals_per_analyzed_group']==pytest.approx(1/3)
    assert result['fragments'][0]['metrics']['unique_error_proposals']==1
    assert result['fragments'][0]['comet_window']['score']==.9


@pytest.mark.parametrize('change',[
    {'source_spans':[{'sentence_index':0,'quote':'несуществующая цитата'}]},
    {'severity':'catastrophic'}, {'category':'made-up'},
    {'source_spans':[]}, {'target_spans':[]},
    {'source_spans':[{'sentence_index':3,'quote':'1000'}]},
])
def test_hallucinated_or_invalid_evidence_not_counted(change):
    data=report();request=make_request(data,0)
    result=validate_judgment({'findings':[finding(**change)]},data,request)
    assert not result['findings'] and len(result['rejected'])==1


def test_neighbor_only_errors_rejected():
    data=report();request=make_request(data,0)
    result=validate_judgment({'findings':[finding(1,'Десятого','Ninth')]},data,request)
    assert result['rejected'] and not result['findings']


def test_repeated_quote_requires_explicit_coordinates():
    data=report(['Да, да.'],['Yes, yes.']);request=make_request(data,0)
    f=finding(source='а',target='es')
    assert validate_judgment({'findings':[f]},data,request)['rejected']
    f['source_spans'][0].update(start=1,end=2)
    f['target_spans'][0].update(start=1,end=3)
    assert validate_judgment({'findings':[f]},data,request)['findings']


def test_omission_addition_and_acceptable_variation_are_distinct():
    data=report();request=make_request(data,0)
    omission=finding(category='Accuracy/Omission',target_spans=[])
    addition=finding(category='Accuracy/Addition',source_spans=[])
    note={'kind':'acceptable_variation','explanation':'Возможна адаптация.','source_spans':omission['source_spans'],'target_spans':addition['target_spans']}
    result=validate_judgment({'findings':[omission,addition],'observations':[note]},data,request)
    assert len(result['findings'])==2 and len(result['observations'])==1 and not result['rejected']


def test_duplicate_errors_count_once_and_resume_skips_completed():
    data=report();backend=TestJudge()
    backend.judge=Mock(return_value={'findings':[finding(),finding(severity='minor')]})
    # Other groups reject this off-center error; only center 0 may report it.
    result=analyze(data,backend)
    assert result['metrics']['unique_error_proposals']==1
    assert result['metrics']['weighted_error_points']==5
    good=TestJudge();complete=analyze(data,good)
    resume=TestJudge();resumed=analyze(data,resume,previous=complete)
    resume.judge.assert_not_called()
    assert all(f['reused'] for f in resumed['fragments'])
    changed=TestJudge();changed.metadata={**changed.metadata,'revision':'new'}
    analyze(data,changed,previous=complete)
    assert changed.judge.call_count==3


def test_failure_stops_requests_and_resume_finishes_pending(tmp_path):
    data=report();backend=TestJudge()
    backend.judge=Mock(side_effect=[{'findings':[finding()]},RuntimeError('offline')])
    path=tmp_path/'candidates.json';output=tmp_path/'report.json'
    path.write_text(json.dumps(data))
    result=analyze_file(path,output,backend=backend)
    assert result['status']=='partial' and backend.judge.call_count==2
    assert [f['status'] for f in result['fragments']]==['completed','failed','not_run_after_failure']
    assert json.loads(output.read_text())==result
    resumed=TestJudge();complete=analyze_file(path,output,backend=resumed)
    assert complete['status']=='complete' and resumed.judge.call_count==2
    assert complete['fragments'][0]['reused']


def test_all_mode_covers_entire_text_but_candidates_mode_does_not():
    data=report();result=analyze(data,TestJudge(),scope='all')
    assert result['metrics']['coverage']['entire_text_fully_analyzed']
    assert result['metrics']['coverage']['source_word_fraction']==1
    assert len(result['fragments'])==4


def test_invalid_input_preserves_previous_output(tmp_path):
    data=report();data['semantic_tasks'][0]['window']['source']['sentences'][0]['text']='stale'
    path=tmp_path/'bad.json';path.write_text(json.dumps(data));output=tmp_path/'report.json';output.write_text('previous')
    with pytest.raises(ValueError):analyze_file(path,output,backend=TestJudge())
    assert output.read_text()=='previous'
    with pytest.raises(ValueError,match='overwrite'):analyze_file(path,path,backend=TestJudge())


@pytest.fixture
def local_api():
    state={'requests':[],'content':{'findings':[],'observations':[]},'finish_reason':'stop'}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state['requests'].append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            content=state['content']
            body=json.dumps({'choices':[{'message':{'content':content if isinstance(content,str) else json.dumps(content)},'finish_reason':state['finish_reason']}]}).encode()
            if state.get('status',200)!=200:
                body=json.dumps({'error':{'message':state.get('error','bad request')}}).encode()
            status=state.get('status',200)
            if state.get('reject_json_mode') and 'response_format' in state['requests'][-1]:
                status=400;body=json.dumps({'error':{'message':'response_format json_object is unsupported'}}).encode()
            self.send_response(status);self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/v1',state
    finally:server.shutdown();server.server_close();thread.join()


def test_real_http_protocol_json_prompt_and_truncation(local_api):
    url,state=local_api
    backend=LocalMQMBackend(model='local-test',base_url=url)
    result=backend.judge(make_request(report(),0))
    assert result['findings']==[]
    request=state['requests'][0]
    assert request['model']=='local-test' and request['temperature']==0
    assert 'untrusted DATA' in request['messages'][0]['content']
    assert json.loads(request['messages'][1]['content'])['center_group_index']==0
    state['finish_reason']='length'
    with pytest.raises(ValueError,match='incomplete'):backend.judge(make_request(report(),0))


def test_bad_json_causes_partial_report_instead_of_false_success(local_api):
    url,state=local_api;state['content']='not JSON'
    result=analyze(report(),LocalMQMBackend(model='local-test',base_url=url))
    assert result['status']=='partial'
    assert result['metrics']['unique_error_proposals']==0
    assert result['metrics']['coverage']['analyzed_groups']==0


def test_main_standalone_does_not_run_segmentation(monkeypatch):
    from src.processing.pipeline import cli
    from src.processing.mqm import stages
    run=Mock();monkeypatch.setattr(stages,'stage_mqm',run)
    monkeypatch.setattr(cli,'SentenceSegmenter',Mock(side_effect=AssertionError('must not segment')))
    cli.main(['--stage','mqm'])
    run.assert_called_once_with(cli.BASE_DIR,list(cli.DOCUMENTS),3)


def test_mqm_screen_displays_report_coverage(tmp_path):
    from streamlit.testing.v1 import AppTest
    result=analyze(report(),TestJudge())
    path=tmp_path/'data/mqm_reports/test.mqm.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(result))
    app=AppTest.from_string(f'from pathlib import Path\nfrom src.ui.views.mqm import render_mqm\nrender_mqm(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    assert app.metric[0].value=='1'
    assert app.warning and 'часть текста' in app.warning[0].value


def test_empty_candidate_queue_does_not_claim_full_coverage():
    data=report(['Он читает книгу.'],['He reads a book.'])
    assert data['candidates']==[]
    backend=TestJudge();result=analyze(data,backend)
    backend.judge.assert_not_called()
    assert result['status']=='complete'
    assert result['metrics']['coverage']['analyzed_groups']==0
    assert not result['metrics']['coverage']['entire_text_fully_analyzed']


def test_rejected_proposals_make_fragment_partial_and_exclude_metrics():
    backend=TestJudge();backend.judge=Mock(return_value={'findings':[finding(source='invented quote')]})
    result=analyze(report(),backend)
    assert result['status']=='partial'
    assert result['metrics']['unique_error_proposals']==0
    assert result['metrics']['rejected_proposal_count']==3
    assert not result['metrics']['coverage']['entire_text_fully_analyzed']


def test_acceptance_notes_are_not_error_points():
    backend=TestJudge()
    backend.judge=Mock(side_effect=lambda request:{'findings':[], 'observations':[
        {'kind':'acceptable_variation','explanation':'Допустимая адаптация.',
         'source_spans':[{'sentence_index':request['center_group_index'],'quote':request['source']['sentences'][next(i for i,s in enumerate(request['source']['sentences']) if s['index']==request['center_group_index'])]['text']}],
         'target_spans':[]} ]})
    result=analyze(report(),backend)
    assert result['status']=='complete'
    assert result['metrics']['weighted_error_points']==0
    assert result['metrics']['observations_by_kind']=={'acceptable_variation':3}


def test_stage_processes_each_pair_and_uses_mode_suffix(tmp_path,monkeypatch):
    from src.processing.mqm import stages
    from src.processing.pipeline.config import SOURCE,TARGETS,folder
    settings=tmp_path/'data/candidate_settings.json';settings.parent.mkdir()
    settings.write_text(json.dumps({'window_strategy':'adaptive'}))
    for target in TARGETS:
        path=folder(tmp_path,'review_candidates')/f'{SOURCE}_{target}.candidates.adaptive.json'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    judge=Mock();run=Mock(return_value={'status':'complete'})
    monkeypatch.setattr(stages,'configured_judge',Mock(return_value=judge))
    monkeypatch.setattr(stages,'analyze_file',run)
    stages.stage_mqm(tmp_path,[SOURCE,*TARGETS],3)
    assert run.call_count==2
    assert all(call.args[1].name.endswith('.mqm.adaptive.json') and call.kwargs['backend'] is judge for call in run.call_args_list)


def test_cached_hallucinated_citations_are_recomputed():
    data=report();previous=analyze(data,TestJudge())
    previous['fragments'][0]['findings'][0]['source_spans'][0]['quote']='tampered'
    backend=TestJudge();result=analyze(data,backend,previous=previous)
    assert backend.judge.call_count==1
    assert result['findings'][0]['source_spans'][0]['quote']=='Саркома лёгкого'


def test_cached_fragment_centers_are_rebuilt_for_coverage():
    data=report();previous=analyze(data,TestJudge())
    previous['fragments'][0]['source_sentence_indices']=[0,1,2,3]
    backend=TestJudge();result=analyze(data,backend,previous=previous)
    backend.judge.assert_not_called()
    assert result['fragments'][0]['source_sentence_indices']==[0]
    assert not result['metrics']['coverage']['entire_text_fully_analyzed']


def test_requests_explicitly_separate_center_and_pairs():
    from src.processing.mqm.prompt import messages
    request=make_request(report(),1)
    wire=json.loads(messages(request)[1]['content'])
    assert [g['group_index'] for g in wire['paired_groups'] if g['is_center']]==[1]
    assert wire['paired_groups'][0]['source_sentences'][0]['index']==0
    assert 'source' not in wire # text is sent once inside aligned groups
    assert 'tokens' not in wire
    assert 'tokens' in json.loads(messages(request,include_word_links=True)[1]['content'])


def test_corrective_retry_discards_neighbor_error_without_weakening_validation():
    backend=TestJudge();backend.max_retries=1
    backend.judge=Mock(side_effect=[{'findings':[finding(1,'Десятого','Ninth')]},{'findings':[finding()]},
                                   {'findings':[]},{'findings':[]}])
    result=analyze(report(),backend)
    assert result['status']=='complete'
    assert result['metrics']['unique_error_proposals']==1
    assert len(result['fragments'][0]['attempts'])==2
    correction=backend.judge.call_args_list[1].args[0]['correction']
    assert 'neighboring context' in correction['rejected'][0]['reason']


def test_local_bad_json_does_not_stop_remaining_groups():
    backend=TestJudge()
    backend.judge=Mock(side_effect=[ValueError('bad JSON'),{'findings':[]},{'findings':[]}])
    result=analyze(report(),backend)
    assert result['status']=='partial'
    assert [f['status'] for f in result['fragments']]==['failed','completed','completed']


def test_retry_count_is_bounded_and_rejected_not_counted():
    backend=TestJudge();backend.max_retries=1
    backend.judge=Mock(return_value={'findings':[finding(source='invented')]})
    result=analyze(report(),backend)
    assert backend.judge.call_count==6
    assert result['metrics']['unique_error_proposals']==0
    assert all(f['status']=='partial_invalid' for f in result['fragments'])


def test_http_details_visible_and_context_error_not_fatal(local_api):
    from src.processing.mqm.errors import MQMRequestError
    url,state=local_api;state.update(status=400,error='Requested context exceeds token limit')
    backend=LocalMQMBackend(model='local-test',base_url=url,json_mode=False)
    with pytest.raises(MQMRequestError,match='context exceeds') as error:
        backend.judge(make_request(report(),0))
    assert not error.value.fatal
    result=analyze(report(),backend)
    assert [f['status'] for f in result['fragments']]==['failed']*3


def test_http_detail_redacts_api_key(local_api,monkeypatch):
    url,state=local_api;state.update(status=401,error='invalid credential SECRET')
    monkeypatch.setenv('MQM_API_KEY','SECRET')
    with pytest.raises(RuntimeError) as error:
        LocalMQMBackend(model='local-test',base_url=url).judge(make_request(report(),0))
    assert 'SECRET' not in str(error.value) and '[REDACTED]' in str(error.value)


def test_unsupported_json_mode_falls_back_once_to_plain_json_content(local_api):
    url,state=local_api;state['reject_json_mode']=True
    result=LocalMQMBackend(model='local-test',base_url=url).judge(make_request(report(),0))
    assert result['findings']==[]
    assert len(state['requests'])==2
    assert 'response_format' in state['requests'][0]
    assert 'response_format' not in state['requests'][1]


def test_unique_exact_quote_repairs_bad_model_offsets():
    data=report();request=make_request(data,0)
    f=finding(source_spans=[{'sentence_index':0,'quote':'Саркома','start':1,'end':8}])
    result=validate_judgment({'findings':[f]},data,request)
    assert not result['rejected']
    span=result['findings'][0]['source_spans'][0]
    assert span['start']==0 and span['end']==7
    assert span['offset_resolution']=='unique_exact_quote'


def test_positional_citation_pair_is_normalized_but_not_guessed():
    data=report();request=make_request(data,0)
    f=finding(source_spans=[[0,'Саркома лёгкого']])
    result=validate_judgment({'findings':[f]},data,request)
    assert not result['rejected'] and result['findings'][0]['source_spans'][0]['quote']=='Саркома лёгкого'
    f['source_spans']=[['invented']]
    result=validate_judgment({'findings':[f]},data,request)
    assert 'Citation must be an object' in result['rejected'][0]['reason']


def test_hint_kind_becomes_uncertain_observation_not_error():
    data=report();request=make_request(data,0)
    note={'kind':'negation_mismatch','explanation':'Требуется проверка.',
          'source_spans':[{'sentence_index':0,'quote':'Саркома'}],'target_spans':[]}
    result=validate_judgment({'findings':[],'observations':[note]},data,request)
    assert not result['rejected']
    assert result['observations'][0]['kind']=='uncertain'
    assert result['observations'][0]['model_kind']=='negation_mismatch'
    assert not result['findings']


def test_lean_prompt_is_smaller_and_preserves_aligned_texts():
    from src.processing.mqm.prompt import messages
    req=make_request(report(),0)
    lean=messages(req)[1]['content'];detailed=messages(req,include_word_links=True)[1]['content']
    assert len(lean)<len(detailed)
    assert json.loads(lean)['paired_groups']==json.loads(detailed)['paired_groups']


def test_fragment_limit_can_be_expanded_without_repeating_completed_calls():
    data=report();first=analyze(data,TestJudge(),max_fragments=1)
    assert len(first['fragments'])==1
    assert not first['metrics']['coverage']['entire_text_fully_analyzed']
    backend=TestJudge();second=analyze(data,backend,previous=first,max_fragments=0)
    assert backend.judge.call_count==2
    assert second['fragments'][0]['reused']
    assert len(second['fragments'])==3
