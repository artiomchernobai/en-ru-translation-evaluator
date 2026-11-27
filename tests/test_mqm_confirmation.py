"""Target-wide counterpart search, critical gates, bounded unions and checkpoints."""
from copy import deepcopy
import json
from unittest.mock import Mock
import pytest
from test_mqm import report, finding, local_api
from src.processing.mqm.backend import LocalMQMBackend
from src.processing.mqm.contracts import make_request, validate_judgment
from src.processing.mqm.judging import judge_fragment
from src.processing.mqm.engine import analyze
from src.processing.mqm.prompt import verification_messages
from src.processing.mqm.reports import compact_report, restore_report, validate_fragment, diagnostics
from src.processing.mqm.windows import merge_units, DEFAULT_POLICY
from src.processing.analysis.fragments import review_fragments


def judge(raw, decision):
    class Judge:
        metadata={'name':'two-pass-test'}
        max_retries=0
    backend=Judge()
    backend.judge=Mock(return_value={'findings':[raw]})
    backend.verify=Mock(return_value={'decisions':[decision]})
    return backend


def test_omission_is_disproved_by_paraphrase_in_reordered_neighbor():
    data=report(['Он обиделся.','Он ушёл.'],['He left.','He took offence.'])
    request=make_request(data,0)
    raw=finding(category='Accuracy/Omission',severity='minor',source='обиделся',target_spans=[])
    backend=judge(raw,{'id':'P0','verdict':'dismissed','explanation':'Смысл выражен в соседнем предложении через took offence.',
                       'target_spans':[{'ref':'T1','quote':'took offence'}]})
    result=judge_fragment(data,request,backend)
    assert not result['findings']
    assert result['observations'][0]['target_spans'][0]['sentence_index']==1
    assert result['confirmation']['dismissed']==1
    wire=json.loads(verification_messages(request,backend.verify.call_args.args[1])[1]['content'])
    assert [s['ref'] for s in wire['target']['sentences']]==['T0','T1']
    assert set(wire)=={'source','target','proposals'}  # No candidate priming or large token tables.


@pytest.mark.parametrize('refs,found,expected', [(['T0'],False,False),(['T0','T1'],True,False),(['T0','T1'],False,True)])
def test_confirming_omission_requires_search_of_whole_target(refs,found,expected):
    data=report();request=make_request(data,0)
    raw=finding(category='Accuracy/Omission',severity='minor',target_spans=[])
    decision={'id':'P0','verdict':'confirmed','explanation':'Значение отсутствует во всём переданном переводе.',
              'target_search':{'searched_refs':refs,'counterpart_found':found,'explanation':'Проверены все предложения, смыслового соответствия не найдено.'}}
    result=judge_fragment(data,request,judge(raw,decision))
    assert bool(result['findings']) is expected
    if not expected:assert result['observations'][0]['semantic_validation']=='unconfirmed'


def test_major_is_rechecked_and_minor_does_not_spend_another_call():
    data=report();request=make_request(data,0)
    backend=judge(finding(),{'id':'P0','verdict':'confirmed','explanation':'Саркома заменена другим заболеванием.'})
    result=judge_fragment(data,request,backend)
    assert result['findings'][0]['semantic_validation']=='model_rechecked'
    backend.verify.assert_called_once()
    backend=judge(finding(severity='minor'),{})
    assert judge_fragment(data,request,backend)['findings']
    backend.verify.assert_not_called()


def test_failed_or_duplicate_confirmation_cannot_count_as_error():
    data=report();request=make_request(data,0)
    backend=judge(finding(),{})
    backend.verify.side_effect=RuntimeError('server unavailable')
    result=judge_fragment(data,request,backend)
    assert result['status']=='partial_invalid' and not result['findings']
    assert result['confirmation']['uncertain']==1
    backend.verify.side_effect=None
    backend.verify.return_value={'decisions':[{'id':'P0','verdict':'confirmed','explanation':'yes'}]*2}
    assert not judge_fragment(data,request,backend)['findings']


def test_no_verifier_or_invalid_severity_cannot_bypass_confirmation():
    data=report();request=make_request(data,0)
    backend=judge(finding(),{'id':'P0','verdict':'confirmed','explanation':'Проверено.', 'severity':['major']})
    assert not judge_fragment(data,request,backend)['findings']
    del backend.verify
    result=judge_fragment(data,request,backend)
    assert result['status']=='partial_invalid' and not result['findings']


def test_critical_requires_consequence_in_both_passes():
    data=report();req=make_request(data,0)
    initial=validate_judgment({'findings':[finding(severity='critical')]},data,req)
    assert initial['findings'][0]['severity']=='major'
    raw=finding(severity='critical',critical_criterion='material_fact_reversal',
                critical_consequence='Изменён ключевой факт, от которого зависит понимание дальнейшего события.')
    backend=judge(raw,{'id':'P0','verdict':'confirmed','explanation':'Смысл изменён.'})
    assert judge_fragment(data,req,backend)['findings'][0]['severity']=='major'
    backend.verify.return_value={'decisions':[{'id':'P0','verdict':'confirmed','explanation':'Подтверждено изменение ключевого факта.',
                                'critical_criterion':raw['critical_criterion'],'critical_consequence':raw['critical_consequence']}]}
    assert judge_fragment(data,req,backend)['findings'][0]['critical_gate']=='model_rechecked'


def test_overlap_union_preserves_all_signals_and_citations_with_budget_and_selection():
    data=report(['не первый','не второй','не третий','не четвертый'],['first','second','third','fourth'])
    backend=judge(finding(1,'второй','second'),{'id':'P0','verdict':'confirmed','explanation':'Проверка выполнена.'})
    backend.review_policy={**DEFAULT_POLICY,'merge_overlap':.5}
    result=analyze(data,backend,max_fragments=1)
    fragment=result['fragments'][0]
    request=backend.judge.call_args.args[0]
    assert fragment['context_group_indices']==[0,1,2,3]
    assert {s['index'] for s in request['target']['sentences']}=={0,1,2,3}
    assert len(fragment['candidate_ids'])==4 and len(fragment['anchor_group_indices'])==4
    portable=compact_report(result);validate_fragment(portable['fragments'][0])
    restored=restore_report(portable,data)
    backend.judge.reset_mock();backend.verify.reset_mock()
    resumed=analyze(data,backend,max_fragments=1,previous=restored)
    backend.judge.assert_not_called();backend.verify.assert_not_called()
    assert resumed['findings'][0]['semantic_validation']=='model_rechecked'
    assert diagnostics(result)['fragments'][0]['initial_proposals']
    selected=analyze(data,backend,group_indices=[0])
    assert selected['fragments'][0]['context_group_indices']==[0,1]
    original=review_fragments(data)
    long=deepcopy(data)
    for side in ('source','target'):
        for sentence in long[side]['sentences']:sentence['normalized_text']+='x'*1000
    assert len(merge_units(long,original,{**DEFAULT_POLICY,'merge_overlap':.5,'merge_max_chars':1000}))==len(original)


def test_confirmation_uses_real_http_json_fallback_without_reverting_to_first_pass(local_api):
    url,state=local_api
    state.update(reject_json_mode=True,content={'decisions':[]})
    backend=LocalMQMBackend(model='local',base_url=url)
    req=make_request(report(),0)
    assert backend.verify(req,[])=={'decisions':[]}
    assert len(state['requests'])==2
    for sent in state['requests']:
        assert 'second-pass verifier' in sent['messages'][0]['content']
        assert 'proposals' in json.loads(sent['messages'][1]['content'])


def test_resuming_preserves_confirmation_audit_in_sidecar(tmp_path):
    from src.processing.mqm.io import analyze_file
    data=report()
    source=tmp_path/'candidates.json';source.write_text(json.dumps(data))
    output=tmp_path/'report.json'
    backend=judge(finding(),{'id':'P0','verdict':'confirmed','explanation':'Заболевание заменено другим.'})
    analyze_file(source,output,backend=backend,max_fragments=1)
    sidecar=output.with_suffix('.diagnostics.json')
    before=json.loads(sidecar.read_text())['fragments'][0]
    backend.judge.reset_mock();backend.verify.reset_mock()
    analyze_file(source,output,backend=backend,max_fragments=1)
    backend.verify.assert_not_called();backend.judge.assert_not_called()
    after=json.loads(sidecar.read_text())['fragments'][0]
    assert after['initial_proposals']==before['initial_proposals']
    assert after['confirmation_response']==before['confirmation_response']
