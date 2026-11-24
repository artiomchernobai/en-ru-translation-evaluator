"""Review fragments with an injected judge; checkpoint progress without model imports."""
import hashlib
import json
import math
import time
from copy import deepcopy
from .contracts import validate_candidates, make_request, validate_judgment
from .metrics import summarize, DEFAULT_WEIGHTS, fragment_metrics
from .prompt import PROMPT_VERSION
from .judging import judge_fragment
from .errors import MQMRequestError
from src.processing.analysis.fragments import review_fragments


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf-8')).hexdigest()


def analyze(data, backend, *, scope='candidates', weights=None, previous=None, progress=None, max_fragments=0, min_priority=0, group_indices=None):
    validate_candidates(data)
    if scope not in ('candidates','all'):raise ValueError('MQM scope must be candidates or all')
    weights=dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(weights)!=set(DEFAULT_WEIGHTS) or any(type(n) not in (int,float) or not math.isfinite(n) or n<=0 for n in weights.values()):
        raise ValueError('MQM weights must contain positive finite minor/major/critical values')
    if type(max_fragments) is not int or max_fragments<0 or type(min_priority) is not int or not 0<=min_priority<=100:
        raise ValueError('Invalid fragment limit or minimum priority')
    candidate_map={c['group_index']:c for c in data['candidates']}
    units=review_fragments(data,scope=scope,min_priority=min_priority)
    if group_indices is not None:
        if not isinstance(group_indices,list) or any(type(i) is not int or i<0 for i in group_indices):raise ValueError('Invalid selected groups')
        units=[u for u in units if set(u['anchor_group_indices']).intersection(group_indices)]
        if not units:raise ValueError('Selected candidate is absent from this report')
    if max_fragments:units=units[:max_fragments]
    semantic_input={k:v for k,v in data.items() if k not in ('created_at','inputs','review_fragments')}
    run_id=fingerprint({'input':semantic_input,'backend':backend.metadata,'scope':scope,'weights':weights,'prompt':PROMPT_VERSION})
    cache={}
    if isinstance(previous,dict) and previous.get('run_fingerprint')==run_id and isinstance(previous.get('fragments'),list):
        cache={f['request_hash']:f for f in previous['fragments'] if isinstance(f,dict)
               and f.get('status')=='completed' and isinstance(f.get('request_hash'),str)}
    report={'schema_version':1,'source_document':data['source_document'],'target_document':data['target_document'],
            'source':deepcopy(data['source']),'target':deepcopy(data['target']),
            'window_config':deepcopy(data.get('window_config',{})),
            'model':deepcopy(backend.metadata),'scope':scope,'assessment_unit':'entire_fragment','anchor_role':'navigation_only','run_fingerprint':run_id,
            'selection':{'max_fragments':max_fragments,'min_priority':min_priority,'group_indices':group_indices},
            'status':'running','fragments':[],'findings':[],'metrics':{}}
    def update(status):
        report['status']=status
        report.update(summarize(data,report['fragments'],weights))
        if progress:progress(report)
    for position,unit in enumerate(units):
        index=unit['group_index']
        started=time.perf_counter()
        request=make_request(data,index)
        request['hints']=[{k:s[k] for k in ('code','message','source_spans','target_spans')} for s in unit['signals']]
        request_hash=fingerprint(request)
        cached=cache.get(request_hash)
        if cached:
            try:
                validated=validate_judgment(cached,data,request)
                if validated['rejected'] or any(key not in cached for key in ('group_index','source_sentence_indices','target_sentence_indices','observations')):cached=None
            except (ValueError,TypeError,KeyError):cached=None
        if cached:
            fragment=deepcopy(cached)
            fragment.update(validated,group_index=index,reused=True,
                            source_sentence_indices=request['source']['central_sentence_indices'],
                            target_sentence_indices=request['target']['central_sentence_indices'],
                            context_group_indices=[g['group_index'] for g in request['alignments']],
                            context={'source':request['source']['sentences'],'target':request['target']['sentences']})
        else:
            central=request
            fragment={'group_index':index,'candidate_id':candidate_map.get(index,{}).get('id'),
                      'priority':candidate_map.get(index,{}).get('priority'), 'request_hash':request_hash,'reused':False,
                      'source_sentence_indices':central['source']['central_sentence_indices'],
                      'target_sentence_indices':central['target']['central_sentence_indices'],
                      'context_group_indices':[g['group_index'] for g in request['alignments']],
                      'comet_window':deepcopy(data['semantic_tasks'][index]['window']['verdict']),
                      'context':{'source':request['source']['sentences'],'target':request['target']['sentences']},
                      'findings':[],'observations':[],'rejected':[]}
            try:
                print(f'[MQM] Анализ группы {index} локальной моделью {backend.metadata["name"]}',flush=True)
                fragment.update(judge_fragment(data,request,backend))
            except (ValueError,RuntimeError,TypeError,KeyError,IndexError) as error:
                fragment.update(status='failed',error=str(error),fatal=isinstance(error,RuntimeError) and getattr(error,'fatal',True))
        fragment['assessed_source_sentence_indices']=[s['index'] for s in request['source']['sentences']]
        fragment['assessed_target_sentence_indices']=[s['index'] for s in request['target']['sentences']]
        fragment.update(fragment_id=unit['fragment_id'],anchor_group_indices=unit['anchor_group_indices'],
                        candidate_ids=unit['candidate_ids'],priority=unit['priority'],signals=deepcopy(request['hints']))
        fragment['elapsed_seconds']=round(time.perf_counter()-started,3)
        fragment['metrics']=fragment_metrics(data,fragment,weights)
        report['fragments'].append(fragment)
        print(f'[MQM] {position+1}/{len(units)} fragment {unit["fragment_id"]} anchors {unit["anchor_group_indices"]}: {fragment["status"]}',flush=True)
        update('running')
        if fragment.get('error'):print(f'[MQM] group {index}: {fragment["error"]}',flush=True)
        if fragment['status']=='failed' and fragment.get('fatal',False):
            # A server/model failure should not produce hundreds of identical failed calls.
            for pending_unit in units[position+1:]:
                pending=pending_unit['group_index']
                pending_request=make_request(data,pending)
                report['fragments'].append({'group_index':pending,'status':'not_run_after_failure',
                                            'fragment_id':pending_unit['fragment_id'],
                                            'anchor_group_indices':pending_unit['anchor_group_indices'],
                                            'candidate_ids':pending_unit['candidate_ids'],
                                            'signals':pending_unit['signals'],
                                            'context_group_indices':pending_unit['context_group_indices'],
                                            'context':{side:pending_request[side]['sentences'] for side in ('source','target')},
                                            'source_sentence_indices':data['semantic_tasks'][pending]['window']['center_alignment']['source_indices'],
                                            'target_sentence_indices':data['semantic_tasks'][pending]['window']['center_alignment']['target_indices'],
                                            'findings':[],'observations':[],'rejected':[]})
            break
    update('complete' if all(f['status']=='completed' for f in report['fragments']) else 'partial')
    return report
