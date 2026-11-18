"""Review fragments with an injected judge; checkpoint progress without model imports."""
import hashlib
import json
import math
from copy import deepcopy
from .contracts import validate_candidates, make_request, validate_judgment
from .metrics import summarize, DEFAULT_WEIGHTS, fragment_metrics
from .prompt import PROMPT_VERSION


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf-8')).hexdigest()


def analyze(data, backend, *, scope='candidates', weights=None, previous=None, progress=None):
    validate_candidates(data)
    if scope not in ('candidates','all'):raise ValueError('MQM scope must be candidates or all')
    weights=dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(weights)!=set(DEFAULT_WEIGHTS) or any(type(n) not in (int,float) or not math.isfinite(n) or n<=0 for n in weights.values()):
        raise ValueError('MQM weights must contain positive finite minor/major/critical values')
    candidate_map={c['group_index']:c for c in data['candidates']}
    indices=[c['group_index'] for c in data['candidates']] if scope=='candidates' else list(range(len(data['semantic_tasks'])))
    run_id=fingerprint({'input':data,'backend':backend.metadata,'scope':scope,'weights':weights,'prompt':PROMPT_VERSION})
    cache={}
    if isinstance(previous,dict) and previous.get('run_fingerprint')==run_id and isinstance(previous.get('fragments'),list):
        cache={f['request_hash']:f for f in previous['fragments'] if isinstance(f,dict)
               and f.get('status')=='completed' and isinstance(f.get('request_hash'),str)}
    report={'schema_version':1,'source_document':data['source_document'],'target_document':data['target_document'],
            'source':deepcopy(data['source']),'target':deepcopy(data['target']),
            'window_config':deepcopy(data.get('window_config',{})),
            'model':deepcopy(backend.metadata),'scope':scope,'run_fingerprint':run_id,
            'status':'running','fragments':[],'findings':[],'metrics':{}}
    def update(status):
        report['status']=status
        report.update(summarize(data,report['fragments'],weights))
        if progress:progress(report)
    for position,index in enumerate(indices):
        request=make_request(data,index)
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
                result=validate_judgment(backend.judge(request),data,request)
                fragment.update(result,status='partial_invalid' if result['rejected'] else 'completed')
            except (ValueError,RuntimeError,TypeError,KeyError,IndexError) as error:
                fragment.update(status='failed',error=str(error))
        fragment['metrics']=fragment_metrics(data,fragment,weights)
        report['fragments'].append(fragment)
        print(f'[MQM] {position+1}/{len(indices)} group {index}: {fragment["status"]}',flush=True)
        update('running')
        if fragment['status']=='failed':
            # A server/model failure should not produce hundreds of identical failed calls.
            for pending in indices[position+1:]:
                report['fragments'].append({'group_index':pending,'status':'not_run_after_failure',
                                            'source_sentence_indices':data['semantic_tasks'][pending]['window']['center_alignment']['source_indices'],
                                            'target_sentence_indices':data['semantic_tasks'][pending]['window']['center_alignment']['target_indices'],
                                            'findings':[],'observations':[],'rejected':[]})
            break
    update('complete' if all(f['status']=='completed' for f in report['fragments']) else 'partial')
    return report
