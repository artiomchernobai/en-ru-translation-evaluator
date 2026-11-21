"""Bounded corrective retries; every response still passes strict citation checks."""
from copy import deepcopy
from .contracts import validate_judgment


def judge_fragment(data, request, backend):
    attempts=[]
    best=None
    outgoing=deepcopy(request)
    for number in range(getattr(backend,'max_retries',0)+1):
        try:
            result=validate_judgment(backend.judge(outgoing),data,request)
            attempts.append({'attempt':number+1,'status':'partial_invalid' if result['rejected'] else 'completed',
                             'rejected':result['rejected'][:10], 'usage':deepcopy(getattr(backend,'last_usage',None))})
            if best is None or len(result['rejected'])<len(best['rejected']):best=result
            if not result['rejected']:break
            reasons=[r['reason'] for r in result['rejected']]
            print(f'[MQM] group {request["center_group_index"]}: rejected: {reasons}',flush=True)
            outgoing['correction']={'instruction':'Return a COMPLETE replacement JSON for the CENTER only. Fix rejected proposals or omit unsupported claims. Do not invent new citations.',
                                    'rejected':result['rejected'][:10], 'usage':deepcopy(getattr(backend,'last_usage',None))}
        except (ValueError,TypeError,KeyError,IndexError) as error:
            attempts.append({'attempt':number+1,'status':'invalid_response','error':str(error)})
            print(f'[MQM] group {request["center_group_index"]}: invalid response: {error}',flush=True)
            outgoing['correction']={'instruction':'Return only a valid JSON object with findings/observations arrays; exact central citations required.', 'error':str(error)}
    if best is not None:
        return {**best,'attempts':attempts,'status':'partial_invalid' if best['rejected'] else 'completed'}
    return {'status':'failed','error':attempts[-1]['error'],'attempts':attempts,
            'findings':[],'observations':[],'rejected':[]}
