"""A separate, fail-closed semantic check of high-risk model proposals."""
from copy import deepcopy


def confirm(data, request, result, backend):
    risky=[f for f in result['findings'] if f['category']=='Accuracy/Omission' or f['severity'] in ('major','critical')]
    if not risky:return result
    result=deepcopy(result)
    result['initial_proposals']=deepcopy(result['findings'])
    result['findings']=[f for f in result['findings'] if f not in risky]
    audit={'requested':len(risky),'confirmed':0,'dismissed':0,'uncertain':0}
    result['confirmation']=audit
    try:
        if not callable(getattr(backend,'verify',None)):
            raise RuntimeError('MQM backend does not implement separate confirmation')
        response=backend.verify(request,risky)
        decisions=response['decisions']
        if not isinstance(decisions,list):raise ValueError('Confirmation decisions must be an array')
    except (ValueError,RuntimeError,TypeError,KeyError,IndexError) as error:
        decisions=[]
        result['rejected'].append({'section':'confirmation','reason':str(error),'proposal':None})
        result['confirmation_error']=str(error)
    result['confirmation_usage']=deepcopy(getattr(backend,'last_usage',None))
    result['confirmation_response']=deepcopy(decisions)
    from .contracts import _spans
    for i,finding in enumerate(risky):
        matches=[d for d in decisions if isinstance(d,dict) and d.get('id')==f'P{i}']
        decision=matches[0] if len(matches)==1 else {}
        verdict=decision.get('verdict','uncertain')
        explanation=decision.get('explanation','')
        valid=isinstance(explanation,str) and bool(explanation.strip())
        ranks={'minor':0,'major':1,'critical':2}
        severity=decision.get('severity',finding['severity'])
        valid=valid and isinstance(severity,str) and severity in ranks
        if valid:valid=ranks[severity]<=ranks[finding['severity']]
        if finding['category']=='Accuracy/Omission' and verdict=='confirmed':
            search=decision.get('target_search',{})
            expected={f'T{s["index"]}' for s in request['target']['sentences']}
            valid=valid and isinstance(search,dict) and isinstance(search.get('searched_refs'),list)
            if valid:
                valid=(bool(expected) and all(isinstance(r,str) for r in search['searched_refs']) and set(search['searched_refs'])==expected
                       and search.get('counterpart_found') is False
                       and isinstance(search.get('explanation'),str) and len(search['explanation'].strip())>=10)
        if verdict=='confirmed' and valid:
            item=deepcopy(finding)
            item['severity']=severity
            if item['severity']=='critical':
                if decision.get('critical_criterion')!=item.get('critical_criterion') or not isinstance(decision.get('critical_consequence'),str) or len(decision['critical_consequence'].strip())<20:
                    item.update(severity='major',critical_gate='downgraded_on_confirmation')
                else:item['critical_gate']='model_rechecked'
            elif item.get('critical_gate')=='requires_confirmation':
                item['critical_gate']='downgraded_on_confirmation'
            item.update(explanation=explanation,semantic_validation='model_rechecked',confirmation_status='confirmed')
            result['findings'].append(item);audit['confirmed']+=1
        elif verdict=='dismissed' and valid:
            # A purported counterpart must be an exact citation if one is supplied.
            try:
                resolved={side:_spans(decision.get(side+'_spans',[]),side,data,request) for side in ('source','target')}
                if finding['category']=='Accuracy/Omission' and not resolved['target']:
                    raise ValueError('Dismissing an omission requires a target counterpart')
                audit['dismissed']+=1
                result['observations'].append({'kind':'acceptable_variation','explanation':explanation,
                    **{side+'_spans':resolved[side] or finding[side+'_spans'] for side in ('source','target')},
                    'status':'needs_review','citation_validation':'passed','semantic_validation':'model_rechecked',
                    'confirmation_status':'dismissed'})
            except (ValueError,TypeError,KeyError,IndexError):
                verdict='uncertain'
        else:
            verdict='uncertain'
            result['rejected'].append({'section':'confirmation','reason':'Missing, ambiguous or incomplete confirmation decision', 'proposal':decision})
        if verdict=='uncertain':
            audit['uncertain']+=1
            result['observations'].append({'kind':'uncertain','explanation':'Находка не подтверждена повторной проверкой: '+finding['explanation'],
                'source_spans':finding['source_spans'],'target_spans':finding['target_spans'],
                'status':'needs_review','citation_validation':'passed','semantic_validation':'unconfirmed',
                'confirmation_status':'uncertain'})
    return result
