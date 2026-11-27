"""Bounded union of strongly overlapping review windows before inference."""
from copy import deepcopy
from src.processing.analysis.fragments import fragment_id

DEFAULT_POLICY = {'merge_windows': True, 'merge_overlap': .75, 'merge_max_chars': 12000}


def merge_units(data, units, policy):
    if not policy['merge_windows']:
        return units
    result=[]
    def indices(groups, side):
        return {i for g in groups for i in data['semantic_tasks'][g]['window']['center_alignment'][side+'_indices']}
    def compatible(a,b):
        for side in ('source','target'):
            x,y=indices(a,side),indices(b,side)
            if x|y and len(x&y)/len(x|y)<policy['merge_overlap']:return False
        return True
    for original in units:
        unit=deepcopy(original)
        groups=set(unit['context_group_indices'])
        for existing in result:
            old=set(existing['context_group_indices'])
            union=old|groups
            chars=sum(len(data[side]['sentences'][i]['normalized_text'])+1
                      for side in ('source','target') for i in indices(union,side))
            if chars<=policy['merge_max_chars'] and compatible(old,groups):
                existing['context_group_indices']=sorted(union)
                existing['anchor_group_indices']=sorted(set(existing['anchor_group_indices']+unit['anchor_group_indices']))
                existing['priority']=max(existing['priority'],unit['priority'])
                break
        else:
            result.append(unit)
    for unit in result:
        groups=unit['context_group_indices']
        unit['fragment_id']=fragment_id({side:{'sentence_indices':sorted(indices(groups,side))} for side in ('source','target')})
        candidates=[c for c in data['candidates'] if c['group_index'] in groups]
        unit['candidate_ids']=[c['id'] for c in candidates]
        unit['signals']=[]
        for candidate in candidates:
            for signal in candidate['signals']:
                if signal not in unit['signals']:unit['signals'].append(deepcopy(signal))
    return result
