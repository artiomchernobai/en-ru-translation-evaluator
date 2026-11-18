"""Descriptive metrics of model proposals, with explicit denominators/coverage."""
from collections import Counter
from copy import deepcopy
import re

DEFAULT_WEIGHTS={'minor':1,'major':5,'critical':25}
WORD_PATTERN=r"[^\W_]+(?:['’\-][^\W_]+)*"


def error_key(finding):
    return (finding['category'], *[tuple(sorted((s['sentence_index'],s['start'],s['end']) for s in finding[side+'_spans'])) for side in ('source','target')])


def summarize(data, fragments, weights):
    unique={}
    rank={'minor':0,'major':1,'critical':2}
    for fragment in fragments:
        for finding in fragment.get('findings',[]):
            key=error_key(finding)
            if key not in unique:
                unique[key]={**deepcopy(finding),'group_indices':[]}
            record=unique[key]
            if fragment['group_index'] not in record['group_indices']:record['group_indices'].append(fragment['group_index'])
            if rank[finding['severity']]>rank[record['severity']]:
                record.update(severity=finding['severity'],explanation=finding['explanation'])
    notes={}
    for fragment in fragments:
        for observation in fragment.get('observations',[]):
            key=(observation['kind'], *[tuple(sorted((s['sentence_index'],s['start'],s['end']) for s in observation[side+'_spans'])) for side in ('source','target')])
            if key not in notes:notes[key]={**deepcopy(observation),'group_indices':[]}
            if fragment['group_index'] not in notes[key]['group_indices']:notes[key]['group_indices'].append(fragment['group_index'])
    observations=list(notes.values())
    errors=list(unique.values())
    for i,error in enumerate(errors):error['id']=f'error-{i}'
    processed=[f for f in fragments if f['status'] in ('completed','partial_invalid')]
    complete=[f for f in fragments if f['status']=='completed']
    indices={i for f in processed for i in f['source_sentence_indices']}
    word_count=lambda text:len(re.findall(WORD_PATTERN,text,re.UNICODE))
    total_words=word_count(data['source']['text'])
    checked_words=sum(word_count(data['source']['sentences'][i]['normalized_text']) for i in indices)
    points=sum(weights[e['severity']] for e in errors)
    def rate(n, denominator):return n*1000/denominator if denominator else None
    statuses=dict(Counter(f['status'] for f in fragments))
    return {'findings':errors,'observations':observations,'metrics':{
        'assessment':'automated_proposals_require_human_review',
        'normalization':{'unit':'1000_source_words','word_pattern':WORD_PATTERN,'severity_weights':weights},
        'coverage':{'total_groups':len(data['semantic_tasks']),'requested_groups':len(fragments),
                    'analyzed_groups':len(processed),'fully_completed_groups':len(complete),'statuses':statuses,
                    'group_fraction':len(processed)/len(data['semantic_tasks']) if data['semantic_tasks'] else None,
                    'source_word_fraction':checked_words/total_words if total_words else None,
                    'entire_text_fully_analyzed':len(complete)==len(data['semantic_tasks'])},
        'source_words':total_words,'reviewed_center_source_words':checked_words,
        'unique_error_proposals':len(errors),'by_category':dict(Counter(e['category'] for e in errors)),
        'by_severity':dict(Counter(e['severity'] for e in errors)),
        'unique_observations':len(observations),'observations_by_kind':dict(Counter(o['kind'] for o in observations)),
        'acceptable_variations_per_1000_document_source_words':rate(sum(o['kind']=='acceptable_variation' for o in observations),total_words),
        'rejected_proposal_count':sum(len(f.get('rejected',[])) for f in fragments),
        'weighted_error_points':points,
        'observed_errors_per_1000_document_source_words':rate(len(errors),total_words),
        'observed_weighted_points_per_1000_document_source_words':rate(points,total_words),
        'errors_per_1000_reviewed_center_source_words':rate(len(errors),checked_words),
        'mean_error_proposals_per_analyzed_group':len(errors)/len(processed) if processed else None,
        'document_rates_are_exhaustive':len(complete)==len(data['semantic_tasks']),
        'extrapolated_unreviewed_errors':None}}


def fragment_metrics(data, fragment, weights):
    unique={}
    for finding in fragment['findings']:
        key=error_key(finding)
        if key not in unique or weights[finding['severity']]>weights[unique[key]['severity']]:unique[key]=finding
    count=sum(len(re.findall(WORD_PATTERN,data['source']['sentences'][i]['normalized_text'])) for i in fragment['source_sentence_indices'])
    return {'analyzed':fragment['status'] in ('completed','partial_invalid'),
            'central_source_words':count,'unique_error_proposals':len(unique),
            'weighted_error_points':sum(weights[f['severity']] for f in unique.values()),
            'by_category':dict(Counter(f['category'] for f in unique.values()))}
