from copy import deepcopy
import json
import numpy as np
import pytest
from src.processing.alignment.link_evidence import filter_links
from src.processing.alignment.words import WordAligner
from src.ui.review_model import review_payload, validate_review
from src.processing.analysis.rules import lexical_signals


def test_secondary_one_to_many_is_kept_but_low_scores_and_nonproposals_are_not():
    scores=np.array([[.9,.84,.1],[.89,.83,.3]])
    raw={'itermax':[(np.int64(0),np.int64(0)),(0,1),(1,2)]}
    accepted,evidence=filter_links(raw,scores,.55,.02,.12)
    assert accepted['itermax']==[(0,0),(0,1)]
    assert evidence['itermax'][1]['role']=='secondary'
    assert json.loads(json.dumps(evidence))['itermax'][0]['source_index']==0
    assert evidence['itermax'][2]['reasons']==['low_similarity']
    assert (1,0) not in accepted['itermax']  # Dense similarity is not a model proposal.
    strict,_=filter_links(raw,scores,.55,.02,.01)
    assert strict['itermax']==[(0,0)]
    reverse,_=filter_links({'itermax':[(j,i) for i,j in raw['itermax']]},scores.T,.55,.02,.12)
    assert set(reverse['itermax'])=={(j,i) for i,j in accepted['itermax']}


class EvidenceBackend:
    metadata={'name':'test'}
    def align_tokens(self,source,target):
        self.last_scores=np.full((len(source),len(target)),.1)
        self.last_scores[0,0]=.9
        if min(len(source),len(target))>1: self.last_scores[1,1]=.5
        self.last_token_map={side:[
            [{'index':int(j),'similarity':float(row[j])} for j in np.argsort(-row)[:3]] for row in matrix]
            for side,matrix in [('source',self.last_scores),('target',self.last_scores.T)]}
        raw={m:([(0,0),(1,1)] if min(len(source),len(target))>1 else [(0,0)]) for m in ('inter','mwmf','itermax')}
        accepted,self.last_link_evidence=filter_links(raw,self.last_scores,.55,.02,.12)
        return accepted


def result(neighbor=False):
    doc=lambda name,lang,text:{'document':name,'language':lang,'sentences':[{'sentence_id':0,'normalized_text':text}]}
    source=doc('s','ru','слово второе третье четвертое')
    target=doc('t','en','word second third fourth')
    if neighbor:
        source['sentences'].append({'sentence_id':1,'normalized_text':'пятое'})
        target['sentences'].append({'sentence_id':1,'normalized_text':'fifth'})
    groups=[{'source_indices':[i],'target_indices':[i],'cost':.2} for i in range(len(source['sentences']))]
    return WordAligner(EvidenceBackend()).align(source,target,groups)


def test_model_proposals_filter_rejections_and_weak_ui_are_separate():
    data=result(); validate_review(data)
    tokens=data['blocks'][0]['word_alignment']['token_map']['source']
    assert tokens[1]['status']=='filter_rejected'
    assert tokens[2]['status']=='model_unlinked'
    payload=review_payload(data)
    assert payload['panes']['source']['tokens'][1]['partners']==[]
    assert payload['panes']['source']['tokens'][1]['weak_partners']==[{'id':1,'similarity':.5}]
    assert not review_payload(data,show_weak=False)['panes']['source']['tokens'][1]['weak_partners']
    assert not review_payload(data,weak_minimum=.6)['panes']['source']['tokens'][1]['weak_partners']
    assert not payload['panes']['source']['tokens'][2]['weak_partners']


def test_uncertainty_reduces_candidate_weight_without_hiding_other_missing_content():
    data=result(neighbor=True); signals=lexical_signals(data,data['blocks'][0],[])
    uncertainty=next(s for s in signals if s['code']=='word_alignment_uncertain')
    missing=next(s for s in signals if s['code']=='unlinked_content')
    assert uncertainty['weight']==0  # No double counting correlated word signals.
    assert missing['weight']==10
    assert 'второе' in [s['quote'] for s in uncertainty['source_spans']]
    assert 'третье' in [s['quote'] for s in missing['source_spans']]
    assert 'четвертое' in [s['quote'] for s in uncertainty['source_spans']]
    legacy=deepcopy(data)
    del legacy['blocks'][0]['word_alignment']['link_evidence']
    assert not any(s['code']=='word_alignment_uncertain' for s in lexical_signals(legacy,legacy['blocks'][0],[]))


@pytest.mark.parametrize('field', ['proposal','alternative','accepted'])
def test_invalid_evidence_and_alternatives_are_rejected(field):
    data=result(); word=data['blocks'][0]['word_alignment']
    if field=='proposal': word['link_evidence']['itermax'][0]['target_token_id']=999
    elif field=='alternative': word['token_map']['source'][0]['alternatives'][0]['token_id']=999
    else: word['link_evidence']['itermax'][1]['accepted']=True
    with pytest.raises(ValueError): validate_review(data)
