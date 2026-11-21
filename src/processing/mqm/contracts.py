"""Validate input tasks and model citations; never guess repeated quote positions."""
import re
from src.processing.analysis.locations import locate
from src.processing.alignment.sentences import validate_alignment
from src.processing.evaluation.context_validation import validate_context

CATEGORIES = ('Accuracy/Mistranslation', 'Accuracy/Omission', 'Accuracy/Addition',
              'Terminology', 'Fluency/Grammar', 'Fluency/Spelling', 'Style/Register')
SEVERITIES = ('minor', 'major', 'critical')
OBSERVATIONS = ('acceptable_variation', 'uncertain', 'alignment_issue')


def validate_candidates(data):
    if data.get('schema_version') != 1 or not isinstance(data.get('semantic_tasks'), list):
        raise ValueError('Expected candidate report schema 1')
    for side in ('source', 'target'):
        doc = data[side]
        if doc['document_id'] != data[side+'_document'] or doc['text'] != '\n'.join(s['normalized_text'] for s in doc['sentences']):
            raise ValueError('Candidate document identity/text differs')
        offset = 0
        for i, sentence in enumerate(doc['sentences']):
            if (sentence['index'] != i or sentence['document_char_start'] != offset
                    or sentence['document_char_end'] != offset+len(sentence['normalized_text'])):
                raise ValueError('Invalid candidate sentence coordinates')
            offset = sentence['document_char_end']+1
        previous_end=0
        for i, token in enumerate(doc['tokens']):
            start,end=token['document_char_start'],token['document_char_end']
            if (type(start) is not int or type(end) is not int or not previous_end <= start < end <= len(doc['text'])
                    or token['token_id'] != i or doc['text'][start:end] != token['text']):
                raise ValueError('Invalid candidate token coordinates')
            previous_end=end
    tasks = data['semantic_tasks']
    groups = [{key: task['window']['center_alignment'][key] for key in ('source_indices','target_indices','cost')} for task in tasks]
    validate_alignment(groups, len(data['source']['sentences']), len(data['target']['sentences']))
    config = data.get('window_config', {'strategy':'fixed', 'radius':1})
    for i, task in enumerate(tasks):
        window = task['window']
        if task['center_group_index'] != i or window['center_group_index'] != i:
            raise ValueError('Invalid semantic task ordering')
        ids = validate_context(window,i,len(tasks),config.get('radius',1),config)
        if [g['group_index'] for g in window['alignments']] != ids:
            raise ValueError('Invalid semantic alignment membership')
        for group in window['alignments']:
            if any(group[k] != groups[group['group_index']][k] for k in groups[group['group_index']]):
                raise ValueError('Candidate alignment windows differ')
        for side in ('source','target'):
            indices = [j for k in ids for j in groups[k][side+'_indices']]
            if window[side]['sentence_indices'] != indices:
                raise ValueError('Candidate context indices differ')
            expected = [(j, data[side]['sentences'][j]['sentence_id'], data[side]['sentences'][j]['normalized_text']) for j in indices]
            if [(s['index'],s['sentence_id'],s['text']) for s in window[side]['sentences']] != expected:
                raise ValueError('Candidate context text differs')
        for block in task['word_links']:
            if block['group_index'] not in ids:
                raise ValueError('Word links outside context')
            for link in block['links']:
                for side in ('source','target'):
                    token_id = link[side+'_token_id']
                    if type(token_id) is not int or not 0 <= token_id < len(data[side]['tokens']):
                        raise ValueError('Invalid word link token ID')
                    if data[side]['tokens'][token_id]['sentence_index'] not in groups[block['group_index']][side+'_indices']:
                        raise ValueError('Word link outside group')
    seen = set()
    for candidate in data['candidates']:
        index = candidate['group_index']
        if type(index) is not int or not 0 <= index < len(tasks) or index in seen:
            raise ValueError('Invalid or duplicate candidate group')
        seen.add(index)
    return data


def make_request(data, group_index):
    task = data['semantic_tasks'][group_index]
    candidate = next((c for c in data['candidates'] if c['group_index']==group_index), None)
    window = task['window']
    result = {'center_group_index':group_index, 'source_language':data['source']['language'],
              'target_language':data['target']['language'], 'alignments':window['alignments'],
              'word_links':task['word_links'], 'hints':[{key:signal[key] for key in ('code','message','source_spans','target_spans')} for signal in candidate['signals']] if candidate else []}
    for side in ('source','target'):
        result[side] = {'central_sentence_indices':window['center_alignment'][side+'_indices'],
                        'sentences':window[side]['sentences'],
                        'tokens':[{'token_id':t['token_id'], 'sentence_index':t['sentence_index'],
                                   'text':t['text'], 'start':t['sentence_char_start'], 'end':t['sentence_char_end']}
                                  for t in data[side]['tokens'] if t['sentence_index'] in window[side]['sentence_indices']]}
    result['paired_groups'] = [
        {'group_index':g['group_index'], 'is_center':g['group_index']==group_index,
         'source_sentences':[s for s in window['source']['sentences'] if s['index'] in g['source_indices']],
         'target_sentences':[s for s in window['target']['sentences'] if s['index'] in g['target_indices']],
         'has_gap':not g['source_indices'] or not g['target_indices']}
        for g in window['alignments']]
    return result


def _spans(raw, side, data, request):
    if not isinstance(raw, list):
        raise ValueError('Citation spans must be arrays')
    allowed = {s['index'] for s in request[side]['sentences']}
    result=[]
    for span in raw:
        if isinstance(span,list) and len(span)==2 and type(span[0]) is int and isinstance(span[1],str):
            span={'sentence_index':span[0],'quote':span[1]}
        if not isinstance(span,dict):raise ValueError('Citation must be an object with sentence_index and quote')
        index, quote = span.get('sentence_index'), span.get('quote')
        if type(index) is not int or index not in allowed or not isinstance(quote,str) or not quote:
            raise ValueError('Citation missing or outside context')
        text = data[side]['sentences'][index]['normalized_text']
        start,end = span.get('start'),span.get('end')
        if start is None and end is None:
            matches = list(re.finditer(re.escape(quote),text))
            if len(matches)!=1:
                raise ValueError('Quote is absent or ambiguous; explicit offsets required')
            start,end=matches[0].span()
        try:
            location=locate(data[side],index,start,end)
            if location['quote'] != quote:raise ValueError('Quote does not match its offsets')
        except ValueError:
            matches=list(re.finditer(re.escape(quote),text))
            if len(matches)!=1:raise ValueError('Quote is absent or ambiguous; valid offsets required')
            location=locate(data[side],index,*matches[0].span())
            location['offset_resolution']='unique_exact_quote'
        result.append(location)
    return result


def validate_judgment(raw, data, request):
    if not isinstance(raw,dict) or not isinstance(raw.get('findings'),list) or not isinstance(raw.get('observations',[]),list):
        raise ValueError('Model must return findings and observations arrays')
    accepted,notes,rejected=[],[],[]
    for name, items, destination in [('findings',raw['findings'],accepted),('observations',raw.get('observations',[]),notes)]:
        for item in items:
            try:
                if not isinstance(item,dict): raise ValueError('Finding must be an object')
                original_category=item.get('category')
                aliases={'Mistranslation':'Accuracy/Mistranslation','Omission':'Accuracy/Omission','Addition':'Accuracy/Addition'}
                if name=='findings' and original_category in aliases:item={**item,'category':aliases[original_category]}
                if name=='findings' and (item.get('category') not in CATEGORIES or item.get('severity') not in SEVERITIES):
                    raise ValueError('Unknown MQM category or severity')
                original_kind=item.get('kind')
                if name=='observations' and original_kind in ('negation_mismatch','quantity_mismatch','terminology_mismatch','unlinked_content','sentence_gap','low_comet_window','high_vecalign_cost','comet_unscored','word_alignment_unavailable'):
                    item={**item,'kind':'uncertain'}
                if name=='observations' and item.get('kind') not in OBSERVATIONS:
                    raise ValueError('Unknown observation kind')
                if not isinstance(item.get('explanation'),str) or not item['explanation'].strip():
                    raise ValueError('Missing explanation')
                spans={side:_spans(item.get(side+'_spans'),side,data,request) for side in ('source','target')}
                if not any(spans.values()): raise ValueError('At least one citation is required')
                category=item.get('category')
                if name=='findings':
                    if not spans['source'] and category!='Accuracy/Addition': raise ValueError('Source evidence required')
                    if not spans['target'] and category!='Accuracy/Omission': raise ValueError('Target evidence required')
                record={key:item[key] for key in (('category','severity') if name=='findings' else ('kind',))}
                if name=='observations' and original_kind!=item['kind']:record['model_kind']=original_kind
                if name=='findings' and original_category!=item['category']:record['model_category']=original_category
                record.update(explanation=item['explanation'], source_spans=spans['source'],target_spans=spans['target'],
                              status='needs_review',citation_validation='passed',semantic_validation='model_only')
                if isinstance(item.get('suggestion'),str):record['suggestion']=item['suggestion']
                destination.append(record)
            except (ValueError,TypeError,KeyError,AttributeError,IndexError) as error:
                rejected.append({'section':name,'reason':str(error),'proposal':item})
    return {'findings':accepted,'observations':notes,'rejected':rejected}


def attach_corpus(data, corpus):
    """Use canonical token tables/links after proving identity with candidate windows."""
    from copy import deepcopy
    if corpus.get('schema_version')!=2 or corpus.get('word_alignment_status')!='included':
        raise ValueError('Пересоздайте корпус этапом corpus: требуется версия 2 со связями SimAlign')
    if any(corpus[side+'_document']!=data[side+'_document'] or
           corpus['documents'][side]!=data[side] for side in ('source','target')):
        raise ValueError('Corpus and candidate document/token tables differ')
    if len(corpus['segments'])!=len(data['semantic_tasks']):raise ValueError('Corpus and candidate groups differ')
    method=corpus['selected_word_method']
    if method!=data['selected_word_method']:raise ValueError('Corpus SimAlign method differs')
    for i,segment in enumerate(corpus['segments']):
        center=data['semantic_tasks'][i]['window']['center_alignment']
        if segment['segment_id']!=i or segment['vecalign_cost']!=center['cost'] or any(segment[side]['sentence_indices']!=center[side+'_indices'] for side in ('source','target')):
            raise ValueError('Corpus and candidate alignment membership differs')
    result=deepcopy(data)
    result['source']=deepcopy(corpus['documents']['source']);result['target']=deepcopy(corpus['documents']['target'])
    for task in result['semantic_tasks']:
        for group in task['word_links']:
            word=corpus['segments'][group['group_index']]['word_alignment']
            links=word.get('methods',{}).get(method,[])
            if group['links']!=links or group['status']!=word['status']:raise ValueError('Corpus and candidate token links differ')
            group['links']=deepcopy(links)
    return result
