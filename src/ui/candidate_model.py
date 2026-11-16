"""Exact normalized-text segments for the interactive candidate reader."""
from collections import defaultdict
from bisect import bisect_left, bisect_right


def candidate_payload(data, candidate, visible, signals, *, mode, identity):
    task = data['semantic_tasks'][candidate['group_index']]
    panes = {}
    eligible = {c['id'] for c in visible}
    for side in ('source', 'target'):
        doc = data[side]
        suspects, eligible_spans, active = defaultdict(list), [], defaultdict(list)
        eligible_sentences = defaultdict(list)
        for item in data['candidates']:
            spans = [s for signal in item['signals'] for s in signal[side+'_spans']]
            if not spans:
                spans = item[side+'_spans']
            for span in spans:
                sentence = doc['sentences'][span['sentence_index']]
                if (type(span['start']) is not int or type(span['end']) is not int
                        or not 0 <= span['start'] < span['end'] <= len(sentence['normalized_text'])
                        or sentence['normalized_text'][span['start']:span['end']] != span['quote']):
                    raise ValueError('Цитата кандидата не совпадает с текстом')
                suspects[span['sentence_index']].append(span)
                if item['id'] in eligible:
                    value = (span['document_start'], span['document_end'], item['id'], item['priority'], item['group_index'])
                    eligible_spans.append(value)
                    eligible_sentences[span['sentence_index']].append(value)
        for signal in signals:
            for span in signal[side+'_spans']:
                active[span['sentence_index']].append(span)
        if signals and not any(signal['source_spans'] or signal['target_spans'] for signal in signals):
            for span in candidate[side+'_spans']:
                active[span['sentence_index']].append(span)
        context = set(task['window'][side]['sentence_indices'])
        central = [s['sentence_index'] for s in candidate[side+'_spans']]
        focus = next((s['sentence_index'] for signal in signals for s in signal[side+'_spans']), next(iter(central), min(context) if context else None))
        # Endpoint indices avoid scanning every candidate for every highlighted word.
        endpoints = []
        for coordinate in (0, 1):
            best = {}
            for span in eligible_spans:
                position = span[coordinate]
                if position not in best or (-span[3], span[4]) < (-best[position][3], best[position][4]):
                    best[position] = span
            points = sorted(best)
            endpoints.append((points, best))
        sentences = []
        for sentence in doc['sentences']:
            i = sentence['index']
            if mode == 'Фрагмент' and i not in context:
                continue
            text = sentence['normalized_text']
            spans = suspects[i]
            boundaries = sorted({0, len(text), *(s[k] for s in spans+active[i] for k in ('start','end'))})
            pieces = []
            for start,end in zip(boundaries,boundaries[1:]):
                is_suspect = any(s['start'] < end and s['end'] > start for s in spans)
                is_active = any(s['start'] < end and s['end'] > start for s in active[i])
                pick = None
                if is_suspect and eligible_spans:
                    position = sentence['document_char_start']+(start+end)/2
                    nearby = [p for p in eligible_sentences[i] if p[0] <= position <= p[1]]
                    if not nearby:
                        starts, start_map = endpoints[0]
                        ends, end_map = endpoints[1]
                        right = bisect_left(starts, position)
                        left = bisect_right(ends, position)-1
                        if right < len(starts): nearby.append(start_map[starts[right]])
                        if left >= 0: nearby.append(end_map[ends[left]])
                    pick = min(nearby, key=lambda p:(max(p[0]-position,position-p[1],0),-p[3],p[4]))[2]
                pieces.append({'text':text[start:end], 'suspect':is_suspect, 'active':is_active, 'pick':pick})
            sentences.append({'index':i,'context':i in context,'focus':i==focus,'pieces':pieces})
        panes[side] = {'document':doc['document_id'],'sentences':sentences,
                       'missing_center':not central}
    return {'identity':identity,'selection':candidate['id'],'mode':mode,'panes':panes}
