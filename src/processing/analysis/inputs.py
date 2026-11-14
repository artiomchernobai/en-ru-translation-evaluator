"""Reject stale or incompatible word/COMET results before joining them."""
import math
from src.processing.alignment.sentences import validate_alignment, ALIGNMENT, parse_indices


def validate_inputs(words, comet):
    if words.get('schema_version') != '1.0' or comet.get('schema_version') != 1:
        raise ValueError('Unsupported word alignment or COMET schema')
    if words.get('selected_method') not in ('inter', 'itermax', 'mwmf'):
        raise ValueError('Unknown selected SimAlign method')
    for side in ('source', 'target'):
        doc = words[side]
        if doc['document_id'] != comet[side + '_document']:
            raise ValueError('Document pair mismatch')
        texts = [s['normalized_text'] for s in doc['sentences']]
        if doc['text'] != '\n'.join(texts):
            raise ValueError('Normalized document text mismatch')
        offset = 0
        for i, sentence in enumerate(doc['sentences']):
            if sentence['index'] != i or sentence['document_char_start'] != offset:
                raise ValueError('Invalid sentence offsets')
            if sentence['document_char_end'] != offset + len(texts[i]):
                raise ValueError('Invalid sentence end')
            offset += len(texts[i]) + 1
        previous = 0
        membership = [[] for _ in doc['sentences']]
        for i, token in enumerate(doc['tokens']):
            start, end = token['document_char_start'], token['document_char_end']
            if token['token_id'] != i or not previous <= start < end <= len(doc['text']) or doc['text'][start:end] != token['text']:
                raise ValueError('Invalid token coordinates')
            if type(token['sentence_index']) is not int or not 0 <= token['sentence_index'] < len(membership):
                raise ValueError('Invalid token sentence index')
            sentence = doc['sentences'][token['sentence_index']]
            membership[token['sentence_index']].append(i)
            if (token['sentence_id'] != sentence['sentence_id']
                    or token['sentence_char_start'] != start - sentence['document_char_start']
                    or token['sentence_char_end'] != end - sentence['document_char_start']):
                raise ValueError('Token sentence coordinates differ')
            if not sentence['document_char_start'] <= start < end <= sentence['document_char_end']:
                raise ValueError('Token outside sentence')
            previous = end
        if any(sentence['token_ids'] != ids for sentence, ids in zip(doc['sentences'], membership)):
            raise ValueError('Sentence token membership mismatch')
    blocks, windows = words['blocks'], comet['windows']
    if len(blocks) != len(windows):
        raise ValueError('Word/COMET alignment group count mismatch')
    groups = [{'source_indices': b['source']['sentence_indices'],
               'target_indices': b['target']['sentence_indices'], 'cost': b['vecalign_cost']} for b in blocks]
    validate_alignment(groups, len(words['source']['sentences']), len(words['target']['sentences']))
    seen = {'source': set(), 'target': set()}
    radius = comet['radius']
    if type(radius) is not int or radius < 0:
        raise ValueError('Invalid COMET window radius')
    for i, (block, window) in enumerate(zip(blocks, windows)):
        if block['block_id'] != i or window['center_group_index'] != i:
            raise ValueError('Invalid group ordering')
        ids = list(range(max(0, i-radius), min(len(blocks), i+radius+1)))
        if window['group_indices'] != ids or len(window['alignments']) != len(ids):
            raise ValueError('Invalid COMET window groups')
        for j, entry in zip(ids, window['alignments']):
            match = ALIGNMENT.fullmatch(entry['vecalign_line'])
            if (entry['group_index'] != j or any(entry[k] != groups[j][k] for k in groups[j]) or not match
                    or parse_indices(match[1]) != groups[j]['source_indices']
                    or parse_indices(match[2]) != groups[j]['target_indices'] or float(match[3]) != groups[j]['cost']):
                raise ValueError('Vecalign rows differ between inputs')
        if window['center_alignment'] != window['alignments'][ids.index(i)]:
            raise ValueError('COMET center mismatch')
        verdict = window['verdict']
        if verdict['status'] not in ('scored', 'empty_side', 'too_long'):
            raise ValueError('Unknown COMET status')
        if verdict['status'] == 'scored' and (type(verdict['score']) not in (int, float) or not math.isfinite(verdict['score'])):
            raise ValueError('Invalid COMET score')
        for side in ('source', 'target'):
            indices = [index for j in ids for index in groups[j][side+'_indices']]
            records = words[side]['sentences']
            if (window[side]['sentence_indices'] != indices
                    or window[side]['text'] != ' '.join(records[k]['normalized_text'] for k in indices)
                    or [(s['index'], s['sentence_id'], s['text']) for s in window[side]['sentences']] !=
                       [(k, records[k]['sentence_id'], records[k]['normalized_text']) for k in indices]):
                raise ValueError('COMET text differs from word-alignment text')
            expected = [token_id for j in block[side]['sentence_indices'] for token_id in records[j]['token_ids']]
            if block[side]['token_ids'] != expected or seen[side].intersection(expected):
                raise ValueError('Invalid token membership')
            seen[side].update(expected)
        word = block['word_alignment']
        if word['status'] not in ('computed', 'not_applicable', 'skipped_too_long'):
            raise ValueError('Unknown word alignment status')
        if word['status'] == 'computed' and words['selected_method'] not in word['methods']:
            raise ValueError('Missing selected SimAlign method')
        for links in word.get('methods', {}).values():
            for link in links:
                if any(link[side+'_token_id'] not in block[side]['token_ids'] for side in ('source', 'target')):
                    raise ValueError('Word link outside group')
    for key in ('source', 'target', 'alignment'):
        a = words.get('inputs', {}).get('sentence_alignment' if key == 'alignment' else key, {}).get('sha256')
        b = comet.get('inputs', {}).get(key, {}).get('sha256')
        if a and b and a != b:
            raise ValueError(f'Stale input fingerprint: {key}')
