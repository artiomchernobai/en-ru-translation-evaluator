"""Portable fragment reports. All saved spans use fragment-relative Unicode offsets."""
from copy import deepcopy
from src.processing.analysis.fragments import fragment_id

SIDES = ('source', 'target')


def validate_fragment(fragment):
    for side in SIDES:
        text, mapping = fragment[side]['text'], fragment[side]['sentences']
        offset = 0
        for i, sentence in enumerate(mapping):
            if sentence['start'] != offset or not offset <= sentence['end'] <= len(text):
                raise ValueError('Invalid fragment sentence map')
            offset = sentence['end'] + (1 if i+1 < len(mapping) else 0)
            if i+1 < len(mapping) and text[sentence['end']:offset] != '\n':
                raise ValueError('Invalid fragment sentence separator')
        if offset != len(text): raise ValueError('Incomplete fragment sentence map')
        for record in fragment['findings']+fragment['observations']:
            for span in record[side+'_spans']:
                start, end = span['start'], span['end']
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != span['quote']:
                    raise ValueError('Invalid fragment-relative citation')
                if sum(s['start'] <= start < end <= s['end'] for s in mapping) != 1:
                    raise ValueError('Citation outside fragment sentence map')


def fragment_text(sentences):
    text, mapping = '', []
    for sentence in sentences:
        if mapping: text += '\n'
        start = len(text)
        text += sentence['text']
        mapping.append({'index': sentence['index'], 'sentence_id': sentence['sentence_id'],
                        'start': start, 'end': len(text)})
    return {'text': text, 'sentences': mapping}


def localize(record, fragment):
    result = {k: deepcopy(v) for k, v in record.items() if k not in ('source_spans', 'target_spans', 'group_indices')}
    for side in SIDES:
        mapping = {s['index']: s for s in fragment[side]['sentences']}
        result[side+'_spans'] = []
        for span in record[side+'_spans']:
            sentence = mapping[span['sentence_index']]
            start, end = sentence['start']+span['start'], sentence['start']+span['end']
            if fragment[side]['text'][start:end] != span['quote']:
                raise ValueError('Cannot save mismatched fragment citation')
            result[side+'_spans'].append({'start': start, 'end': end, 'quote': span['quote']})
    return result


def compact_report(report):
    if report.get('schema_version') == 2:
        return deepcopy(report)
    result = {key: deepcopy(report[key]) for key in (
        'source_document', 'target_document', 'window_config', 'model', 'scope', 'run_fingerprint',
        'selection', 'status', 'metrics', 'inputs', 'updated_at') if key in report}
    result.update(schema_version=2, assessment_unit=report.get('assessment_unit','central_group_legacy'),
                  anchor_role=report.get('anchor_role','legacy'),
                  offsets={'relative_to': 'fragment_side_text', 'unit': 'unicode_code_point', 'end': 'exclusive'},
                  documents={side: {key: report[side][key] for key in ('document_id', 'language', 'fingerprint')} for side in SIDES},
                  fragments=[])
    for raw in report['fragments']:
        fragment = {key: deepcopy(raw[key]) for key in (
            'group_index', 'fragment_id', 'anchor_group_indices', 'candidate_ids', 'priority', 'status',
            'request_hash', 'reused', 'error', 'elapsed_seconds', 'context_group_indices') if key in raw}
        for side in SIDES:
            fragment[side] = fragment_text(raw.get('context', {}).get(side, []))
        if 'fragment_id' not in fragment:
            window = {side: {'sentence_indices': [s['index'] for s in fragment[side]['sentences']]} for side in SIDES}
            fragment['fragment_id'] = fragment_id(window)
        fragment.setdefault('anchor_group_indices', [raw['group_index']])
        for section in ('findings', 'observations'):
            fragment[section] = []
            for i, item in enumerate(raw.get(section, [])):
                localized = localize(item, fragment)
                localized['id'] = f'{fragment["fragment_id"]}/{section}-{i}'
                localized['marker'] = f'{"E" if section == "findings" else "O"}{i+1}'
                fragment[section].append(localized)
        fragment['rejected_count'] = len(raw.get('rejected', []))
        result['fragments'].append(fragment)
    return result


def absolute_record(record, fragment, documents):
    """Resolve validated fragment spans through sentence maps; never search the full book."""
    from src.processing.analysis.locations import locate
    result = {k: deepcopy(v) for k, v in record.items() if k not in ('source_spans', 'target_spans')}
    for side in SIDES:
        text = fragment[side]['text']
        result[side+'_spans'] = []
        for span in record[side+'_spans']:
            start, end = span['start'], span['end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != span['quote']:
                raise ValueError('Invalid fragment-relative citation')
            matches = [s for s in fragment[side]['sentences'] if s['start'] <= start < end <= s['end']]
            if len(matches) != 1: raise ValueError('Citation outside fragment sentence map')
            sentence = matches[0]
            location = locate(documents[side], sentence['index'], start-sentence['start'], end-sentence['start'])
            if location['quote'] != span['quote']: raise ValueError('Corpus text differs from report fragment')
            result[side+'_spans'].append(location)
    return result


def restore_report(report, data):
    if report.get('schema_version') != 2: return report
    if any(report['documents'][side]['fingerprint'] != data[side]['fingerprint'] for side in SIDES):
        raise ValueError('MQM report and candidate documents differ')
    result = deepcopy(report)
    result['schema_version'] = 1
    for side in SIDES: result[side] = data[side]
    result['fragments'] = []
    for raw in report['fragments']:
        fragment = deepcopy(raw)
        for side in SIDES:
            mapped = raw[side]['sentences']
            for s in mapped:
                if raw[side]['text'][s['start']:s['end']] != data[side]['sentences'][s['index']]['normalized_text']:
                    raise ValueError('Cached MQM input text differs')
            fragment[side+'_sentence_indices'] = [s['index'] for s in mapped]
        fragment['context'] = {side: [{'index': s['index'], 'sentence_id': s['sentence_id'],
                                     'text': raw[side]['text'][s['start']:s['end']]} for s in raw[side]['sentences']] for side in SIDES}
        for section in ('findings', 'observations'):
            fragment[section] = [absolute_record(item, raw, data) for item in raw[section]]
        fragment['rejected'] = []
        result['fragments'].append(fragment)
    return result


def diagnostics(report):
    fragments = []
    for f in report['fragments']:
        value = {key: deepcopy(f[key]) for key in ('fragment_id', 'group_index', 'status', 'attempts', 'rejected', 'error') if key in f}
        value['citation_resolutions'] = [
            {'section': section, 'item': i, 'side': side, **deepcopy(span)}
            for section in ('findings', 'observations') for i, record in enumerate(f[section])
            for side in SIDES for span in record[side+'_spans']
            if 'index_resolution' in span or 'offset_resolution' in span]
        fragments.append(value)
    return {'schema_version': 1, 'run_fingerprint': report['run_fingerprint'], 'fragments': fragments}
