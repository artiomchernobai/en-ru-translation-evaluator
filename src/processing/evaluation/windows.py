"""Build reproducible neighboring Vecalign windows; offsets are zero-based."""
from copy import deepcopy
from src.processing.alignment.sentences import validate_alignment


def document_sentences(document):
    records = document.get('sentences')
    if not isinstance(records, list):
        raise ValueError('Normalized document requires sentences')
    if 'sentence_count' in document and document['sentence_count'] != len(records):
        raise ValueError('sentence_count does not match sentences')
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get('normalized_text'), str) or not record['normalized_text'].strip():
            raise ValueError('Each sentence requires nonempty normalized_text')
    return records


def build_windows(source, target, groups, *, radius=1, raw_lines=None):
    if type(radius) is not int or radius < 0:
        raise ValueError('Window radius must be a nonnegative integer')
    src, tgt = document_sentences(source), document_sentences(target)
    validate_alignment(groups, len(src), len(tgt))
    if raw_lines is not None and len(raw_lines) != len(groups):
        raise ValueError('Alignment line count mismatch')

    def side(records, indices):
        sentences = [{'index': i, 'sentence_id': records[i].get('sentence_id', i),
                      'text': records[i]['normalized_text']} for i in indices]
        return {'sentence_indices': indices, 'sentences': sentences,
                'text': ' '.join(r['text'] for r in sentences)}

    windows = []
    for center, group in enumerate(groups):
        ids = list(range(max(0, center-radius), min(len(groups), center+radius+1)))
        entries = []
        for i in ids:
            g = groups[i]
            line = raw_lines[i] if raw_lines is not None else f"{g['source_indices']}:{g['target_indices']}:{g['cost']}"
            entries.append({'group_index': i, **deepcopy(g), 'vecalign_line': line})
        windows.append({'center_group_index': center, 'group_indices': ids,
                        'center_alignment': deepcopy(entries[ids.index(center)]),
                        'alignments': entries,
                        'source': side(src, [j for i in ids for j in groups[i]['source_indices']]),
                        'target': side(tgt, [j for i in ids for j in groups[i]['target_indices']]),
                        'alignment_gap_present': any(not groups[i]['source_indices'] or not groups[i]['target_indices'] for i in ids)})
    return windows
