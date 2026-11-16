"""Build reproducible neighboring Vecalign windows; offsets are zero-based."""
from copy import deepcopy
from src.processing.alignment.sentences import validate_alignment
from .context import expansion_needs


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


def build_windows(source, target, groups, *, radius=1, raw_lines=None,
                  strategy='fixed', token_counter=None, max_tokens=None, max_extra_groups=6):
    if type(radius) is not int or radius < 0:
        raise ValueError('Window radius must be a nonnegative integer')
    if strategy not in ('fixed', 'adaptive'):
        raise ValueError('Unknown window strategy')
    if strategy == 'adaptive':
        if radius != 1 or not callable(token_counter) or type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('Adaptive windows require radius=1, COMET tokenizer and token limit')
        if type(max_extra_groups) is not int or max_extra_groups < 0:
            raise ValueError('Invalid adaptive expansion limit')
    src, tgt = document_sentences(source), document_sentences(target)
    validate_alignment(groups, len(src), len(tgt))
    if raw_lines is not None and len(raw_lines) != len(groups):
        raise ValueError('Alignment line count mismatch')

    def side(records, indices):
        sentences = [{'index': i, 'sentence_id': records[i].get('sentence_id', i),
                      'text': records[i]['normalized_text']} for i in indices]
        return {'sentence_indices': indices, 'sentences': sentences,
                'text': ' '.join(r['text'] for r in sentences)}

    def assemble(center, ids):
        entries = []
        for i in ids:
            g = groups[i]
            line = raw_lines[i] if raw_lines is not None else f"{g['source_indices']}:{g['target_indices']}:{g['cost']}"
            entries.append({'group_index': i, **deepcopy(g), 'vecalign_line': line})
        return {'center_group_index': center, 'group_indices': ids,
                'center_alignment': deepcopy(entries[ids.index(center)]), 'alignments': entries,
                'source': side(src, [j for i in ids for j in groups[i]['source_indices']]),
                'target': side(tgt, [j for i in ids for j in groups[i]['target_indices']]),
                'alignment_gap_present': any(not groups[i]['source_indices'] or not groups[i]['target_indices'] for i in ids)}

    counts_cache = {}
    def counts(window):
        result = {}
        for side, key in [('source', 'src'), ('target', 'mt')]:
            text = window[side]['text']
            if text not in counts_cache:
                count = token_counter(text)
                if type(count) is not int or count < 0:
                    raise ValueError('Tokenizer must return a nonnegative integer')
                counts_cache[text] = count
            result[key] = counts_cache[text]
        return result

    windows = []
    for center in range(len(groups)):
        base = list(range(max(0, center-radius), min(len(groups), center+radius+1)))
        window = assemble(center, base)
        if strategy == 'adaptive':
            expansions, stopped, blocked = [], [], set()
            lengths = counts(window)
            if max(lengths.values()) > max_tokens:
                stopped.append('minimum_exceeds_token_limit')
            else:
                while len(expansions) < max_extra_groups:
                    needed = expansion_needs(window)
                    progressed = False
                    directions = ('right', 'left') if expansions and expansions[-1]['direction'] == 'left' else ('left', 'right')
                    for direction in directions:
                        if not needed[direction] or direction in blocked or len(expansions) >= max_extra_groups:
                            continue
                        ids = window['group_indices']
                        extra = ids[0]-1 if direction == 'left' else ids[-1]+1
                        if not 0 <= extra < len(groups):
                            stopped.append(direction+':document_boundary')
                            blocked.add(direction)
                            continue
                        proposal = assemble(center, list(range(min(ids[0], extra), max(ids[-1], extra)+1)))
                        proposal_lengths = counts(proposal)
                        if max(proposal_lengths.values()) > max_tokens:
                            stopped.append(direction+':token_limit')
                            blocked.add(direction)
                            continue
                        window, lengths = proposal, proposal_lengths
                        expansions.append({'direction': direction, 'group_index': extra, 'reasons': needed[direction]})
                        progressed = True
                        # Recompute boundary reasons after every accepted group.
                        break
                    if not progressed:
                        break
                if len(expansions) == max_extra_groups and any(expansion_needs(window).values()):
                    stopped.append('expansion_limit')
            window['context'] = {'strategy': 'adaptive', 'base_group_indices': base,
                                 'expansions': expansions, 'token_counts': lengths,
                                 'max_tokens': max_tokens, 'stop_reasons': stopped}
        windows.append(window)
    return windows
