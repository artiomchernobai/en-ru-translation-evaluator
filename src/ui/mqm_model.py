"""MQM reader: portable excerpts and verified corpus projection into complete texts."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from src.processing.mqm.reports import compact_report, absolute_record, fragment_text, validate_fragment
from src.processing.analysis.fragments import review_fragments, window_key

SIDES = ('source', 'target')


def load_review(report, root):
    documents = {side: report[side] for side in SIDES} if report.get('schema_version') == 1 else None
    compact = compact_report(report)
    for fragment in compact['fragments']: validate_fragment(fragment)
    candidates, warning = None, None
    for kind, directory in [('corpus', 'corpus'), ('candidates', 'review_candidates')]:
        reference = report.get('inputs', {}).get(kind)
        if not reference: continue
        basename = reference['path'].replace('\\', '/').rsplit('/', 1)[-1]
        paths = [Path(root)/'data'/directory/basename, Path(reference['path'])]
        path = next((p for p in paths if p.is_file()), None)
        if path is None: continue
        try:
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != reference['sha256']:
                raise ValueError('Хеш файла не совпадает с отчётом')
            value = json.loads(payload.decode('utf-8-sig'))
            docs = value['documents'] if kind == 'corpus' else {side: value[side] for side in SIDES}
            if any(docs[side]['fingerprint'] != compact['documents'][side]['fingerprint'] for side in SIDES):
                raise ValueError('Версия текста не совпадает с отчётом')
            documents = docs
            if kind == 'candidates': candidates = value
        except (OSError, ValueError, KeyError, TypeError) as error:
            warning = f'Не удалось подключить {kind}: {error}'
    return compact, documents, candidates, warning


def review_items(report, candidates):
    items = deepcopy(report['fragments'])
    keys = {tuple(tuple(s['index'] for s in f[side]['sentences']) for side in SIDES) for f in items}
    if candidates is not None:
        for unit in review_fragments(candidates):
            window = candidates['semantic_tasks'][unit['group_index']]['window']
            if window_key(window) in keys: continue
            items.append({**{k: unit[k] for k in ('fragment_id', 'group_index', 'anchor_group_indices', 'priority', 'context_group_indices')},
                          **{side: fragment_text(window[side]['sentences']) for side in SIDES},
                          'status': 'not_analyzed', 'findings': [], 'observations': []})
            keys.add(window_key(window))
    return items


def mqm_payload(report, items, selected, *, documents, candidates, mode, selected_finding, identity):
    current = items[selected]
    marks = {side: defaultdict(list) for side in SIDES}
    picks = {}
    for index, fragment in enumerate(items):
        picks[fragment['fragment_id']] = (index, '__all__')
        for section in ('findings', 'observations'):
            for record in fragment[section]:
                picks[record['id']] = (index, record['id'])
                projected = absolute_record(record, fragment, documents) if documents is not None else None
                for side in SIDES:
                    spans = projected[side+'_spans'] if projected else record[side+'_spans']
                    for span in spans:
                        sentence_index = span['sentence_index'] if projected else None
                        marks[side][sentence_index].append((span, record, index))
    panes = {}
    for side in SIDES:
        source = current[side]
        context = {s['index'] for s in source['sentences']}
        suspects = defaultdict(list)
        if candidates:
            # Every candidate remains visible, including candidates without an MQM verdict yet.
            for candidate in candidates['candidates']:
                for signal in candidate['signals']:
                    for span in signal[side+'_spans'] or candidate[side+'_spans']:
                        suspects[span['sentence_index']].append(span)
        rows = []
        sentences = documents[side]['sentences'] if documents is not None else [
            {'index': None, 'normalized_text': source['text']}]
        focus_used = False
        for sentence in sentences:
            i, text = sentence['index'], sentence['normalized_text']
            if mode == 'Фрагмент' and documents is not None and i not in context: continue
            spans = marks[side][i]
            if documents is None:
                spans = [(s, r, n) for s, r, n in spans if n == selected]
            ranges = suspects[i]
            boundaries = sorted({0, len(text), *(s[k] for s, _, _ in spans for k in ('start', 'end')),
                                 *(s[k] for s in ranges for k in ('start', 'end'))})
            pieces = []
            for start, end in zip(boundaries, boundaries[1:]):
                matching = [(s, r, n) for s, r, n in spans if s['start'] < end and s['end'] > start]
                matching.sort(key=lambda v: (v[2] != selected, v[1]['id'] != selected_finding,
                                             -{'critical': 3, 'major': 2, 'minor': 1}.get(v[1].get('severity'), 0)))
                match = matching[0] if matching else None
                suspect = any(s['start'] < end and s['end'] > start for s in ranges)
                pick, marker, severity, title = None, None, None, None
                active = False
                if match:
                    _, record, n = match
                    pick, marker, severity = record['id'], record['marker'], record.get('severity')
                    title = f"{marker} · {record.get('category', record.get('kind'))} · {severity or 'наблюдение'}: {record['explanation']}"
                    active = n == selected and selected_finding in ('__all__', record['id'])
                elif suspect:
                    # Choose a window containing this sentence; favor the current window.
                    containing = [n for n, f in enumerate(items) if any(s['index'] == i for s in f[side]['sentences'])]
                    if containing: pick = items[selected if selected in containing else containing[0]]['fragment_id']
                pieces.append({'text': text[start:end], 'suspect': suspect or bool(match), 'active': active,
                               'pick': pick, 'marker': marker, 'severity': severity, 'title': title})
            focus = not focus_used and (any(p['active'] for p in pieces) or i in context or documents is None)
            if focus: focus_used = True
            rows.append({'index': i, 'context': i in context or documents is None, 'focus': focus, 'pieces': pieces})
        # Prefer a selected finding over the beginning of its context when centering.
        active_rows = [row for row in rows if any(p['active'] for p in row['pieces'])]
        if active_rows:
            for row in rows: row['focus'] = row is active_rows[0]
        panes[side] = {'document': f"{'Оригинал' if side == 'source' else 'Перевод'} · {report['documents'][side]['document_id']}",
                       'sentences': rows, 'missing_center': not source['sentences'], 'has_review_area': False}
    return {'identity': identity, 'selection': current['fragment_id'], 'highlight': [selected_finding],
            'mode': mode, 'panes': panes}, picks
