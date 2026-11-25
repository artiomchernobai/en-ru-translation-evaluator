"""Auditable filtering of model proposals, separate from semantic correctness."""
import numpy as np


def filter_links(raw, scores, minimum, margin_minimum, secondary_drop):
    accepted, evidence = {}, {}
    for method, pairs in raw.items():
        records = []
        for i, j in sorted(set(pairs)):
            i, j = int(i), int(j)  # SimAlign may return NumPy scalars; JSON needs Python ints.
            score = float(scores[i, j])
            row, column = np.delete(scores[i], j), np.delete(scores[:, j], i)
            margin = max(score - float(row.max()) if len(row) else score + 1,
                         score - float(column.max()) if len(column) else score + 1)
            reasons = ['low_similarity'] if score < minimum else ([] if margin >= margin_minimum else ['ambiguous'])
            records.append(dict(source_index=i, target_index=j, similarity=score, margin=margin,
                                accepted=not reasons, role='primary' if not reasons else None, reasons=reasons))
        # Secondary proposals must attach to a primary anchor and introduce a new endpoint.
        # Only model-proposed edges qualify; dense similarity alternatives never become links here.
        primary = [r for r in records if r['accepted']]
        src = {r['source_index'] for r in primary}
        tgt = {r['target_index'] for r in primary}
        for r in sorted(records, key=lambda r: -r['similarity']):
            i, j = r['source_index'], r['target_index']
            if r['accepted'] or r['similarity'] < minimum or ((i in src) == (j in tgt)):
                continue
            anchors = [a['similarity'] for a in primary if a['source_index'] == i or a['target_index'] == j]
            if anchors and r['similarity'] >= max(anchors) - secondary_drop:
                r.update(accepted=True, role='secondary', reasons=[])
                src.add(i); tgt.add(j)
        accepted[method] = [(r['source_index'], r['target_index']) for r in records if r['accepted']]
        evidence[method] = records
    return accepted, evidence


def token_evidence(word, side, token_id, method, *, weak_minimum=.45, weak_delta=.1):
    """Classify an unmatched token without declaring an omission or inventing a link."""
    records = word.get('link_evidence', {}).get(method)
    entry = next((e for e in word.get('token_map', {}).get(side, []) if e['token_id'] == token_id), {})
    alternatives = entry.get('alternatives', [])
    best = max((a['similarity'] for a in alternatives), default=-1)
    weak = [a for a in alternatives if a['similarity'] >= weak_minimum and a['similarity'] >= best-weak_delta]
    if records is None:
        return 'unlinked', weak  # Old results cannot explain why an edge is missing.
    proposed = [r for r in records if r[side+'_token_id'] == token_id]
    return ('filter_rejected' if proposed else 'model_unlinked'), weak
