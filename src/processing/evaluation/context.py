"""Deterministic boundary heuristics; no model or file dependencies."""
import re

ENDING = re.compile(r'[.!?…][»”\"\'’)]*\s*$')
SHORT_REPLY = re.compile(r"^[\s—–\-'\"«“]*(?:да|нет|гм|ага|конечно|yes|no|well|hm|sure)\b", re.I)
ATTRIBUTION = re.compile(r"^[\s—–\-'\"«“]*(?:(?:he|she|they|berlioz)\s+)?(?:said|asked|replied|answered|сказал\w*|спросил\w*|ответил\w*|продолжил\w*)\b", re.I)


def expansion_needs(window):
    reasons = {'left': [], 'right': []}
    for side in ('source', 'target'):
        text = window[side]['text'].strip()
        records = window[side]['sentences']
        if not text:
            reasons['left'].append(f'{side}:empty_side')
            reasons['right'].append(f'{side}:empty_side')
            continue
        head = text.lstrip('—–- \"\'«“(')
        if head and (head[0].islower() or ATTRIBUTION.match(text)):
            reasons['left'].append(f'{side}:continuation_or_attribution')
        if not ENDING.search(text) or re.search(r'[,;:—–…][»”\"\'’)]*\s*$', text) or text.endswith('...'):
            reasons['right'].append(f'{side}:unfinished_tail')
        for opening, closing in [('«', '»'), ('“', '”')]:
            balance = text.count(opening) - text.count(closing)
            if balance:
                reasons['right' if balance > 0 else 'left'].append(f'{side}:unbalanced_quote')
        # Apostrophes within English words are not quotation delimiters.
        quotes = re.sub(r"(?<=\w)['’](?=\w)", '', text)
        for quote in ('"', "'"):
            if quotes.count(quote) % 2:
                direction = 'right' if text.lstrip('—–- ').startswith(quote) else 'left'
                reasons[direction].append(f'{side}:open_straight_quote')
        center_ids = set(window['center_alignment'][side+'_indices'])
        central = ' '.join(r['text'] for r in records if r['index'] in center_ids)
        preceding = ' '.join(r['text'] for r in records if center_ids and r['index'] < min(center_ids))
        if SHORT_REPLY.match(central) and len(central.split()) <= 10 and '?' not in preceding:
            reasons['left'].append(f'{side}:short_reply_needs_question')
    return {direction: list(dict.fromkeys(values)) for direction, values in reasons.items()}
