"""Conservative RU/EN screening heuristics. Every signal needs review."""
import re
from collections import Counter
from decimal import Decimal
from .locations import locate

ORDINALS = [
    (1, r'перв\w*', 'first'), (2, r'втор\w*', 'second'), (3, r'трет\w*', 'third'),
    (4, r'четв[её]рт\w*', 'fourth'), (5, r'пят\w*', 'fifth'), (6, r'шест\w*', 'sixth'),
    (7, r'седьм\w*', 'seventh'), (8, r'восьм\w*', 'eighth'), (9, r'девят\w*', 'ninth'),
    (10, r'десят\w*', 'tenth'), (11, r'одиннадцат\w*', 'eleventh'),
    (12, r'двенадцат\w*', 'twelfth'), (20, r'двадцат\w*', 'twentieth'),
    (21, r'двадцать\s+перв\w*', 'twenty-first')]
UNITS = {
    'ru': r'км|километр\w*|см|сантиметр\w*|м|метр\w*|кг|килограмм\w*|г|грамм\w*',
    'en': r'km|kilomet(?:er|re)s?|cm|centimet(?:er|re)s?|m|met(?:er|re)s?|kg|kilograms?|g|grams?'}
DEFAULT_GLOSSARY = [
    {'id': 'lung_sarcoma', 'source': r'сарком\w*\s+л[её]гк\w*', 'target': r'(?:lung|pulmonary)\s+sarcoma|sarcoma\s+of\s+(?:the\s+)?lung'},
    {'id': 'heart_attack', 'source': r'инфаркт\w*|сердечн\w*\s+приступ\w*', 'target': r'heart\s+attack|myocardial\s+infarction'},
]
STOP = {
    'ru': set('и а но в на к с со у по от до за из не ни да нет он она они его ее её это как что бы же то мы вы я ты их был была'.split()),
    'en': set('a an the and or but in on at to of for from with by not no he she it they we you i his her their is are was were be been do did does have has had'.split())}


def matches(document, indices, pattern):
    return [locate(document, i, m.start(), m.end()) for i in indices
            for m in re.finditer(pattern, document['sentences'][i]['normalized_text'], re.I)]


def quantities(document, indices):
    lang = document['language']
    result, occupied = [], set()
    for i in indices:
        text = document['sentences'][i]['normalized_text']
        # Only spelled-out centuries are recognized as ordinal quantities here.
        century_ranges = []
        for value, ru, en in sorted(ORDINALS, reverse=True):
            pattern = rf'\b(?:{ru})\s+век\w*\b' if lang == 'ru' else rf'\b{en}[ -]+centur(?:y|ies)\b'
            for m in re.finditer(pattern, text, re.I):
                if any(m.start() < end and m.end() > start for start, end in century_ranges):
                    continue
                century_ranges.append((m.start(), m.end()))
                result.append(('century', str(value), locate(document, i, m.start(), m.end())))
        numeric_century = r'(?<!\w)(\d+)(?:-?(?:го|й|ый|ой|th|st|nd|rd))?[ -]+(?:век\w*|centur(?:y|ies))\b'
        for m in re.finditer(numeric_century, text, re.I):
            result.append(('century', str(int(m.group(1))), locate(document, i, m.start(), m.end())))
            occupied.update((i, j) for j in range(m.start(), m.end()))
        pattern = rf'(?<!\w)\d+(?:[.,]\d+)?\s*(?:{UNITS.get(lang, "(?!)")})(?!\w)'
        for m in re.finditer(pattern, text, re.I):
            raw = m.group().lower()
            number = re.match(r'\d+(?:[.,]\d+)?', raw).group()
            unit = raw[len(number):].strip()
            if unit.startswith(('км', 'кило', 'km', 'kilo')):
                kind, factor = ('mass', 1000) if unit.startswith(('кг', 'килограмм', 'kg', 'kilogram')) else ('length', 1000)
            elif unit.startswith(('см', 'санти', 'cm', 'centi')):
                kind, factor = 'length', Decimal('.01')
            elif unit.startswith(('кг', 'kg')):
                kind, factor = 'mass', 1000
            elif unit.startswith(('г', 'g')):
                kind, factor = 'mass', 1
            else:
                kind, factor = 'length', 1
            result.append((kind, str((Decimal(number.replace(',', '.')) * factor).normalize()), locate(document, i, m.start(), m.end())))
            occupied.update((i, j) for j in range(m.start(), m.end()))
        for m in re.finditer(r'(?<!\w)\d+(?:[.,]\d+)?(?!\w)', text):
            if (i, m.start()) not in occupied:
                result.append(('number', str(Decimal(m.group().replace(',', '.')).normalize()), locate(document, i, m.start(), m.end())))
    return result


def lexical_signals(words, block, glossary):
    source, target = words['source'], words['target']
    si, ti = block['source']['sentence_indices'], block['target']['sentence_indices']
    signals = []
    def add(code, message, a, b, weight):
        signals.append({'code': code, 'message': message, 'source_spans': a, 'target_spans': b, 'weight': weight})
    a, b = quantities(source, si), quantities(target, ti)
    if Counter((kind, value) for kind, value, _ in a) != Counter((kind, value) for kind, value, _ in b):
        add('quantity_mismatch', 'Числа, века или величины требуют сопоставления; возможен сдвиг границ.', [x[2] for x in a], [x[2] for x in b], 35)
    patterns = {'ru': r'\b(?:не|нет|никогда|нельзя|невозможно)\b', 'en': r"\b(?:not|no|never|cannot|neither|without)\b|n['’]t\b"}
    a, b = matches(source, si, patterns.get(source['language'], '(?!)')), matches(target, ti, patterns.get(target['language'], '(?!)'))
    if bool(a) != bool(b):
        add('negation_mismatch', 'Различается явное отрицание; возможны идиома или перефразирование.', a, b, 15)
    concepts = {'source': [], 'target': []}
    term_spans = {'source': [], 'target': []}
    for entry in glossary:
        a, b = matches(source, si, entry['source']), matches(target, ti, entry['target'])
        for side, spans in [('source', a), ('target', b)]:
            if spans:
                concepts[side].append(entry['id'])
                term_spans[side].extend(spans)
    if set(concepts['source']) != set(concepts['target']):
        add('terminology_mismatch', f"Различаются понятия словаря: {concepts['source']} / {concepts['target']}. Проверить контекст и границы.",
            term_spans['source'], term_spans['target'], 40)
    word = block['word_alignment']
    if word['status'] == 'computed':
        links = word['methods'][words['selected_method']]
        spans = {}
        for side, doc in [('source', source), ('target', target)]:
            linked = {link[side+'_token_id'] for link in links}
            selected = [doc['tokens'][i] for i in block[side]['token_ids']
                        if i not in linked and not doc['tokens'][i]['is_punctuation']
                        and doc['tokens'][i]['text'].casefold() not in STOP.get(doc['language'], set())
                        and len(doc['tokens'][i]['text']) > 1]
            spans[side] = [locate(doc, t['sentence_index'], t['sentence_char_start'], t['sentence_char_end']) for t in selected]
        if spans['source'] or spans['target']:
            add('unlinked_content', 'Содержательные токены без связей SimAlign; это не доказательство пропуска.', spans['source'], spans['target'], 10)
    else:
        add('word_alignment_unavailable', 'Выравнивание слов для группы не вычислено.', [], [], 5)
    return signals
