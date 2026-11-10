"""Validated, language-independent display data for word-alignment review."""
from __future__ import annotations

import re
import math
import unicodedata
from src.processing.alignment.words import TOKEN_PATTERN
from src.processing.preprocessing.cleaning import clean_text

METHODS = ('itermax', 'inter', 'mwmf')
CITATION = re.compile(r'\[\s*\d{1,4}(?:\s*[,;\-–—]\s*\d{1,4})*\s*\]')
QUOTE_KEYS = str.maketrans({'“':'"', '”':'"', '«':'"', '»':'"', '‘':"'", '’':"'", '`':"'"})


def validate_review(data: dict) -> None:
    if not isinstance(data, dict) or data.get('schema_version') != '1.0':
        raise ValueError('Ожидается JSON выравнивания слов версии 1.0')
    for side in ('source', 'target'):
        doc = data.get(side, {})
        if not isinstance(doc, dict) or not isinstance(doc.get('text'), str) or not isinstance(doc.get('tokens'), list):
            raise ValueError(f'Отсутствуют текст или токены: {side}')
        last_end = 0
        for index, token in enumerate(doc['tokens']):
            if not isinstance(token, dict) or type(token.get('token_id')) is not int or token['token_id'] != index:
                raise ValueError(f'Некорректный ID токена: {side}/{index}')
            start, end = token.get('document_char_start'), token.get('document_char_end')
            if (type(start) is not int or type(end) is not int or not last_end <= start < end <= len(doc['text'])
                    or doc['text'][start:end] != token.get('text')):
                raise ValueError(f'Текст и смещения токена не совпадают: {side}/{index}')
            last_end = end
    blocks = data.get('blocks')
    if not isinstance(blocks, list):
        raise ValueError('Отсутствует массив blocks')
    seen = {'source': set(), 'target': set()}
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or type(block.get('block_id')) is not int or block['block_id'] != index:
            raise ValueError('Некорректная нумерация блоков')
        if type(block.get('vecalign_cost')) not in (int, float) or not math.isfinite(block['vecalign_cost']):
            raise ValueError('Некорректная стоимость Vecalign')
        for side in seen:
            if not isinstance(block.get(side), dict):
                raise ValueError(f'Отсутствует сторона блока: {side}')
            ids = block[side].get('token_ids')
            if not isinstance(ids, list):
                raise ValueError(f'Отсутствуют token_ids: {side}')
            for token_id in ids:
                if type(token_id) is not int or not 0 <= token_id < len(data[side]['tokens']) or token_id in seen[side]:
                    raise ValueError(f'Некорректная принадлежность токена блоку: {side}/{token_id}')
                seen[side].add(token_id)
        word = block.get('word_alignment', {})
        if not isinstance(word, dict):
            raise ValueError('Некорректные данные словесного выравнивания')
        if word.get('status') not in ('computed', 'not_applicable', 'skipped_too_long'):
            raise ValueError('Неизвестное состояние выравнивания слов')
        methods = word.get('methods', {})
        if not isinstance(methods, dict):
            raise ValueError('Некорректный массив методов')
        if word['status'] != 'computed' and any(methods.values()):
            raise ValueError('Необработанный блок содержит словесные связи')
        for method, links in methods.items():
            if method not in METHODS or not isinstance(links, list):
                raise ValueError('Некорректный метод выравнивания')
            for link in links:
                if not isinstance(link, dict):
                    raise ValueError('Некорректная связь токенов')
                for side in seen:
                    token_id = link.get(f'{side}_token_id')
                    if type(token_id) is not int or token_id not in block[side]['token_ids']:
                        raise ValueError('Связь указывает за пределы блока')
    for side in seen:
        if seen[side] != set(range(len(data[side]['tokens']))):
            raise ValueError(f'Не все токены принадлежат блокам: {side}')


def _key(text: str) -> str:
    return unicodedata.normalize('NFC', clean_text(text, remove_numeric_references=False)).translate(QUOTE_KEYS)


def original_projection(doc: dict, raw: str | None) -> tuple[str, list[tuple[int, int]], str]:
    """Use raw offsets only if the entire canonical token sequence agrees.

    Whitespace/quote forms may differ; edited/missing/reordered words trigger a
    visible normalized-text fallback. No fuzzy matching of repeated words.
    """
    normal = [(t['document_char_start'], t['document_char_end']) for t in doc['tokens']]
    if raw is None:
        return doc['text'], normal, 'Нормализованный текст · исходный файл не найден'
    references = list(CITATION.finditer(raw))
    raw_tokens = []
    ref_index = 0
    for match in TOKEN_PATTERN.finditer(raw):
        while ref_index < len(references) and references[ref_index].end() <= match.start():
            ref_index += 1
        if ref_index < len(references) and references[ref_index].start() <= match.start() < references[ref_index].end():
            continue
        raw_tokens.append(match)
    if [_key(m.group()) for m in raw_tokens] == [_key(t['text']) for t in doc['tokens']]:
        return raw, [(m.start(), m.end()) for m in raw_tokens], 'Исходный текст'
    return doc['text'], normal, 'Нормализованный текст · исходник отличается, подсветка сохранена по проверенным смещениям'


def review_payload(data: dict, *, method: str = 'itermax', raw_texts: dict | None = None,
                   block_id: int | None = None, focus_block: int | None = None) -> dict:
    validate_review(data)
    if method not in METHODS:
        raise ValueError('Неизвестный метод')
    if block_id is not None and not 0 <= block_id < len(data['blocks']):
        raise ValueError('Фрагмент вне диапазона')
    result = {'panes': {}, 'method': method, 'focus_block': focus_block,
              'identity': f"{data['source'].get('fingerprint')}:{data['target'].get('fingerprint')}:{method}:{block_id}:{focus_block}"}
    for side in ('source', 'target'):
        doc = data[side]
        text, spans, label = original_projection(doc, (raw_texts or {}).get(side))
        tokens = [{'id': token['token_id'], 'text': token['text'], 'start': a, 'end': b,
                   'partners': [], 'state': 'not_processed', 'block': None}
                  for token, (a, b) in zip(doc['tokens'], spans)]
        result['panes'][side] = {'text': text, 'tokens': tokens, 'label': label,
                                 'document_id': doc.get('document_id', side)}
    for block in data['blocks']:
        word = block['word_alignment']
        if word['status'] == 'computed' and method not in word['methods']:
            raise ValueError(f'В файле нет связей метода {method}')
        for side in ('source', 'target'):
            for token_id in block[side]['token_ids']:
                token = result['panes'][side]['tokens'][token_id]
                token['block'] = block['block_id']
                token['state'] = 'unlinked' if word['status'] == 'computed' else (
                    'sentence_unaligned' if word['status'] == 'not_applicable' else 'not_processed')
        for link in word['methods'].get(method, []):
            for side, other in (('source', 'target'), ('target', 'source')):
                token = result['panes'][side]['tokens'][link[f'{side}_token_id']]
                token['partners'].append(link[f'{other}_token_id'])
                token['state'] = 'linked'
    if block_id is not None:
        for pane in result['panes'].values():
            chosen = [t for t in pane['tokens'] if t['block'] == block_id]
            start = chosen[0]['start'] if chosen else 0
            end = chosen[-1]['end'] if chosen else 0
            pane['text'] = pane['text'][start:end]
            pane['tokens'] = [dict(t, start=t['start']-start, end=t['end']-start) for t in chosen]
    return result
