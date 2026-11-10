"""File-independent word alignment of normalized documents and sentence groups.

All indices are zero-based; character ranges are half-open Unicode code-point
ranges in normalized text, never offsets into the pre-cleaning raw file.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from numbers import Integral
import re
from typing import Callable, Protocol
import unicodedata

from .sentences import validate_alignment

TOKEN_PATTERN = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)
METHODS = {"inter", "mwmf", "itermax"}


class BlockTooLongError(ValueError):
    """The entire block must be skipped rather than silently truncated."""


class AlignmentBackend(Protocol):
    @property
    def metadata(self) -> dict: ...

    def align_tokens(self, source: list[str], target: list[str]) -> Mapping: ...


def _fingerprint(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _document(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Normalized document must be a JSON object")
    for field in ("document", "language"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"Normalized document requires a nonempty {field}")
    records = data.get("sentences")
    if not isinstance(records, list):
        raise ValueError("Normalized document requires a sentences array")
    if "sentence_count" in data and (type(data['sentence_count']) is not int or data['sentence_count'] != len(records)):
        raise ValueError("sentence_count does not match the sentences array")
    sentences, tokens, seen = [], [], set()
    offset = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid sentence at index {index}")
        identity = record.get("sentence_id")
        if type(identity) not in (str, int) or identity == "" or (type(identity), identity) in seen:
            raise ValueError(f"Missing, invalid or duplicate sentence_id at index {index}")
        seen.add((type(identity), identity))
        text = record.get("normalized_text")
        if not isinstance(text, str) or not text.strip() or '\n' in text or '\r' in text:
            raise ValueError(f"Invalid normalized_text at sentence index {index}")
        ids = []
        for position, match in enumerate(TOKEN_PATTERN.finditer(text)):
            token_id = len(tokens)
            ids.append(token_id)
            tokens.append({
                "token_id": token_id, "sentence_index": index, "sentence_id": identity,
                "index_in_sentence": position, "text": match.group(),
                "sentence_char_start": match.start(), "sentence_char_end": match.end(),
                "document_char_start": offset + match.start(), "document_char_end": offset + match.end(),
                "is_punctuation": all(unicodedata.category(c).startswith('P') for c in match.group()),
                "aligned_token_ids": [], "alignment_status": "not_processed",
            })
        sentences.append({"index": index, "sentence_id": identity, "normalized_text": text,
                          "document_char_start": offset, "document_char_end": offset + len(text),
                          "token_ids": ids})
        offset += len(text) + 1
    text = '\n'.join(record['normalized_text'] for record in sentences)
    return {"document_id": data['document'], "language": data['language'], "text": text,
            "sentence_count": len(sentences), "token_count": len(tokens),
            "fingerprint": _fingerprint(data), "sentences": sentences, "tokens": tokens}


def _block_side(document: dict, indices: list[int]) -> dict:
    records = [document['sentences'][i] for i in indices]
    ids, spans, texts = [], [], []
    offset = 0
    for record in records:
        texts.append(record['normalized_text'])
        for token_id in record['token_ids']:
            token = document['tokens'][token_id]
            spans.append({"index": len(ids), "token_id": token_id,
                          "char_start": offset + token['sentence_char_start'],
                          "char_end": offset + token['sentence_char_end']})
            ids.append(token_id)
        offset += len(record['normalized_text']) + 1
    return {"sentence_indices": list(indices), "sentence_ids": [r['sentence_id'] for r in records],
            "text": ' '.join(texts), "token_ids": ids, "token_spans": spans}


def _links(raw, source_ids: list[int], target_ids: list[int]) -> list[dict]:
    pairs = set()
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid word alignment pair: {pair!r}")
        a, b = pair
        if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, Integral)
                or not isinstance(b, Integral) or not 0 <= a < len(source_ids) or not 0 <= b < len(target_ids)):
            raise ValueError(f"Word alignment indices outside token range: {pair!r}")
        pairs.add((int(a), int(b)))
    return [{"source_index": a, "target_index": b,
             "source_token_id": source_ids[a], "target_token_id": target_ids[b]} for a, b in sorted(pairs)]


class WordAligner:
    """Align arbitrary document pairs. No project names, paths or file writes.

    Backend indices refer to the supplied word/punctuation tokens, not BPEs.
    Missing links describe model output; they are not MQM omissions/additions.
    """
    def __init__(self, backend: AlignmentBackend | None = None, *, selected_method: str = "itermax"):
        if selected_method not in METHODS:
            raise ValueError(f"selected_method must be one of {sorted(METHODS)}")
        if backend is None:
            from .simalign_backend import SimAlignBackend
            backend = SimAlignBackend()
        self.backend = backend
        self.selected_method = selected_method

    def align(self, source: dict, target: dict, sentence_alignments: Sequence[dict], *,
              progress: Callable[[int, int], None] | None = None) -> dict:
        source, target = _document(source), _document(target)
        # Work on copies and validate full sentence coverage before any inference.
        groups = deepcopy(list(sentence_alignments))
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("Sentence alignment must be an object")
            for field in ('source_indices', 'target_indices'):
                if not isinstance(group.get(field), list) or any(type(i) is not int for i in group[field]):
                    raise ValueError(f"{field} must contain integer sentence positions")
            if type(group.get('cost')) not in (int, float):
                raise ValueError("Sentence alignment cost must be numeric")
        validate_alignment(groups, source['sentence_count'], target['sentence_count'])
        blocks = []
        for block_id, group in enumerate(groups):
            sides = [_block_side(doc, group[field]) for doc, field in
                     ((source, 'source_indices'), (target, 'target_indices'))]
            src_ids, tgt_ids = (side['token_ids'] for side in sides)
            status = 'aligned' if src_ids and tgt_ids else ('source_only' if src_ids else 'target_only')
            word = {"status": "not_applicable", "methods": {}}
            if status == 'aligned':
                src_tokens = [source['tokens'][i]['text'] for i in src_ids]
                tgt_tokens = [target['tokens'][i]['text'] for i in tgt_ids]
                try:
                    raw = self.backend.align_tokens(src_tokens, tgt_tokens)
                except BlockTooLongError as error:
                    word.update(status='skipped_too_long', reason=str(error))
                else:
                    if not isinstance(raw, Mapping) or self.selected_method not in raw:
                        raise ValueError(f"Backend did not return {self.selected_method}")
                    word['methods'] = {name: _links(links, src_ids, tgt_ids) for name, links in raw.items()}
                    word['status'] = 'computed'
            links = word['methods'].get(self.selected_method, [])
            partners = ({i: [] for i in src_ids}, {i: [] for i in tgt_ids})
            for link in links:
                partners[0][link['source_token_id']].append(link['target_token_id'])
                partners[1][link['target_token_id']].append(link['source_token_id'])
            for side, doc, ids in zip(range(2), (source, target), (src_ids, tgt_ids)):
                for token_id in ids:
                    token = doc['tokens'][token_id]
                    token['aligned_token_ids'] = sorted(partners[side][token_id])
                    token['alignment_status'] = ('linked' if token['aligned_token_ids'] else 'unlinked') if word['status'] == 'computed' else (
                        'sentence_unaligned' if status != 'aligned' else 'not_processed')
            for label, ids, mapping in (('source', src_ids, partners[0]), ('target', tgt_ids, partners[1])):
                word[f'{label}_unlinked_token_ids'] = [i for i in ids if not mapping[i]] if word['status'] == 'computed' else []
            blocks.append({"block_id": block_id, "sentence_alignment_status": status,
                           "sentence_alignment_review": "unreviewed", "vecalign_cost": group['cost'],
                           "source": sides[0], "target": sides[1], "word_alignment": word})
            if progress is not None:
                progress(block_id + 1, len(groups))
        skipped = sum(b['word_alignment']['status'] == 'skipped_too_long' for b in blocks)
        return {"schema_version": "1.0", "processing_status": "partial" if skipped else "complete", "offsets": {"unit": "unicode_code_point", "end": "exclusive",
                "document_text": "normalized sentences joined with newline", "block_text": "normalized sentences joined with space"},
                "tokenization": {"name": "unicode_words_and_punctuation_v1", "pattern": TOKEN_PATTERN.pattern},
                "backend": deepcopy(self.backend.metadata), "selected_method": self.selected_method,
                "sentence_alignment_fingerprint": _fingerprint(groups), "source": source, "target": target,
                "summary": {"block_count": len(blocks), "computed_blocks": sum(b['word_alignment']['status'] == 'computed' for b in blocks),
                            "sentence_gap_blocks": sum(b['sentence_alignment_status'] != 'aligned' for b in blocks),
                            "skipped_too_long_blocks": skipped,
                            "selected_link_count": sum(len(b['word_alignment']['methods'].get(self.selected_method, [])) for b in blocks)},
                "blocks": blocks}
