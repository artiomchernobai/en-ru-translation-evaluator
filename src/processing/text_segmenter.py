"""Boundary-aware multilingual sentence segmentation with absolute offsets.

Russian: RuBERT verdicts, validated by a punctuation safety guard; English:
local quote/attribution boundaries and spaCy.  Preserves all non-whitespace
characters, including quotation marks and dialogue dashes.
"""
from __future__ import annotations

import re
from src.processing.boundary_candidate_detector import BoundaryCandidateDetector
from src.processing.ru_boundary_classifier import dash_boundary_allowed


class SentenceSegmenter:
    PARAGRAPH_BREAK = re.compile(r"\r?\n[^\S\r\n]*\r?\n(?:[^\S\r\n]*\r?\n)*")
    SINGLE_LINE_BREAK = re.compile(r"\r?\n")

    def __init__(self, models: dict | None = None, *,
                 line_breaks_are_paragraphs: bool = False):
        # Default: a single newline may be a PDF line wrap. True is an opt-in
        # for guaranteed paragraph-per-line corpora.  Explicit dialogue
        # candidates still split regardless of this switch.
        self.line_breaks_are_paragraphs = line_breaks_are_paragraphs
        self.models = dict(models or {})
        self.detectors = {
            "ru": BoundaryCandidateDetector("ru"),
            "en": BoundaryCandidateDetector("en"),
        }

    def _get_nlp(self, language: str):
        if language not in self.detectors:
            raise ValueError(f"Unsupported language: {language}")
        if language not in self.models:
            import spacy
            self.models[language] = spacy.load(
                "ru_core_news_sm" if language == "ru" else "en_core_web_sm"
            )
        return self.models[language]

    def _spacy_starts(self, text: str, nlp) -> set[int]:
        """Find starts inside independent paragraphs; keep original offsets."""
        boundaries: set[int] = set()
        blocks = []
        start = 0
        separator = (self.SINGLE_LINE_BREAK if self.line_breaks_are_paragraphs
                     else self.PARAGRAPH_BREAK)
        for match in separator.finditer(text):
            blocks.append((start, match.start()))
            boundaries.add(match.end())
            start = match.end()
        blocks.append((start, len(text)))
        for offset, end in blocks:
            chunk = text[offset:end]
            if not chunk.strip():
                continue
            doc = nlp(chunk)
            for i, sentence in enumerate(doc.sents):
                if i:
                    boundaries.add(offset + sentence.start_char)
        if not self.line_breaks_are_paragraphs:
            # spaCy sometimes interprets PDF soft line breaks as sentence ends,
            # even without terminal punctuation.  Reject only boundaries
            # immediately following an unfinished physical line.
            filtered = set()
            for pos in boundaries:
                # Inspect only the immediate whitespace preceding this start;
                # slicing text[:pos] for every sentence is quadratic on novels.
                previous_end = pos
                while previous_end > 0 and text[previous_end - 1].isspace():
                    previous_end -= 1
                between = text[previous_end:pos]
                if ("\n" in between and not re.search(r"\n[^\S\r\n]*\n", between) and previous_end
                        and text[previous_end - 1] not in ".!?…»”\"'"):
                    continue
                filtered.add(pos)
            boundaries = filtered
        return boundaries

    @staticmethod
    def _english_quote_start(
        text: str,
        position: int,
        opening_quotes: set[int] | None = None,
        closing_quotes: set[int] | None = None,
    ) -> int:
        """Attach a *genuine opening quote* to its sentence, never a closer.

        The offset returned by spaCy may point at the first word after an
        opening quote or at the closing quote of the preceding sentence.
        Quote roles come from _spans(text), computed once in segment().
        Unpaired quotes use a conservative local fallback.
        """
        if position <= 0 or position >= len(text):
            return position
        opening_quotes = opening_quotes or set()
        closing_quotes = closing_quotes or set()

        # spaCy occasionally starts a new sentence *at* the closing quote.
        # Keep that symbol with the preceding segment instead of separating it.
        if position in closing_quotes:
            return position + 1

        # An adjacent quote, or one separated from the first word by spaces
        # and/or one PDF line wrap.  Do not scan further back than that.
        lookbehind_start = max(0, position - 48)
        prefix = text[lookbehind_start:position]
        match = re.search(r"(['\"‘’“”«»])([ \t]*(?:\r?\n[ \t]*)?)$", prefix)
        if match is None:
            return position
        quote_pos = lookbehind_start + match.start(1)
        if quote_pos in closing_quotes:
            return position
        if quote_pos in opening_quotes:
            return quote_pos

        quote = text[quote_pos]
        if quote in "’”»":
            return position
        if quote in "‘“«":
            return quote_pos

        # When pair recovery failed, a straight quote directly after a word
        # or terminal punctuation is normally a closing delimiter:
        #   'Hello.' The ...         (never move the boundary onto ').
        # This still allows: He stopped. ' Hello ... (opening quote).
        immediate_before = text[quote_pos - 1] if quote_pos else ""
        if immediate_before and not immediate_before.isspace():
            # OCR often omits the gap before an opening title: labelled'Beer.
            # The quote parser recognises most of these; do not infer a new
            # sentence start from an unpaired title quote.
            return position
        previous = text[max(0, quote_pos - 48):quote_pos].rstrip()
        if previous and previous[-1] in ",:;":
            return quote_pos
        if not previous or previous[-1] in ".!?…\n":
            return quote_pos
        return position

    @staticmethod
    def _near_kept_dash(text: str, boundary: int, dash_pos: int) -> bool:
        """Veto spaCy boundaries immediately before/after a kept dash."""
        if abs(boundary - dash_pos) > 12:
            return False
        low, high = sorted((boundary, dash_pos))
        between = text[low:high]
        return not between.strip(" \t\r\n\u00a0\u202f,.!?…;:»”\"' ") or (
            boundary > dash_pos and not text[dash_pos + 1:boundary].strip()
        )

    def segment(self, text: str, language: str, *, candidates: list[dict] | None = None,
                verdicts: dict[int, int] | None = None) -> list[dict]:
        """Return JSON-compatible list of {sentence_id, text}.

        Candidates must be extracted from exactly the same `text` input;
        verdicts are offset -> 0/1.  No model is used for English.
        """
        if language not in self.detectors:
            raise ValueError(f"Unsupported language: {language}")
        if not text or not text.strip():
            return []
        nlp = self._get_nlp(language)
        base = self._spacy_starts(text, nlp)
        if language == "en":
            from src.processing.english_dialogue_boundaries import _spans
            spans = _spans(text)
            opening_quotes = {left for left, _ in spans}
            closing_quotes = {right for _, right in spans}
            base = {
                self._english_quote_start(text, pos, opening_quotes, closing_quotes)
                for pos in base
            }
            # Nested quotations must not create sentence starts in the
            # middle of an outer spoken turn. O(Q log Q), not O(Q^2).
            active_ends = []
            nested_veto = set()
            for left, right in sorted(spans):
                while active_ends and left >= active_ends[-1]:
                    active_ends.pop()
                if active_ends and right < active_ends[-1]:
                    k = right + 1
                    while k < len(text) and k <= right + 5 and text[k].isspace():
                        k += 1
                    if k <= right + 5 and text[k:k + 1].islower():
                        nested_veto.add(k)
                active_ends.append(right)
            base.difference_update(nested_veto)

        if candidates is None:
            from .boundry_dataset_builder import BoundaryDatasetBuilder
            candidates = BoundaryDatasetBuilder().extract(text, language, "inline")
        if verdicts is None:
            verdicts = {}
        mandatory: set[int] = set()
        for row in candidates:
            pos = int(row["position"])
            rules = str(row["rule"]).split(";")
            if language == "ru":
                if "line_dash" not in rules and pos not in verdicts:
                    raise ValueError(f"Missing RuBERT verdict for dash at offset {pos}")
                label = verdicts.get(pos, 1 if "line_dash" in rules else 0)
                if label not in (0, 1):
                    raise ValueError(f"Invalid boundary label at {pos}: {label}")
                # The guard also protects mock classifiers, imported CSV
                # predictions and callers that bypass RuBoundaryClassifier.
                allowed = dash_boundary_allowed(text, pos)
                if label == 1 and allowed:
                    mandatory.add(pos)
                else:
                    base = {p for p in base if not self._near_kept_dash(text, p, pos)}
            else:
                mandatory.add(pos)

        if language == "en":
            # A spaCy sentence start immediately AFTER a dash is usually an
            # artifact of punctuation inside the preceding quotation:
            # "The play 'Hello!' - he smiled - 'World!'".
            # If the dash is a real dialogue transition, the quote detector
            # places the boundary AT the dash instead.
            base = {p for p in base if not (
                text[p:p + 1].islower()
                and re.search(
                    r"(?:--|—|–|(?<![\w-])-(?!-))[ \t\r\n]*$",
                    text[max(0, p - 32):p],
                )
            )}

        if language == "en":
            # spaCy can treat punctuation in an inline title as a full stop:
            # She wrote 'Hello!' and smiled.  The quote parser knows this
            # is not a dialogue, so veto the corresponding spaCy boundary.
            from src.processing.english_dialogue_boundaries import _spoken_quote
            non_speech_closes = {
                end for opening, end in spans if not _spoken_quote(text, opening)
            }
            corrected = set()
            for p in base:
                # When a strong dialogue boundary occurs at an opening quote,
                # remove a weaker spaCy start immediately after that opener.
                window_start = max(0, p - 32)
                near = re.search(r"(['\"‘“«])[ \t]*(?:\r?\n[ \t]*)?$",
                                 text[window_start:p])
                if near and window_start + near.start(1) in mandatory:
                    continue
                k = p
                while k > 0 and text[k - 1].isspace():
                    k -= 1
                if (k > 0 and k - 1 in non_speech_closes
                        and text[p:p + 1].islower() and p not in mandatory):
                    continue
                corrected.add(p)
            base = corrected

        # Explicit candidate evidence wins over spaCy except when a Russian
        # dash was vetoed above.  Delimit only at true Unicode positions.
        positions = sorted(p for p in base | mandatory if 0 < p < len(text))
        records: list[dict] = []
        start = 0
        for stop in positions + [len(text)]:
            fragment = text[start:stop].strip()
            if fragment and not self.line_breaks_are_paragraphs:
                # PDF line wraps aren't meaningful inside a final segment.
                # Join "А-\nа!" as "А-а!", not "А- а!".
                fragment = re.sub(
                    r"(?<=[A-Za-zА-Яа-яЁё]-)[ \t]*\r?\n[ \t]*(?=[a-zа-яё])",
                    "", fragment,
                )
                fragment = re.sub(r"[ \t]*\r?\n[ \t]*", " ", fragment)
            if fragment:
                records.append({"sentence_id": len(records), "text": fragment})
            start = stop
        return records
