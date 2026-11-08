
"""Conservative cleaning of Russian and English literary text before segmentation.

Use ``clean_text(raw_text)`` exactly once, BEFORE extracting boundary offsets,
running RuBERT, or calling SentenceSegmenter. Paragraphs, dialogue lines,
quotation marks and dashes are significant and therefore preserved.

Only the standard library is required.
"""
from __future__ import annotations

import re
import unicodedata

# Numeric references such as [2], [12, 13], [3-5], [7–9].
# We deliberately do not remove bracketed words, e.g. [sic].
_REFERENCE_BODY = r"\d{1,4}(?:\s*[,;\-–—]\s*\d{1,4})*"
_NUMERIC_REFERENCE = re.compile(r"\[\s*" + _REFERENCE_BODY + r"\s*\]")
_REFERENCE_LINE = re.compile(
    r"(?m)^[ \t]*\[\s*" + _REFERENCE_BODY + r"\s*\][ \t]*(?:\n|$)"
)
_REPEATED_SPACES = re.compile(r" {2,}")
_EDGE_SPACES = re.compile(r"(?m)^ +| +$")
_REPEATED_BLANK_LINES = re.compile(r"\n{3,}")

# PDF text extraction occasionally produces typographic Unicode ligatures.
_LIGATURES = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
})


def clean_text(
    text: str,
    *,
    remove_numeric_references: bool = True,
    normalize_backticks: bool = True,
) -> str:
    """Clean Unicode and formatting artifacts without changing literary structure.

    Operations:
    - normalize CRLF/CR, Unicode line separators, and PDF form feeds;
    - remove BOM, zero-width/control formatting chars and soft hyphens;
    - map NBSP, narrow NBSP, thin/em spaces, tabs to ordinary spaces;
    - normalize Latin ligatures and use Unicode NFC normalization;
    - remove only numeric square-bracket citations (if enabled);
    - normalize a backtick used as an English opening quote to a single quote;
    - collapse repeated spaces and excessive blank lines, remove indentation.

    The function preserves dashes (— and –), punctuation, straight/curly quotes,
    apostrophes in contractions, and one/two line breaks. It does NOT try to
    infer PDF hard-wrapped paragraphs, strip page headings or remove footnote
    *bodies*, since these operations need document-specific rules.

    Offsets change! Always pass the returned text to BoundaryDatasetBuilder
    and SentenceSegmenter; do not use positions extracted from raw ``text``.
    """
    if not isinstance(text, str):
        raise TypeError(f"clean_text expects str, got {type(text).__name__}")
    if not text:
        return ""

    # Do not apply NFKC globally: it can rewrite meaningful literary symbols.
    text = unicodedata.normalize("NFC", text).translate(_LIGATURES)
    text = (text.replace("\r\n", "\n").replace("\r", "\n")
                .replace("\u0085", "\n").replace("\u2028", "\n")
                .replace("\u2029", "\n").replace("\f", "\n"))
    if normalize_backticks:
        # English editions sometimes use: `Beer'll be delivered,' he said.
        # The existing quote detector recognizes apostrophes, not backticks.
        text = text.replace("`", "'")

    chars: list[str] = []
    for ch in text:
        if ch == "\n":
            chars.append(ch)
        elif ch == "\t" or unicodedata.category(ch) == "Zs":
            chars.append(" ")
        elif unicodedata.category(ch) in {"Cc", "Cf"}:
            # Includes: null bytes, BOM, zero-width space/joiner, soft hyphen,
            # word joiner, and bidirectional display control characters.
            continue
        else:
            chars.append(ch)
    result = "".join(chars)

    if remove_numeric_references:
        # Delete a numeric-only line entirely, without introducing an extra
        # paragraph break in its place.
        result = _REFERENCE_LINE.sub("", result)
        result = _NUMERIC_REFERENCE.sub("", result)

    result = _REPEATED_SPACES.sub(" ", result)
    result = _EDGE_SPACES.sub("", result)
    result = _REPEATED_BLANK_LINES.sub("\n\n", result)
    return result.strip(" \n")


__all__ = ["clean_text"]

