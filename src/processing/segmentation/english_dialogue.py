
"""English literary dialogue transitions, using source-character offsets.

Single responsibility: identify boundaries at changes of discourse role.
spaCy handles ordinary sentence boundaries. No text is rewritten here.
"""
from __future__ import annotations

import re

OPEN = "'‘\"“«"
CLOSE = "'’\"”»"
QUOTES = OPEN + "’”»"
SPACE = r"[ \t]*(?:\r?\n[ \t]*)?"
_WS_RE = re.compile(SPACE)
DASH = r"(?:--|—|–|(?<![\w-])-(?![\w-]))"
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

REPORT = re.compile(
    r"\b(?:said|says|saying|asked|answered|replied|inquired|enquired|"
    r"thought|whispered|muttered|exclaimed|cried|declared|remarked|"
    r"observed|objected|interrupted|affirmed|confirmed|retorted|"
    r"rejoined|spoke|speaking|began|added|adding|continued|shouted|hissed|"
    r"grunted|mumbled|assured|responded|repeated|wondered|"
    r"went\s+on|was\s+saying|were\s+saying)\b",
    re.I
)

ACTION = re.compile(
    r"\b(?:smil(?:ed|ing)|grinn(?:ed|ing)|laugh(?:ed|ing)|"
    r"frown(?:ed|ing)|turn(?:ed|ing)|look(?:ed|ing)|"
    r"glanc(?:ed|ing)|stare(?:d|ing)|narrow(?:ed|ing)|"
    r"nod(?:ded|ding)|blink(?:ed|ing)|shrugg(?:ed|ing)|"
    r"gesture(?:d|ing)|point(?:ed|ing))\b",
    re.I
)

LABEL = re.compile(
    r"\b(?:sign|labelled|labeled|title|titled|name|named|"
    r"book|novel|movie|play|slogan|word|printed|poster)\b",
    re.I
)

END_QUOTE = re.compile(
    r"(?P<punct>[.!?,;…])"
    + SPACE
    + r"(?P<close>['’\"”»])"
)

ASIDE = re.compile(
    r"(?P<close>['’\"”»])"
    + SPACE
    + r"(?P<dash1>" + DASH + r")"
    + r"(?P<aside>[\s\S]{2,380}?)"
    + r"(?P<dash2>" + DASH + r")"
    + SPACE
    + r"(?P<open>['‘\"“«])"
)

DASH_AFTER_QUOTE = re.compile(
    r"(?P<close>['’\"”»])"
    + SPACE
    + r"(?P<dash>" + DASH + r")"
)


def _apostrophe_inside_word(text: str, i: int) -> bool:
    """Ignore contractions; `labelled'Beer` is a quote, not contraction."""
    return (
        0 < i < len(text) - 1
        and text[i - 1].isalnum()
        and text[i + 1].isalnum()
        and not (
            text[i - 1].islower()
            and text[i + 1].isupper()
        )
    )


def _skip_space(text: str, pos: int) -> int:
    m = _WS_RE.match(text, pos)
    return m.end() if m else pos


def _is_label(text: str, opening: int) -> bool:
    """A title/sign inside a prose sentence is not automatically direct speech."""
    prefix = re.split(
        r"[.!?;\n]",
        text[max(0, opening - 110):opening]
    )[-1]

    return bool(
        LABEL.search(prefix)
        or (
            re.search(r"['\"’”][ \t]*[-—–]{1,2}", prefix)
            and LABEL.search(text[max(0, opening - 110):opening])
        )
    )


def _report_before(text: str, opening: int) -> bool:
    prefix = text[max(0, opening - 165):opening].rstrip()

    if not prefix or prefix[-1] not in ",:":
        return False

    clause = re.split(
        r"[.!?;:\n\r]",
        prefix[:-1]
    )[-1]

    # Named character labels:
    # "Berlioz: '...'" or "and Homeless: '...'".
    speaker = re.split(r"[,;]", clause)[-1].strip()

    named = bool(
        prefix[-1] == ":"
        and re.fullmatch(
            r"(?:and\s+)?[A-Z][a-z]+"
            r"(?:\s+[A-Z][a-z]+){0,2}",
            speaker
        )
    )

    return bool(
        named
        or REPORT.search(clause)
        or ACTION.search(clause)
        or re.search(r"\b(?:voice|words)\b", clause, re.I)
    )


def _spoken_quote(text: str, opening: int) -> bool:
    """Do not treat incidental inline citations as dialogue."""
    if _is_label(text, opening):
        return False

    before = text[max(0, opening - 200):opening].rstrip()

    if not before or before[-1] in ".!?…:—–-":
        return True

    if re.search(r"[.!?…]['’\"”»]$", before):
        return True

    return _report_before(text, opening)


def _author_after(text: str, pos: int) -> bool:
    if pos >= len(text) or not text[pos].isalpha():
        return False

    clause = re.split(
        r"[.!?;:\r\n\"‘’“”]",
        text[pos:pos + 175],
        maxsplit=1
    )[0]

    # Stop at the next opening quote
    # (but not at possessive apostrophes).
    next_quote = re.search(
        r"(?<![A-Za-z])['‘\"“«](?=\s|[A-Za-z])",
        clause
    )

    if next_quote:
        clause = clause[:next_quote.start()]

    words = WORD.findall(clause)

    if not (1 <= len(words) <= 26):
        return False

    if re.match(
        r"(?i)^(?:was|is)\s+the\s+(?:answer|reply)\b",
        clause
    ):
        return True

    if words[0].lower() in {
        "and", "but", "because", "although",
        "when", "while", "then"
    }:
        return bool(ACTION.search(" ".join(words[:9])))

    # The reporter must occur in the beginning
    # of the attribution, not in a distant sentence.
    if not (
        words[0].islower()
        or words[0][0].isupper()
    ):
        return False

    if REPORT.search(" ".join(words[:15])):
        return True

    # Narrative gesture in a dialogue sandwich:
    # 'speech,' here he turned, '...'
    return bool(
        ACTION.search(" ".join(words[:15]))
        and re.match(
            r"(?i)^(?:here|he|she|the|at)\b",
            clause
        )
        and re.search(
            r"[,;:]\s*['‘\"“«]",
            text[pos:pos + 200]
        )
    )


def _spans(text: str) -> list[tuple[int, int]]:
    """Best-effort quote pairs, only for spaCy quote ownership / nesting.

    Boundaries do NOT depend on having a complete pair.
    Different styles may close the same turn when
    the local typography is damaged.
    """
    pairs: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []

    family = {
        "'": "s",
        "‘": "s",
        "’": "s",
        '"': "d",
        "“": "d",
        "”": "d",
        "«": "a",
        "»": "a"
    }

    last_quote = 0

    for i, char in enumerate(text):
        if (
            char not in family
            or (
                char in "'’"
                and _apostrophe_inside_word(text, i)
            )
        ):
            continue

        if re.search(
            r"\n[ \t]*\n",
            text[last_quote:i]
        ):
            stack.clear()

        last_quote = i
        kind = family[char]

        before = text[i - 1] if i else ""
        after = text[i + 1] if i + 1 < len(text) else ""

        # A quoted title after a lower-case word
        # with omitted OCR space.
        opening = char in "‘“«" or (
            char in "'\""
            and (
                not before
                or before.isspace()
                or before in "([{:—–"
                or (
                    before.islower()
                    and after.isupper()
                )
            )
        )

        if char in "’”»":
            opening = False

        # Closing quote after punctuation
        # is unambiguous even when another
        # unmatched quote exists earlier.
        if (
            before
            and before in ".!?…,:;"
            and char in "'\""
        ):
            opening = False

        if (
            opening
            and not (
                stack
                and stack[-1][0] == kind
                and before in ".!?…,:;"
            )
        ):
            stack.append((kind, i))
        else:
            j = next(
                (
                    j
                    for j in range(len(stack) - 1, -1, -1)
                    if stack[j][0] == kind
                ),
                None
            )

            if j is not None:
                pairs.append((stack.pop(j)[1], i))
            elif opening:
                stack.append((kind, i))

    return pairs


def detect_english_dialogue_boundaries(
    text: str
) -> dict[int, set[str]]:
    """Detect only structural transitions; return absolute Unicode offsets.

    The four mutually complementary event types are:
      author -> speech;
      speech -> author;
      speech -> new speech;
      speech -> dash-enclosed aside -> speech.
    """
    found: dict[int, set[str]] = {}

    pairs = _spans(text)

    paired_opens = {a for a, _ in pairs}
    paired_closes = {b for _, b in pairs}
    close_to_open = {b: a for a, b in pairs}

    def add(pos: int, reason: str):
        if 0 < pos < len(text):
            found.setdefault(pos, set()).add(reason)

    # A) Opening quote after terminal punctuation,
    # structural dash, or an attribution ending
    # in comma/colon. Handle zero or more spaces.
    for i, ch in enumerate(text):
        if (
            ch not in OPEN
            or (
                ch == "'"
                and _apostrophe_inside_word(text, i)
            )
        ):
            continue

        if i in paired_closes:
            continue

        j = i - 1

        while j >= 0 and text[j].isspace():
            j -= 1

        if j < 0:
            continue

        previous = text[j]

        if previous in ".!?…":
            # Distinguish . 'new speech'
            # from end-of-speech .'
            # (also tolerate a missing space
            # after the period).
            immediately_after_quote = text[i + 1:i + 2]
            gap_before = text[j + 1:i]

            after_word = _skip_space(text, i + 1)
            word_after = text[after_word:after_word + 1]

            looks_open = (
                immediately_after_quote.isalpha()
                or (
                    immediately_after_quote.isspace()
                    and word_after.isupper()
                    and bool(gap_before)
                    and i not in paired_closes
                    and (
                        not re.search(
                            r"['’\"”»][^.!?]{0,90}$",
                            text[max(0, j - 95):j]
                        )
                        or REPORT.search(
                            text[max(0, j - 95):j]
                        )
                    )
                )
            )

            if looks_open:
                add(
                    i,
                    "english_new_quote_after_sentence"
                )

        elif previous in "—–-":
            if not _is_label(text, i):
                add(
                    i,
                    "english_quote_after_dash"
                )

        elif (
            previous in ",:"
            and _report_before(text, i)
        ):
            add(
                i,
                "english_author_to_speech"
            )

        elif (
            previous in CLOSE
            and j in paired_closes
        ):
            if not _is_label(text, i):
                add(
                    i,
                    "english_speech_to_next_speech"
                )

    # B) Closing quote, with or without preceding
    # punctuation: add a boundary only if
    # the next fragment is an attribution
    # or distinct sentence/quote.
    #
    # Even unmatched closing marks are accepted
    # after terminal punctuation.
    for i, ch in enumerate(text):
        if (
            ch not in CLOSE
            or (
                ch in "'’"
                and _apostrophe_inside_word(text, i)
            )
        ):
            continue

        if (
            i in found
            and "english_new_quote_after_sentence"
            in found[i]
        ):
            continue

        n = _skip_space(text, i + 1)

        if n >= len(text):
            continue

        prev = text[max(0, i - 100):i].rstrip()

        punctuation = bool(
            prev
            and prev[-1] in ",.!?;…"
        )

        is_close = (
            i in paired_closes
            or (
                punctuation
                and i not in paired_opens
            )
        )

        if not is_close:
            continue

        opening = close_to_open.get(i)

        if (
            opening is not None
            and not _spoken_quote(text, opening)
        ):
            continue

        if re.match(
            r",?\s+and\s+[A-Z][a-z]+"
            r"(?:\s+[A-Z][a-z]+){0,2}\s*:",
            text[i + 1:i + 100]
        ):
            add(
                i + 1,
                "english_speech_to_named_speaker"
            )

        elif (
            text[n] in OPEN
            and not (
                text[n] == "'"
                and _apostrophe_inside_word(text, n)
            )
        ):
            add(
                n,
                "english_speech_to_next_speech"
            )

        elif text[n] in "-—–":
            continue  # dash asides below

        elif _author_after(text, n):
            add(
                i + 1,
                "english_speech_to_author"
            )

        elif (
            punctuation
            and prev[-1] in ".!?…"
            and text[n].isupper()
        ):
            add(
                i + 1,
                "english_speech_to_narration"
            )

        elif (
            punctuation
            and re.match(
                r"(?i)(?:and|but)\b",
                text[n:]
            )
        ):
            if ACTION.search(text[n:n + 100]):
                add(
                    i + 1,
                    "english_speech_to_action"
                )

    # C) v4 dash asides:
    # 'speech' - narrator - 'speech'.
    #
    # Both surrounding dash tokens stay
    # attached to the author segment.
    for m in ASIDE.finditer(text):
        close = m.start("close")
        d1 = m.start("dash1")
        op = m.start("open")

        aside = m.group("aside")

        if re.search(
            r"\n\s*\n|[.!?]\s+[A-Z]",
            aside
        ):
            continue

        if not (
            REPORT.search(aside)
            or ACTION.search(aside)
        ):
            continue

        qstart = close_to_open.get(close)

        if (
            qstart is not None
            and _is_label(text, qstart)
        ):
            continue

        if (
            qstart is None
            and not re.search(
                r"[,.!?;…]\s*$",
                text[max(0, close - 12):close]
            )
        ):
            continue

        add(
            d1,
            "english_dash_aside"
        )

        add(
            op,
            "english_dash_resume"
        )

    # D) Speech followed by dash +
    # standalone reporting clause;
    # no resumption.
    for m in DASH_AFTER_QUOTE.finditer(text):
        close = m.start("close")
        dash = m.start("dash")

        n = _skip_space(
            text,
            m.end("dash")
        )

        if not _author_after(text, n):
            continue

        opening = close_to_open.get(close)

        if (
            opening is not None
            and not _spoken_quote(text, opening)
        ):
            continue

        if (
            opening is not None
            or re.search(
                r"[,.!?;…]\s*$",
                text[max(0, close - 12):close]
            )
        ):
            add(
                dash,
                "english_quote_dash_author"
            )

    return {
        pos: found[pos]
        for pos in sorted(found)
    }
