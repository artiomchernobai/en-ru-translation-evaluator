"""Shared candidate detection for inference and future annotation.

Character positions refer to the original, UNMODIFIED Python string.
A Russian candidate position points to the first character of its dash.
No candidate is automatically assigned SPLIT by this detector.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from src.processing.segmentation.english_dialogue import detect_english_dialogue_boundaries


@dataclass(frozen=True)
class BoundaryCandidate:
    position: int
    kind: str
    requires_model: bool
    rules: tuple[str, ...]


class BoundaryCandidateDetector:
    # Match closing quote after punctuation and before an English attribution.
    EN_CLOSE_ATTRIBUTION = re.compile(r"[.!?,…;]\s*['\"’”»]\s*,?\s*(?=[a-z])")
    # These opening dialogue quotes occur at the start of a physical line.
    EN_LINE_QUOTE = re.compile(r"(?m)^[^\S\r\n]*(?=['\"“‘`][^\S\r\n]*\S)")

    def __init__(self, language: str = "ru"):
        if language not in {"ru", "en"}:
            raise ValueError("language must be 'ru' or 'en'")
        self.language = language

    @staticmethod
    def _at_line_start(text: str, pos: int) -> bool:
        line_start = max(text.rfind("\n", 0, pos), text.rfind("\r", 0, pos)) + 1
        return not text[line_start:pos].strip()

    @staticmethod
    def _previous_nonspace(text: str, pos: int) -> str:
        while pos > 0 and text[pos - 1] in " \t\u00a0\u202f":
            pos -= 1
        return text[pos - 1] if pos else ""

    @staticmethod
    def _is_standalone_dash(text: str, pos: int) -> bool:
        """Do not classify numeric ranges and dash characters inside words."""
        left = text[pos - 1] if pos else ""
        right = text[pos + 1] if pos + 1 < len(text) else ""
        if left.isalnum() and right.isalnum():
            return False
        # A dash at the beginning of a line or surrounded by spaces/punctuation.
        if not left or left.isspace() or left in ',.!?…;:()[]{}«»“”"\'':
            return True
        if right.isspace() or right in '«»“”"\'':
            return True
        return False

    def detect(self, text: str) -> list[BoundaryCandidate]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self.language == "ru":
            return self._detect_ru(text)
        return self._detect_en(text)

    def _detect_ru(self, text: str) -> list[BoundaryCandidate]:
        results = []
        last_content = len(text.rstrip())
        for match in re.finditer(r"[—–]", text):
            pos = match.start()
            if not self._is_standalone_dash(text, pos):
                continue
            # There must be actual text after the dash.
            if pos + 1 >= last_content:
                continue
            if self._at_line_start(text, pos):
                rules = ("line_dash",)
                kind = "line_dash"
                needs_model = False
            else:
                prev = self._previous_nonspace(text, pos)
                rule = "punctuation_dash" if prev in ",.!?…;" else (
                    "colon_dash" if prev == ":" else "internal_dash"
                )
                rules = (rule,)
                kind = "internal_dash"
                needs_model = True
            if 0 < pos < len(text):
                results.append(BoundaryCandidate(pos, kind, needs_model, rules))
        return results

    def _detect_en(self, text: str) -> list[BoundaryCandidate]:
        events = detect_english_dialogue_boundaries(text)
        return [BoundaryCandidate(pos, "english_dialogue", False, tuple(sorted(rules)))
                for pos, rules in events.items()]
