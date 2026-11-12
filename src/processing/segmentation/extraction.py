"""Shared boundary records for Russian inference and English segmentation."""
from __future__ import annotations
import hashlib
from .candidates import BoundaryCandidateDetector

class BoundaryCandidateExtractor:
    FIELDS = (
        "id", "source", "language", "position", "left_context",
        "right_context", "rule", "source_sha256", "label",
    )

    def __init__(self, context_size: int = 200):
        if not isinstance(context_size, int) or context_size < 1:
            raise ValueError("context_size must be a positive integer")
        self.context_size = context_size

    @staticmethod
    def _display_context(value: str) -> str:
        return (value.replace("\r\n", "\n").replace("\r", "\n")
                .replace("\u00a0", "⍽").replace("\u202f", "⍽")
                .replace("\t", "⇥").replace("\n", "⏎"))

    def extract(self, text: str, language: str, source: str) -> list[dict]:
        detector = BoundaryCandidateDetector(language)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        candidates = detector.detect(text)
        result = []
        for c in candidates:
            result.append({
                "id": len(result), "source": source,
                "language": language, "position": c.position,
                "left_context": self._display_context(text[max(0, c.position - self.context_size):c.position]),
                "right_context": self._display_context(text[c.position:c.position + self.context_size]),
                "rule": ";".join(c.rules),
                "source_sha256": digest,
                "label": "",  # manual labeling only
            })
        return result
