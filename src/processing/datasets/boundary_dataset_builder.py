"""Build unlabeled boundary candidates with the SHARED inference detector.

Existing labeled datasets remain untouched. This version intentionally expands
Russian detection to standalone internal dashes. No automatic labels are set.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import re
from src.processing.segmentation.candidates import BoundaryCandidateDetector


class BoundaryDatasetBuilder:
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

    @classmethod
    def validate_csv(cls, output_path, expected_rows=None, require_order=False) -> int:
        """Validates SHUFFLED labeled CSV, too. Strict order is opt-in."""
        with Path(output_path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(cls.FIELDS):
                raise ValueError("Invalid CSV columns")
            rows = list(reader)
        if expected_rows is not None and len(rows) != len(expected_rows):
            raise ValueError("Unexpected number of CSV records")
        ids, locations, last_positions = set(), set(), {}
        for i, row in enumerate(rows):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Invalid fields in CSV row {i+2}")
            try:
                ident, pos = int(row["id"]), int(row["position"])
            except ValueError as exc:
                raise ValueError(f"Invalid id/position at row {i+2}") from exc
            if ident < 0 or pos <= 0 or ident in ids:
                raise ValueError(f"Duplicate or invalid id/position at row {i+2}")
            ids.add(ident)
            key = (row["source"], row["source_sha256"], pos)
            if key in locations:
                raise ValueError(f"Duplicate candidate {key}")
            locations.add(key)
            if row["label"] not in {"", "0", "1"}:
                raise ValueError(f"Invalid label at row {i+2}")
            if not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]):
                raise ValueError(f"Invalid source SHA-256 at row {i+2}")
            if require_order:
                if ident != i or pos <= last_positions.get(row["source"], -1):
                    raise ValueError(f"Unordered ID/position at row {i+2}")
                last_positions[row["source"]] = pos
            if expected_rows is not None:
                expected = {key: str(value) for key, value in expected_rows[i].items()}
                if row != expected:
                    raise ValueError(f"CSV differs from generated row {i+2}")
        return len(rows)

    def build(self, sources: dict, output_path) -> list[dict]:
        """sources = { 'chapter_01': {'path':'...', 'language':'ru'}, ... }"""
        all_rows = []
        for name, config in sources.items():
            text = Path(config["path"]).read_text(encoding="utf-8-sig")
            extracted = self.extract(text, config["language"], name)
            for record in extracted:
                record["id"] = len(all_rows)
                all_rows.append(record)
            print(f"{name}: {len(extracted)} candidates")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        self.validate_csv(output, all_rows, require_order=True)
        return all_rows
