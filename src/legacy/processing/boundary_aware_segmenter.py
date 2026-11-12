"""Glue between BoundaryDatasetBuilder, RuBERT, spaCy and the postprocessor."""
from __future__ import annotations

import csv
from pathlib import Path

from src.processing.boundry_dataset_builder import BoundaryDatasetBuilder
from src.processing.text_segmenter import SentenceSegmenter
from src.processing.ru_boundary_classifier import RuBoundaryClassifier, BoundaryVerdict


class BoundaryAwareSegmenter:
    def __init__(self, ru_classifier: RuBoundaryClassifier, *,
                 builder: BoundaryDatasetBuilder | None = None,
                 segmenter: SentenceSegmenter | None = None,
                 postprocessor=None):
        self.ru_classifier = ru_classifier
        self.builder = builder or BoundaryDatasetBuilder(context_size=ru_classifier.context_size)
        self.segmenter = segmenter or SentenceSegmenter()
        if postprocessor is None:
            from src.processing.postprocessor import SentencePostprocessor
            postprocessor = SentencePostprocessor()
        self.postprocessor = postprocessor

    def process(self, text: str, language: str, document: str):
        """Return (final_sentences, builder_candidates, model_verdicts).

        Run *after* clean_text(): all offsets and SHA-256 refer to `text`.
        The English branch never invokes RuBERT.
        """
        candidates = self.builder.extract(text, language, document)
        predictions: list[BoundaryVerdict] = []
        if language == "ru":
            predictions = self.ru_classifier.predict_candidates(text, candidates)
            decisions = {v.position: v.label for v in predictions}
        else:
            decisions = {}
        records = self.segmenter.segment(text, language, candidates=candidates, verdicts=decisions)
        records = self.postprocessor.process(records)
        return records, candidates, predictions

    @staticmethod
    def save_audit(path, candidates: list[dict], predictions: list[BoundaryVerdict]):
        """Optional human-readable file; NOT the data used for segmentation."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        verdicts = {v.position: v for v in predictions}
        fields = list(BoundaryDatasetBuilder.FIELDS) + [
            "prediction", "decision", "prob_split", "origin"
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in candidates:
                row = dict(item)
                verdict = verdicts.get(int(item["position"]))
                row.update({
                    "prediction": "" if verdict is None else str(verdict.label),
                    "decision": "" if verdict is None else verdict.decision,
                    "prob_split": "" if verdict is None else f"{verdict.probability_split:.6f}",
                    "origin": "" if verdict is None else verdict.origin,
                })
                writer.writerow(row)
        return path
