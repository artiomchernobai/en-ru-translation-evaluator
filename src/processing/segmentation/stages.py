"""Sentence/dialogue segmentation stage adapter."""
from __future__ import annotations
from pathlib import Path
from src.processing.pipeline.config import DOCUMENTS, SOURCE, TARGETS, folder
from src.processing.pipeline.io import check_file, save_json

def stage_segment(root: Path, docs: list[str], max_size: int) -> None:
    try:
        from src.processing.segmentation.segmenter import SentenceSegmenter
        from src.processing.segmentation.postprocessor import SentencePostprocessor
    except ImportError as error:
        raise RuntimeError("Для сегментации нужны src.processing.segmentation.segmenter и postprocessor") from error
    segmenter = SentenceSegmenter()
    postprocessor = SentencePostprocessor()
    ru_processor = None
    for doc in docs:
        lang = DOCUMENTS[doc]
        src = check_file(folder(root, "cleaned") / f"{doc}.txt")
        clean = src.read_text(encoding="utf-8-sig")
        if lang == "ru":
            if ru_processor is None:
                try:
                    from src.processing.segmentation.ru_classifier import RuBoundaryClassifier
                    from src.processing.segmentation.boundary_aware import BoundaryAwareSegmenter
                    from src.processing.segmentation.extraction import BoundaryCandidateExtractor
                except ImportError as error:
                    raise RuntimeError("Для русского нужны RuBERT, BoundaryAwareSegmenter и BoundaryCandidateExtractor") from error
                model_dir = root / "models" / "ru_boundary_rubert_v4" / "best_model"
                if not model_dir.exists():
                    raise FileNotFoundError(f"Модель RuBERT не найдена: {model_dir}")
                classifier = RuBoundaryClassifier(
                    model_dir, device="auto", batch_size=8,
                    context_size=200, max_length=256, threshold=0.5,
                )
                ru_processor = BoundaryAwareSegmenter(
                    ru_classifier=classifier,
                    builder=BoundaryCandidateExtractor(context_size=200),
                    segmenter=segmenter,
                    postprocessor=postprocessor,
                )
            sentences, candidates, predictions = ru_processor.process(clean, lang, doc)
            audit = root / "data" / "training" / f"{doc}_boundary_predictions.csv"
            ru_processor.save_audit(audit, candidates, predictions)
            print(f"[RuBERT] {doc}: {len(candidates)} кандидатов, {len(predictions)} решений", flush=True)
        else:
            # English has deterministic quote rules; no RuBERT classifier.
            sentences = postprocessor.process(segmenter.segment(clean, lang))
        dest = folder(root, "segmented") / f"{doc}.json"
        save_json(dest, {
            "document": doc, "language": lang,
            "sentence_count": len(sentences), "sentences": sentences,
        })
        print(f"[Сегментация] {doc}: {len(sentences)} сегментов -> {dest.relative_to(root)}", flush=True)
