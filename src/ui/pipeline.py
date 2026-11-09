"""Bridge to the existing processing modules. No Streamlit dependency.

Each stage is deliberately callable alone. Missing project dependencies raise
clear errors rather than silently generating fake 'processed' output.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import DOCUMENTS, SOURCE, STAGES, TARGETS, folder


def check_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    return path


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_clean(root: Path, docs: list[str], max_size: int) -> None:
    from ..processing.text_normalizer import clean_text
    for doc in docs:
        src = check_file(folder(root, "raw") / f"{doc}.txt")
        target = folder(root, "cleaned") / f"{doc}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        text = clean_text(src.read_text(encoding="utf-8-sig"))
        target.write_text(text, encoding="utf-8")
        print(f"[Очистка] {doc}: {len(text)} символов -> {target.relative_to(root)}", flush=True)


def stage_segment(root: Path, docs: list[str], max_size: int) -> None:
    try:
        from src.processing.text_segmenter import SentenceSegmenter
        from src.processing.postprocessor import SentencePostprocessor
    except ImportError as error:
        raise RuntimeError("Для сегментации нужны src.processing.text_segmenter и postprocessor") from error
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
                    from src.processing.ru_boundary_classifier import RuBoundaryClassifier
                    from src.processing.boundary_aware_segmenter import BoundaryAwareSegmenter
                    from src.processing.boundry_dataset_builder import BoundaryDatasetBuilder
                except ImportError as error:
                    raise RuntimeError("Для русского нужны RuBERT, BoundaryAwareSegmenter и BoundaryDatasetBuilder") from error
                model_dir = root / "models" / "ru_boundary_rubert_v4" / "best_model"
                if not model_dir.exists():
                    raise FileNotFoundError(f"Модель RuBERT не найдена: {model_dir}")
                classifier = RuBoundaryClassifier(
                    model_dir, device="auto", batch_size=8,
                    context_size=200, max_length=256, threshold=0.5,
                )
                ru_processor = BoundaryAwareSegmenter(
                    ru_classifier=classifier,
                    builder=BoundaryDatasetBuilder(context_size=200),
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
        _save_json(dest, {
            "document": doc, "language": lang,
            "sentence_count": len(sentences), "sentences": sentences,
        })
        print(f"[Сегментация] {doc}: {len(sentences)} сегментов -> {dest.relative_to(root)}", flush=True)


def stage_normalize(root: Path, docs: list[str], max_size: int) -> None:
    try:
        from src.processing.normalizer import TextNormalizer
        from src.processing.text_json_txt_exporter import TextJsonTxtExporter
    except ImportError as error:
        raise RuntimeError("Нужны normalizer.TextNormalizer и src.processing.text_json_txt_exporter") from error
    normalizer = TextNormalizer()
    exporter = TextJsonTxtExporter()
    for doc in docs:
        src = check_file(folder(root, "segmented") / f"{doc}.json")
        norm = folder(root, "normalized") / f"{doc}.json"
        txt = folder(root, "overlap_input") / f"{doc}.txt"
        norm.parent.mkdir(parents=True, exist_ok=True)
        txt.parent.mkdir(parents=True, exist_ok=True)
        normalizer.process_file(src, norm)
        exporter.export_to_txt_file(norm, txt)
        print(f"[Нормализация] {doc}: {norm.relative_to(root)}; {txt.relative_to(root)}", flush=True)


def _tools(root: Path):
    try:
        from src.external_tools.vecalign_tools import ExternalToolsRunner
    except ImportError as error:
        raise RuntimeError("Не найден src.external_tools.vecalign_tools.ExternalToolsRunner") from error
    return ExternalToolsRunner(project_dir=root)


def stage_overlap(root: Path, docs: list[str], max_size: int) -> None:
    tools = _tools(root)
    for doc in docs:
        source = check_file(folder(root, "overlap_input") / f"{doc}.txt")
        dest = folder(root, "overlap_output") / f"{doc}.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tools.generate_overlaps(input_path=source, output_path=dest, num_overlaps=max_size)
        print(f"[Overlap] {doc} -> {dest.relative_to(root)}", flush=True)


def stage_embed(root: Path, docs: list[str], max_size: int) -> None:
    tools = _tools(root)
    for doc in docs:
        source = check_file(folder(root, "overlap_output") / f"{doc}.txt")
        dest = folder(root, "embeddings") / f"{doc}.emb"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tools.generate_embeddings(input_path=source, output_path=dest, language=DOCUMENTS[doc])
        print(f"[LASER] {doc} -> {dest.relative_to(root)}", flush=True)


def stage_align(root: Path, docs: list[str], max_size: int) -> None:
    if SOURCE not in docs:
        raise ValueError("Для Vecalign необходимо выбрать original-rus")
    targets = [doc for doc in TARGETS if doc in docs]
    if not targets:
        raise ValueError("Для Vecalign выберите хотя бы один английский перевод")
    tools = _tools(root)
    for target in targets:
        dest = folder(root, "aligned") / f"{SOURCE}_{target}.align"
        dest.parent.mkdir(parents=True, exist_ok=True)
        args = {
            "source_path": check_file(folder(root, "overlap_input") / f"{SOURCE}.txt"),
            "target_path": check_file(folder(root, "overlap_input") / f"{target}.txt"),
            "source_overlap_path": check_file(folder(root, "overlap_output") / f"{SOURCE}.txt"),
            "target_overlap_path": check_file(folder(root, "overlap_output") / f"{target}.txt"),
            "source_embeddings_path": check_file(folder(root, "embeddings") / f"{SOURCE}.emb"),
            "target_embeddings_path": check_file(folder(root, "embeddings") / f"{target}.emb"),
            "output_path": dest,
            "alignment_max_size": max_size,
        }
        tools.align(**args)
        print(f"[Vecalign] {SOURCE} -> {target}; max_size={max_size}: {dest.relative_to(root)}", flush=True)


def stage_corpus(root: Path, docs: list[str], max_size: int) -> None:
    if SOURCE not in docs:
        raise ValueError("Для создания корпуса нужен original-rus")
    try:
        from src.processing.corpus_builder import CorpusBuilder
    except ImportError as error:
        raise RuntimeError("Не найден src.processing.corpus_builder.CorpusBuilder") from error
    builder = CorpusBuilder()
    targets = [d for d in TARGETS if d in docs]
    if not targets:
        raise ValueError("Выберите хотя бы один английский перевод")
    for target in targets:
        output = folder(root, "corpus") / f"{SOURCE}_{target}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        builder.build_corpus(
            source_json=check_file(folder(root, "normalized") / f"{SOURCE}.json"),
            target_json=check_file(folder(root, "normalized") / f"{target}.json"),
            alignment_path=check_file(folder(root, "aligned") / f"{SOURCE}_{target}.align"),
            output=output,
            source_document=SOURCE, target_document=target,
        )
        print(f"[Корпус] {SOURCE} + {target} -> {output.relative_to(root)}", flush=True)


FUNCTIONS = {
    "clean": stage_clean,
    "segment": stage_segment,
    "normalize": stage_normalize,
    "overlap": stage_overlap,
    "embed": stage_embed,
    "align": stage_align,
    "corpus": stage_corpus,
}


def run_stage(root: Path, stage: str, docs: list[str], max_size: int) -> None:
    if stage not in (*STAGES, "all"):
        raise ValueError(f"Неизвестный этап: {stage}")
    if not docs or any(doc not in DOCUMENTS for doc in docs):
        raise ValueError("Недопустимый список документов")
    if stage == "all" and (SOURCE not in docs or not any(d in docs for d in TARGETS)):
        raise ValueError("Полный пайплайн требует оригинал и хотя бы один перевод")
    execution = STAGES if stage == "all" else (stage,)
    for step in execution:
        print(f"\n{'=' * 20} {step.upper()} {'=' * 20}", flush=True)
        FUNCTIONS[step](root, docs, max_size)
    print("\nВсе выбранные этапы успешно завершены.", flush=True)
