"""Overlap, embeddings, sentence/word alignment and corpus stage adapters."""
from __future__ import annotations
from pathlib import Path
from src.processing.pipeline.config import DOCUMENTS, SOURCE, TARGETS, folder
from src.processing.pipeline.io import check_file, save_json
from src.external_tools.vecalign_tools import DEFAULT_DELETION_PERCENTILE, validate_deletion_percentile

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


def stage_align(root: Path, docs: list[str], max_size: int,
                del_percentile_frac: float = DEFAULT_DELETION_PERCENTILE) -> None:
    del_percentile_frac = validate_deletion_percentile(del_percentile_frac)
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
            "del_percentile_frac": del_percentile_frac,
        }
        tools.align(**args)
        print(f"[Vecalign] {SOURCE} -> {target}; max_size={max_size}; del_percentile_frac={del_percentile_frac}: {dest.relative_to(root)}", flush=True)


def stage_word_align(root: Path, docs: list[str], max_size: int) -> None:
    from src.processing.alignment.io import align_word_files, configured_word_aligner
    if SOURCE not in docs or not any(doc in TARGETS for doc in docs):
        raise ValueError("Для выравнивания слов выберите оригинал и хотя бы один перевод")
    aligner = configured_word_aligner()
    for target in (doc for doc in TARGETS if doc in docs):
        align_word_files(
            check_file(folder(root, "normalized") / f"{SOURCE}.json"),
            check_file(folder(root, "normalized") / f"{target}.json"),
            check_file(folder(root, "aligned") / f"{SOURCE}_{target}.align"),
            folder(root, "word_alignment") / f"{SOURCE}_{target}.json", aligner=aligner)


def stage_corpus(root: Path, docs: list[str], max_size: int) -> None:
    if SOURCE not in docs:
        raise ValueError("Для создания корпуса нужен original-rus")
    try:
        from src.processing.alignment.corpus import CorpusBuilder
    except ImportError as error:
        raise RuntimeError("Не найден src.processing.alignment.corpus.CorpusBuilder") from error
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
