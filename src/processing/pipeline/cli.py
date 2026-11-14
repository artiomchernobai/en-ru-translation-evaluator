
import argparse
import json
from pathlib import Path

from src.external_tools.vecalign_tools import ExternalToolsRunner
from src.processing.preprocessing.export import TextJsonTxtExporter
from src.processing.alignment.corpus import CorpusBuilder
from src.processing.segmentation.segmenter import SentenceSegmenter
from src.processing.datasets.boundary_dataset_builder import BoundaryDatasetBuilder
from src.processing.segmentation.extraction import BoundaryCandidateExtractor

from src.processing.segmentation.postprocessor import SentencePostprocessor
from src.processing.preprocessing.normalization import TextNormalizer
from src.processing.preprocessing.cleaning import clean_text
from src.processing.segmentation.boundary_aware import BoundaryAwareSegmenter
from src.processing.segmentation.ru_classifier import RuBoundaryClassifier


from .config import (BASE_DIR, DOCUMENTS, TARGET_DOCUMENTS, RAW_DIR, OUTPUT_DIR,
    NORMALIZED_DIR, TRAINING_DIR, OVERLAP_INPUT_DIR, OVERLAP_OUTPUT_DIR, EMBEDDINGS_DIR,
    ALIGNED_DIR, CORPUS_DIR, WORD_ALIGNMENT_DIR)


def create_directories():
    directories = [
        OUTPUT_DIR,
        NORMALIZED_DIR,
        OVERLAP_INPUT_DIR,
        OVERLAP_OUTPUT_DIR,
        EMBEDDINGS_DIR,
        ALIGNED_DIR,
        CORPUS_DIR,
        WORD_ALIGNMENT_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def process_document(
    name,
    language,
    segmenter,
    postprocessor,
    normalizer,
    text_exporter,
    boundary_segmenter
):
    input_path = RAW_DIR / f"{name}.txt"

    segmented_path = OUTPUT_DIR / f"{name}.json"
    normalized_path = NORMALIZED_DIR / f"{name}.json"
    overlap_input_path = OVERLAP_INPUT_DIR / f"{name}.txt"

    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    # 1. Clean raw text
    cleaned_text = clean_text(text)
    sentences, candidates, predictions = boundary_segmenter.process(
        cleaned_text, language, name
    )
    boundary_segmenter.save_audit(
        TRAINING_DIR / f"{name}_boundary_predictions.csv",
        candidates, predictions
    )

    result = {
        "document": name,
        "language": language,
        "sentence_count": len(sentences),
        "sentences": sentences
    }

    # 4. Save segmented data
    with open(segmented_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    # 5. Normalize sentences
    normalizer.process_file(
        segmented_path,
        normalized_path
    )

    # 6. Export normalized text
    text_exporter.export_to_txt_file(
        normalized_path,
        overlap_input_path
    )

    print(f"{name}: {len(sentences)} sentences")


def generate_overlaps(tools):
    for name in DOCUMENTS:
        tools.generate_overlaps(
            input_path=OVERLAP_INPUT_DIR / f"{name}.txt",
            output_path=OVERLAP_OUTPUT_DIR / f"{name}.txt"
        )


def generate_embeddings(tools):
    for name, language in DOCUMENTS.items():
        tools.generate_embeddings(
            input_path=OVERLAP_OUTPUT_DIR / f"{name}.txt",
            output_path=EMBEDDINGS_DIR / f"{name}.emb",
            language=language
        )


def align_documents(tools):
    source_document = "original-rus"

    for target_document in TARGET_DOCUMENTS:
        tools.align(
            source_path=OVERLAP_INPUT_DIR / f"{source_document}.txt",
            target_path=OVERLAP_INPUT_DIR / f"{target_document}.txt",

            source_overlap_path=OVERLAP_OUTPUT_DIR / f"{source_document}.txt",
            target_overlap_path=OVERLAP_OUTPUT_DIR / f"{target_document}.txt",

            source_embeddings_path=EMBEDDINGS_DIR / f"{source_document}.emb",
            target_embeddings_path=EMBEDDINGS_DIR / f"{target_document}.emb",

            output_path=ALIGNED_DIR / f"{source_document}_{target_document}.align",

            alignment_max_size=3
        )


def build_corpora(corpus_builder):
    source_document = "original-rus"

    for target_document in TARGET_DOCUMENTS:
        corpus_builder.build_corpus(
            source_json=NORMALIZED_DIR / f"{source_document}.json",
            target_json=NORMALIZED_DIR / f"{target_document}.json",

            alignment_path=ALIGNED_DIR / f"{source_document}_{target_document}.align",

            output=CORPUS_DIR / f"{source_document}_{target_document}.json",

            source_document=source_document,
            target_document=target_document
        )


def build_word_alignments():
    from src.processing.alignment.io import align_word_files, configured_word_aligner
    aligner = configured_word_aligner()
    source_document = "original-rus"
    for target_document in TARGET_DOCUMENTS:
        align_word_files(
            NORMALIZED_DIR / f"{source_document}.json",
            NORMALIZED_DIR / f"{target_document}.json",
            ALIGNED_DIR / f"{source_document}_{target_document}.align",
            WORD_ALIGNMENT_DIR / f"{source_document}_{target_document}.json", aligner=aligner)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Text processing pipeline")
    parser.add_argument('--stage', choices=('all', 'word_align', 'cometkiwi', 'candidates'), default='all',
                        help='Run the full pipeline, word alignment, COMETKiwi, or review-candidate screening')
    args = parser.parse_args(argv)
    if args.stage == 'candidates':
        from src.processing.analysis.stages import stage_candidates
        stage_candidates(BASE_DIR, list(DOCUMENTS), 3)
        return
    if args.stage == 'cometkiwi':
        from src.processing.evaluation.stages import stage_cometkiwi
        stage_cometkiwi(BASE_DIR, list(DOCUMENTS), 3)
        return
    create_directories()
    if args.stage == 'word_align':
        build_word_alignments()
        return

    segmenter = SentenceSegmenter()
    postprocessor = SentencePostprocessor()
    normalizer = TextNormalizer()
    text_exporter = TextJsonTxtExporter()

    classifier = RuBoundaryClassifier(
        BASE_DIR / "models" / "ru_boundary_rubert_v4" / "best_model",
        device="auto", batch_size=8, context_size=200, max_length=256,
    )
    boundary_segmenter = BoundaryAwareSegmenter(
        ru_classifier=classifier,
        builder=BoundaryCandidateExtractor(context_size=200),
        segmenter=SentenceSegmenter(),
        postprocessor=SentencePostprocessor(),
    )

    tools = ExternalToolsRunner(project_dir=BASE_DIR)
    corpus_builder = CorpusBuilder()

    # 1. Preprocessing, segmentation and normalization
    for name, language in DOCUMENTS.items():
        process_document(
            name,
            language,
            segmenter,
            postprocessor,
            normalizer,
            text_exporter,
            boundary_segmenter
        )

    # 2. Generate overlaps
    generate_overlaps(tools)

    # 3. Generate LASER embeddings
    generate_embeddings(tools)

    # 4. Align documents using Vecalign
    align_documents(tools)

    # 5. Align words in the sentence groups
    build_word_alignments()

    from src.processing.evaluation.stages import stage_cometkiwi
    stage_cometkiwi(BASE_DIR, list(DOCUMENTS), 3)

    from src.processing.analysis.stages import stage_candidates
    stage_candidates(BASE_DIR, list(DOCUMENTS), 3)

    # 6. Build parallel corpora
    build_corpora(corpus_builder)

    builder = BoundaryDatasetBuilder()

    builder.build(
        sources={
            "original": {
                "path": RAW_DIR / "original-rus.txt",
                "language": "ru"
            },
            "glenny": {
                "path": RAW_DIR / "glenny-translation-en.txt",
                "language": "en"
            },
            "pevear": {
                "path": RAW_DIR / "pevear-translation-en.txt",
                "language": "en"
            }
        },
        output_path=TRAINING_DIR / "boundary_candidates.csv"
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
