
import json
import re

from pathlib import Path


class CorpusBuilder:

    ALIGNMENT_PATTERN = re.compile(
        r"^\[(.*?)\]:\[(.*?)\]:([0-9.eE+-]+)$"
    )

    def load_sentences(self, json_path: Path) -> list[dict]:
        with Path(json_path).open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid JSON file: {json_path}")

        sentences = data.get("sentences")

        if not isinstance(sentences, list):
            raise ValueError(f"Missing sentences array: {json_path}")

        for position, sentence in enumerate(sentences):
            if not isinstance(sentence, dict):
                raise ValueError(f"Invalid sentence: {position}")

            if "sentence_id" not in sentence:
                raise ValueError(f"Missing sentence_id: {position}")

            text = sentence.get("normalized_text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid sentence text: {position}")

        return sentences

    def parse_indices(self, value: str) -> list[int]:
        value = value.strip()

        if not value:
            return []

        try:
            indices = [int(item.strip()) for item in value.split(",")]
        except ValueError as e:
            raise ValueError(f"Invalid indices: {value}") from e

        if any(index < 0 for index in indices):
            raise ValueError(f"Negative index found: {value}")

        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate indices found: {value}")

        return indices

    def parse_alignments(self, alignment_path: Path) -> list[dict]:
        alignments = []

        with Path(alignment_path).open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                match = self.ALIGNMENT_PATTERN.match(line)

                if not match:
                    raise ValueError(
                        f"Invalid alignment at line {line_number}: {line}"
                    )

                source_indices = self.parse_indices(match.group(1))
                target_indices = self.parse_indices(match.group(2))

                cost = float(match.group(3))

                if not source_indices and not target_indices:
                    raise ValueError(
                        f"Both sides are empty at line {line_number}"
                    )

                alignments.append({
                    "source_indices": source_indices,
                    "target_indices": target_indices,
                    "cost": cost
                })

        return alignments

    def build_segment_text(
        self,
        sentences: list[dict],
        indices: list[int],
        alignment_line: int
    ) -> dict:

        selected = []

        for index in indices:
            if index >= len(sentences):
                raise IndexError(
                    f"Alignment line {alignment_line}: "
                    f"sentence index {index} exceeds "
                    f"sentence count {len(sentences)}"
                )

            selected.append(sentences[index])

        return {
            "sentence_ids": [
                sentence["sentence_id"] for sentence in selected
            ],
            "text": " ".join(
                sentence["normalized_text"].strip()
                for sentence in selected
            )
        }

    def build_corpus(
        self,
        source_json: Path,
        target_json: Path,
        alignment_path: Path,
        output: Path,
        source_document: str,
        target_document: str,
        source_lang: str = "ru",
        target_lang: str = "en"
    ):

        source_sentences = self.load_sentences(source_json)
        target_sentences = self.load_sentences(target_json)

        alignments = self.parse_alignments(alignment_path)

        segments = []

        for segment_id, alignment in enumerate(alignments):

            source = self.build_segment_text(
                source_sentences,
                alignment["source_indices"],
                segment_id + 1
            )

            target = self.build_segment_text(
                target_sentences,
                alignment["target_indices"],
                segment_id + 1
            )

            if not alignment["source_indices"] or not alignment["target_indices"]:
                status = "unaligned"
            else:
                status = "aligned"

            segments.append({
                "segment_id": segment_id,
                "source": source,
                "target": target,
                "vecalign_cost": alignment["cost"],
                "alignment_status": status,
                "word_alignment": None,
                "quality_score": None,
                "errors": []
            })

        corpus = {
            "corpus_id": f"{source_document}_{target_document}",
            "source_language": source_lang,
            "target_language": target_lang,
            "source_document": source_document,
            "target_document": target_document,
            "alignment_method": "Vecalign",
            "segment_count": len(segments),
            "segments": segments
        }

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                corpus,
                file,
                ensure_ascii=False,
                indent=4
            )

        print(
            f"Corpus created: {output_path}\n"
            f"Segments: {len(segments)}"
        )
