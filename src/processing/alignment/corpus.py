
import json
import os
import tempfile
from copy import deepcopy
from .words import _document, _fingerprint
from src.processing.alignment.sentences import parse_alignment, parse_indices, validate_alignment

from pathlib import Path


class CorpusBuilder:

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

    parse_indices = staticmethod(parse_indices)

    def parse_alignments(self, alignment_path: Path) -> list[dict]:
        return parse_alignment(Path(alignment_path).read_text(encoding="utf-8-sig"))

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
        target_lang: str = "en",
        word_alignment_path: Path | None = None
    ):

        source_sentences = self.load_sentences(source_json)
        target_sentences = self.load_sentences(target_json)

        alignments = self.parse_alignments(alignment_path)

        validate_alignment(alignments, len(source_sentences), len(target_sentences))

        documents = {}
        for side, path, name, language in [('source',source_json,source_document,source_lang), ('target',target_json,target_document,target_lang)]:
            raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
            raw.setdefault('document',name)
            raw.setdefault('language',language)
            if raw['document']!=name or raw['language']!=language:
                raise ValueError('Corpus document identity mismatch')
            documents[side]=_document(raw)
        words=None
        if word_alignment_path is not None:
            words=json.loads(Path(word_alignment_path).read_text(encoding='utf-8-sig'))
            if words.get('sentence_alignment_fingerprint') != _fingerprint(alignments):
                raise ValueError('Corpus and word alignment groups differ')
            for side in documents:
                if words[side]['fingerprint'] != documents[side]['fingerprint']:
                    raise ValueError('Corpus and word alignment texts differ')
            if len(words['blocks'])!=len(alignments):raise ValueError('Corpus block count differs')
            documents={side:deepcopy(words[side]) for side in documents}
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

            for label, selected in [('source',source),('target',target)]:
                indices=alignment[label+'_indices']
                selected['sentence_indices']=list(indices)
                selected['token_ids']=[token_id for i in indices for token_id in documents[label]['sentences'][i]['token_ids']]
            word=None
            if words is not None:
                block=words['blocks'][segment_id]
                if any(block[side]['sentence_indices']!=alignment[side+'_indices'] or block[side]['token_ids']!=value['token_ids'] for side,value in [('source',source),('target',target)]):
                    raise ValueError('Corpus token membership differs')
                word=deepcopy(block['word_alignment'])
                for links in word.get('methods',{}).values():
                    if any(link['source_token_id'] not in source['token_ids'] or link['target_token_id'] not in target['token_ids'] for link in links):
                        raise ValueError('Corpus token link outside group')
            segments.append({
                "segment_id": segment_id,
                "source": source,
                "target": target,
                "vecalign_cost": alignment["cost"],
                "alignment_status": status,
                "word_alignment": word,
                "quality_score": None,
                "errors": []
            })

        corpus = {
            "schema_version": 2,
            "documents": documents,
            "selected_word_method": words['selected_method'] if words else None,
            "sentence_alignment_fingerprint": _fingerprint(alignments),
            "word_alignment_backend": deepcopy(words.get("backend")) if words else None,
            "offsets": {"unit": "unicode_code_point", "end": "exclusive", "document_text": "normalized sentences joined with newline"},
            "word_alignment_status": 'included' if words else 'not_computed',
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

        temporary=None
        try:
            with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=output_path.parent,delete=False) as stream:
                temporary=Path(stream.name)
                json.dump(corpus,stream,ensure_ascii=False,indent=2,allow_nan=False)
            os.replace(temporary,output_path)
        finally:
            if temporary:temporary.unlink(missing_ok=True)

        print(
            f"Corpus created: {output_path}\n"
            f"Segments: {len(segments)}"
        )

        return corpus
