import json
import re
from pathlib import Path

import unicodedata


class TextNormalizer:

    def normalize(self, text):
        # Unicode normalization
        text = unicodedata.normalize("NFC", text)

        # Remove zero-width characters
        text = re.sub(r"[\u200b-\u200f\uFEFF]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def process_sentences(self, sentences):
        normalized_sentences = []

        for sentence in sentences:
            original_text = sentence["text"]
            normalized_text = self.normalize(original_text)
            normalized_sentence = sentence.copy()
            normalized_sentence["normalized_text"] = normalized_text
            normalized_sentences.append(normalized_sentence)

        return normalized_sentences

    def process_file(self, input_path, output_path):
        input_path = Path(input_path)
        output_path = Path(output_path)

        with open(input_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        data["sentences"] = self.process_sentences(data["sentences"])

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path,"w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(
            f"Normalized {input_path.name}: "
            f"{data['sentence_count']} sentences"
        )
        print(f"Saved: {output_path}")
