import json
from pathlib import Path

class TextJsonTxtExporter:

    def export_to_txt_file(self, input_path, output_path):
        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.is_file():
            raise FileNotFoundError(f"{input_path} is not a file")

        with input_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")

        sentences = data.get("sentences")

        if not isinstance(sentences, list):
            raise ValueError("Expected a 'sentences' array")

        lines = []

        for sentence in sentences:
            if not isinstance(sentence, dict):
                raise ValueError("Invalid sentence record")

            text = sentence.get("normalized_text", "")

            if not isinstance(text, str):
                raise ValueError(
                    f"Invalid text at sentence {sentence.get('sentence_id')}"
                )

            text = text.strip()

            if not text:
                raise ValueError(f"Empty sentence at {sentence.get('sentence_id')}")

            if "\n" in text or "\r" in text:
                raise ValueError(f"Unexpected line break in sentence {sentence.get('sentence_id')}")

            lines.append(text)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as file:
            file.write("\n".join(lines)+"\n")

        print(f"Exported {len(lines)} sentences to {output_path}")
