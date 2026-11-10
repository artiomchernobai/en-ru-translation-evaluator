from pathlib import Path
import numpy as np

class LaserEmbedder:
    EMBEDDING_DIMENSION = 1024

    LANGUAGES = {
        "ru": "rus_Cyrl",
        "en": "eng_Latn"
    }

    def __init__(self, language: str, batch_size: int=16):
        if language not in self.LANGUAGES:
            raise ValueError(
                f"Language unsuported: {language}"
            )

        if batch_size < 1:
            raise ValueError("Batch size must be positive")

        self.language = language
        self.batch_size = batch_size

        from laser_encoders import LaserEncoderPipeline

        self.encoder = LaserEncoderPipeline(
            lang=self.LANGUAGES[language]
        )

    def embed_file(self, input_path, output_path):
        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.is_file():
            raise FileNotFoundError(input_file)

        with input_file.open("r", encoding="utf-8") as file:
            sentences = [
                line.strip()
                for line in file
            ]

        if not sentences:
            raise ValueError("No sentences found")

        if any(not sentence for sentence in sentences):
            raise ValueError("Contains empty lines")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        amount_sentences = len(sentences)

        with output_file.open("wb") as output:
            for start in range(0, amount_sentences, self.batch_size):
                batch = sentences[
                    start : start + self.batch_size
                ]

                embeddings = self.encoder.encode_sentences(batch)

                embeddings = np.asarray(
                    embeddings,
                    dtype=np.float32
                )

                expected_shape = (
                    len(batch),
                    self.EMBEDDING_DIMENSION
                )

                if embeddings.shape != expected_shape:
                    raise ValueError(
                        f"Unexpected shape for embeddings: "
                        f"{embeddings.shape}, "
                        f"expected {expected_shape}"
                    )

                embeddings.tofile(output)

                processed = min(
                    start + self.batch_size,
                    amount_sentences,
                )

                print(f"Processed: {processed}/{amount_sentences}")

        print("\nEmbeddings generation completed.")
        print(f"Input sentences: {amount_sentences}")
        print(f"Output file: {output_file}")
        print(f"Embedding dimension: {self.EMBEDDING_DIMENSION}")
        print(f"Data type: float32")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LASER embeddings"
    )

    parser.add_argument(
        "input",
        help="Input TXT file"
    )

    parser.add_argument(
        "output",
        help="Output binary file"
    )

    parser.add_argument(
        "language",
        choices=["ru", "en"],
        help="Input language"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    args = parser.parse_args()

    embedder = LaserEmbedder(
        language=args.language,
        batch_size=args.batch_size
    )

    embedder.embed_file(
        args.input,
        args.output
    )