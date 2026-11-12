"""Local, batched HuggingFace inference for the fine-tuned Russian dash model.

Input is exactly BoundaryCandidateExtractor.extract(...) output for the *same* text.
Trainer convention: 0 = KEEP; 1 = SPLIT. Inference uses left/right text pairs
and keeps the original absolute offsets; the model does not edit the text.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import re


def dash_boundary_allowed(text: str, pos: int) -> bool:
    """Whether a dash can introduce a NEW Russian dialogue/attribution unit.

    If nothing resembling terminal/dialogue punctuation precedes this dash,
    treat it as syntactic (KEEP), even on a physically new PDF line.  A true
    paragraph break is deliberately exempt: two distinct paragraphs need not
    share punctuation (e.g. a heading before the first dialogue line).
    """
    if not (0 < pos < len(text)) or text[pos] not in "—–":
        return False
    preceding = text[:pos]
    content = preceding.rstrip()
    if not content:
        return True
    between = preceding[len(content):]
    if re.search(r"\n[^\S\r\n]*\n", between):
        return True
    return content[-1] in ".,!?…;:»”\"'"



@dataclass(frozen=True)
class BoundaryVerdict:
    position: int
    label: int
    probability_split: float
    rule: str
    origin: str  # rubert | line_dash | punctuation_guard | rubert_guarded

    @property
    def decision(self) -> str:
        return "SPLIT" if self.label else "KEEP"


class RuBoundaryClassifier:
    def __init__(self, model_dir, *, device: str = "auto", batch_size: int = 8,
                 max_length: int = 256, context_size: int = 200,
                 threshold: float = 0.5):
        if batch_size < 1 or max_length < 8 or context_size < 1 or not 0 <= threshold <= 1:
            raise ValueError("Invalid batch_size, max_length, context_size or threshold")
        self.model_dir = Path(model_dir)
        self.device_name = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.context_size = context_size
        self.threshold = threshold
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"RuBERT model directory missing: {self.model_dir}. "
                "Copy the entire best_model folder (model weights, config and tokenizer) into the project."
            )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers in this project's Python environment") from exc
        device = self.device_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but the current PyTorch cannot access a GPU")
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        if self._model.config.num_labels != 2:
            raise ValueError("Expected a two-class checkpoint (0=KEEP, 1=SPLIT)")
        self._device = torch.device(device)
        self._model.to(self._device).eval()
        self._torch = torch

    def predict_candidates(self, text: str, candidates: list[dict]) -> list[BoundaryVerdict]:
        """Pass builder records, return one verdict per record in original order.

        The source_sha256 check rejects stale offsets (e.g. candidates generated
        from raw text but inference run on clean_text(raw_text)).
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        records = []
        for row in candidates:
            if row.get("language") != "ru":
                raise ValueError("RuBoundaryClassifier accepts only Russian candidates")
            if row.get("source_sha256") != digest:
                raise ValueError(
                    "Boundary candidate source_sha256 mismatch. Generate candidates "
                    "with builder.extract(cleaned_text, ...) for the EXACT text being segmented."
                )
            position = int(row["position"])
            if not (0 < position < len(text)) or text[position] not in "—–":
                raise ValueError(f"Invalid dash candidate at position {position}")
            records.append((position, str(row["rule"])))
        verdicts: list[BoundaryVerdict | None] = [None] * len(records)
        needs_model = []
        for i, (pos, rule) in enumerate(records):
            if "line_dash" in rule.split(";"):
                allowed = dash_boundary_allowed(text, pos)
                verdicts[i] = BoundaryVerdict(
                    pos, int(allowed), 1.0 if allowed else 0.0, rule,
                    "line_dash" if allowed else "punctuation_guard"
                )
            else:
                needs_model.append((i, pos, rule))
        if needs_model:
            self._load()
            for start in range(0, len(needs_model), self.batch_size):
                block = needs_model[start:start + self.batch_size]
                text_a = [text[max(0, pos - self.context_size):pos] for _, pos, _ in block]
                text_b = [text[pos:pos + self.context_size] for _, pos, _ in block]
                tokens = self._tokenizer(
                    text_a, text_b, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt"
                )
                tokens = {key: val.to(self._device) for key, val in tokens.items()}
                with self._torch.inference_mode():
                    logits = self._model(**tokens).logits
                    probabilities = self._torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
                for (index, position, rule), p_split in zip(block, probabilities):
                    predicted = int(p_split >= self.threshold)
                    allowed = dash_boundary_allowed(text, position)
                    verdicts[index] = BoundaryVerdict(
                        position, predicted if allowed else 0, float(p_split), rule,
                        "rubert_guarded" if predicted and not allowed else "rubert"
                    )
        return [v for v in verdicts if v is not None]
