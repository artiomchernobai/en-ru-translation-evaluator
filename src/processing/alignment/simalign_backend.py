"""Lazy SimAlign adapter with explicit protection against model truncation."""
from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError

from .words import BlockTooLongError


class SimAlignBackend:
    def __init__(self, model: str = "bert", *, device: str = "cpu", layer: int = 8):
        if not model or not isinstance(model, str):
            raise ValueError("SimAlign model must be a model name or local path")
        if type(layer) is not int or layer < 0:
            raise ValueError("SimAlign layer must be a nonnegative integer")
        self.model, self.device, self.layer = model, device, layer
        self._aligner = None

    @property
    def metadata(self) -> dict:
        try:
            installed_version = version('simalign')
        except PackageNotFoundError:
            installed_version = None
        return {"name": "SimAlign", "version": installed_version, "model": self.model,
                "device": self.device, "layer": self.layer, "token_type": "word",
                "matching_methods": ["inter", "mwmf", "itermax"], "confidence_scores": False,
                "resolved_model": getattr(self._aligner, 'model', None),
                "model_revision": getattr(self._aligner.embed_loader.emb_model.config, '_commit_hash', None)
                if self._aligner is not None else None}

    def _load(self):
        if self._aligner is None:
            try:
                from simalign import SentenceAligner
            except ImportError as error:
                raise RuntimeError("Install word alignment dependencies: python -m pip install -r "
                                   "src/processing/requirements-word-alignment.txt") from error
            aligner = SentenceAligner(model=self.model, token_type='word', matching_methods='mai',
                                      device=self.device, layer=self.layer)
            self._aligner = aligner
        return self._aligner

    def align_tokens(self, source: list[str], target: list[str]) -> dict:
        aligner = self._load()
        tokenizer = aligner.embed_loader.tokenizer
        config = aligner.embed_loader.emb_model.config
        limits = [limit for limit in (getattr(tokenizer, 'model_max_length', None),
                  getattr(config, 'max_position_embeddings', None))
                  if isinstance(limit, int) and 0 < limit < 1_000_000]
        if not limits:
            raise RuntimeError("Cannot determine model token limit; refusing silent truncation")
        limit = min(limits)
        for side, tokens in (('source', source), ('target', target)):
            if any(not tokenizer.tokenize(word) for word in tokens):
                raise ValueError(f"{side} contains a word the SimAlign tokenizer cannot encode")
            # Match SimAlign's pretokenized transformer input, including special tokens.
            encoded = tokenizer(tokens, is_split_into_words=True, add_special_tokens=True,
                                truncation=False, return_attention_mask=False)['input_ids']
            if len(encoded) > limit:
                raise BlockTooLongError(f"{side} block needs {len(encoded)} model tokens; limit is {limit}")
        return aligner.get_word_aligns(source, target)
