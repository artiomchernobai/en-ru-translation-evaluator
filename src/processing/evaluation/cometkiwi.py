"""Adapter for a preloaded COMETKiwi model. No document paths or UI imports."""
import math
from pathlib import Path

DEFAULT_MODEL = 'Unbabel/wmt22-cometkiwi-da'


class CometKiwiBackend:
    def __init__(self, model, *, model_name=DEFAULT_MODEL, checkpoint=None, batch_size=8, gpus=0, max_tokens=510):
        if type(batch_size) is not int or batch_size < 1 or type(max_tokens) is not int or max_tokens < 1 or gpus not in (0, 1):
            raise ValueError('Invalid COMETKiwi batch size, token limit or device')
        requires_references = getattr(model, 'requires_references', None)
        if callable(requires_references) and requires_references():
            raise ValueError('Нужна модель COMETKiwi без эталонного перевода (ref)')
        self.model = model
        self.batch_size, self.gpus = batch_size, gpus
        self.tokenizer = model.encoder.tokenizer
        limits = [max_tokens]
        for limit in (getattr(model.encoder, 'max_positions', None), getattr(self.tokenizer, 'model_max_length', None)):
            if isinstance(limit, int) and 2 < limit < 1000000:
                limits.append(limit - 2)
        self.max_tokens = min(limits)
        self.metadata = {'name': model_name, 'checkpoint': str(checkpoint) if checkpoint else None,
                         'batch_size': batch_size, 'gpus': gpus, 'max_tokens': self.max_tokens,
                         'score_scope': 'entire_window', 'score_direction': 'higher_is_better'}

    @classmethod
    def load(cls, model_name=DEFAULT_MODEL, **kwargs):
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError as error:
            raise RuntimeError('Установите unbabel-comet: pip install -r src/processing/evaluation/requirements.txt') from error
        print(f'[COMETKiwi] Загрузка модели: {model_name}', flush=True)
        checkpoint = str(Path(model_name)) if Path(model_name).is_file() else download_model(model_name)
        model = load_from_checkpoint(checkpoint)
        model.eval()
        return cls(model, model_name=model_name, checkpoint=checkpoint, **kwargs)

    def predict(self, pairs):
        results, accepted, positions = [], [], []
        for i, pair in enumerate(pairs):
            lengths = {key: len(self.tokenizer.encode(pair[key], add_special_tokens=True, truncation=False)) for key in ('src', 'mt')}
            if max(lengths.values()) > self.max_tokens:
                results.append({'status': 'too_long', 'score': None, 'token_counts': lengths})
            else:
                results.append(None)
                accepted.append(pair)
                positions.append(i)
        if accepted:
            prediction = self.model.predict(accepted, batch_size=self.batch_size, gpus=self.gpus, progress_bar=False)
            scores = list(prediction.scores)
            if len(scores) != len(accepted):
                raise ValueError('COMETKiwi returned an unexpected number of scores')
            for i, score in zip(positions, scores):
                score = float(score)
                if not math.isfinite(score):
                    raise ValueError('COMETKiwi returned a non-finite score')
                results[i] = {'status': 'scored', 'score': score}
        return results
