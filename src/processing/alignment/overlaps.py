"""Dependency-free Vecalign overlap vocabulary (including PAD/BLANK_LINE)."""
from pathlib import Path
import os
import tempfile

MAX_LINE_CHARS = 10000


def overlap_vocabulary(lines, maximum):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('Overlap size must be a positive integer')
    sentences = [line.strip() or 'BLANK_LINE' for line in lines]
    if not sentences:
        raise ValueError('No input sentences')
    vocabulary = set()
    for size in range(1, maximum+1):
        # Vecalign front-pads higher-order layers with this sentinel.
        if size > 1:
            vocabulary.add('PAD')
        for i in range(len(sentences)-size+1):
            vocabulary.add(' '.join(sentences[i:i+size])[:MAX_LINE_CHARS])
    return sorted(vocabulary)


def write_overlaps(source, output, maximum=4):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Overlap output must not overwrite input')
    with source.open(encoding='utf-8-sig') as stream:
        vocabulary = overlap_vocabulary(stream, maximum)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write('\n'.join(vocabulary)+'\n')
        os.replace(temporary, output)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    print(f'[Overlap] {len(vocabulary)} уникальных комбинаций, встроенный генератор', flush=True)
    return vocabulary
