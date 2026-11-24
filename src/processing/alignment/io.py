"""File/CLI adapters for the document-independent word alignment engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .sentences import parse_alignment
from .simalign_backend import SimAlignBackend
from .words import WordAligner


def configured_word_aligner(root=None) -> WordAligner:
    """Runtime configuration belongs to the caller, not the alignment algorithm."""
    from src.processing.pipeline.settings import read_settings
    from src.processing.pipeline.config import ROOT
    options=read_settings(root or ROOT)['word_alignment']
    return WordAligner(SimAlignBackend(model=options['model'],device=options['device'],layer=options['layer'],
                       min_similarity=options['min_similarity'],min_margin=options['min_margin']),
                       selected_method=options['method'])



def align_word_files(source_json: Path, target_json: Path, alignment_path: Path,
                     output_path: Path, *, aligner: WordAligner | None = None) -> dict:
    paths = {'source': Path(source_json), 'target': Path(target_json), 'sentence_alignment': Path(alignment_path)}
    output_path = Path(output_path)
    if any(output_path.resolve() == p.resolve() for p in paths.values()):
        raise ValueError("Word alignment output must not overwrite an input file")
    contents = {name: path.read_bytes() for name, path in paths.items()}
    source, target = (json.loads(contents[name].decode('utf-8-sig')) for name in ('source', 'target'))
    groups = parse_alignment(contents['sentence_alignment'].decode('utf-8-sig'))
    def progress(done: int, total: int) -> None:
        if done == 1 or done % 10 == 0 or done == total:
            print(f"[Word alignment] {source['document']} -> {target['document']}: {done}/{total} blocks", flush=True)

    result = (aligner or configured_word_aligner()).align(source, target, groups, progress=progress)
    result['inputs'] = {name: {'path': str(paths[name]), 'sha256': hashlib.sha256(payload).hexdigest()}
                        for name, payload in contents.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"[Word alignment] {source['document']} -> {target['document']}: {result['summary']}; {output_path}", flush=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Align words in any normalized original/translation pair')
    parser.add_argument('--source', type=Path, required=True, help='Normalized source JSON')
    parser.add_argument('--target', type=Path, required=True, help='Normalized translation JSON')
    parser.add_argument('--alignment', type=Path, required=True, help='Vecalign .align file')
    parser.add_argument('--output', type=Path, required=True, help='Word alignment JSON output')
    args = parser.parse_args(argv)
    align_word_files(args.source, args.target, args.alignment, args.output)


if __name__ == '__main__':
    main()
