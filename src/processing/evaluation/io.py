"""File adapter and CLI for arbitrary original/translation pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from src.processing.alignment.sentences import parse_alignment
from .settings import read_settings
from .windows import build_windows
from .engine import evaluate_windows
from .cometkiwi import CometKiwiBackend, DEFAULT_MODEL


def configured_backend(root):
    settings = read_settings(root)
    return CometKiwiBackend.load(settings.get('model', os.environ.get('COMETKIWI_MODEL', DEFAULT_MODEL)),
                                batch_size=settings.get('batch_size', 8), gpus=settings.get('gpus', 0))


def evaluate_files(source_json, target_json, alignment_path, output_path, *, backend, radius=1):
    paths = {'source': Path(source_json), 'target': Path(target_json), 'alignment': Path(alignment_path)}
    output_path = Path(output_path)
    if output_path.resolve() in [p.resolve() for p in paths.values()]:
        raise ValueError('Evaluation output must not overwrite inputs')
    contents = {key: path.read_bytes() for key, path in paths.items()}
    source, target = [json.loads(contents[key].decode('utf-8-sig')) for key in ('source', 'target')]
    text = contents['alignment'].decode('utf-8-sig')
    windows = build_windows(source, target, parse_alignment(text), radius=radius,
                            raw_lines=[line for line in text.splitlines() if line.strip()])
    print(f"[COMETKiwi] {source.get('document')} → {target.get('document')}: {len(windows)} окон", flush=True)
    result = evaluate_windows(windows, backend)
    result.update({'source_document': source.get('document'), 'target_document': target.get('document'),
                   'radius': radius, 'created_at': datetime.now(timezone.utc).isoformat(),
                   'inputs': {key: {'path': str(paths[key]), 'sha256': hashlib.sha256(value).hexdigest()} for key, value in contents.items()}})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"[COMETKiwi] {result['summary']} → {output_path}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='COMETKiwi on neighboring Vecalign groups')
    for name in ('source', 'target', 'alignment', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--model', default=os.environ.get('COMETKIWI_MODEL', DEFAULT_MODEL))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpus', type=int, choices=(0, 1), default=0)
    parser.add_argument('--radius', type=int, default=1)
    args = parser.parse_args(argv)
    backend = CometKiwiBackend.load(args.model, batch_size=args.batch_size, gpus=args.gpus)
    evaluate_files(args.source, args.target, args.alignment, args.output, backend=backend, radius=args.radius)


if __name__ == '__main__':
    main()
