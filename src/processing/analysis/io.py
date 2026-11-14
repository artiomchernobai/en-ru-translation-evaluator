"""Atomic JSON adapter/CLI. No specific work or translator in the engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from .candidates import collect_candidates


def load_settings(root):
    path = Path(root) / 'data/candidate_settings.json'
    settings = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(settings, dict) or any(k not in ('low_quantile', 'high_quantile', 'glossary') for k in settings):
        raise ValueError('Unsupported candidate settings')
    return settings


def collect_files(word_path, comet_path, output_path, **settings):
    paths = {'word_alignment': Path(word_path), 'cometkiwi': Path(comet_path)}
    output_path = Path(output_path)
    if output_path.resolve() in [p.resolve() for p in paths.values()]:
        raise ValueError('Output must not overwrite inputs')
    payloads = {key: path.read_bytes() for key, path in paths.items()}
    words, comet = [json.loads(payloads[k].decode('utf-8-sig')) for k in paths]
    result = collect_candidates(words, comet, **settings)
    result['created_at'] = datetime.now(timezone.utc).isoformat()
    result['inputs'] = {k: {'path': str(paths[k]), 'sha256': hashlib.sha256(data).hexdigest()} for k, data in payloads.items()}
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
    print(f"[Candidates] {result['source_document']} -> {result['target_document']}: {result['summary']}; {output_path}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Find review candidates from word alignment and COMETKiwi')
    parser.add_argument('--words', type=Path, required=True)
    parser.add_argument('--comet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path, help='JSON with optional quantiles and bilingual glossary')
    args = parser.parse_args(argv)
    settings = json.loads(args.settings.read_text(encoding='utf-8')) if args.settings else {}
    collect_files(args.words, args.comet, args.output, **settings)


if __name__ == '__main__':
    main()
