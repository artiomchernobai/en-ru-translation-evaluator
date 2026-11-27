"""Configurable local model and atomic, resumable per-pair reports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from .backend import LocalMQMBackend
from .engine import analyze
from .reports import compact_report, restore_report, diagnostics


def read_settings(root):
    path=Path(root)/'data/mqm_settings.json'
    settings=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    allowed={'model','base_url','timeout','max_tokens','json_mode','scope','resume','revision','max_retries','include_word_links','max_fragments','min_priority','merge_windows','merge_overlap','merge_max_chars'}
    if not isinstance(settings,dict) or set(settings)-allowed:raise ValueError('Unsupported MQM settings')
    return settings


def configured_judge(settings):
    backend=LocalMQMBackend(model=settings.get('model',''),base_url=settings.get('base_url','http://127.0.0.1:1234/v1'),
                            timeout=settings.get('timeout',120),max_tokens=settings.get('max_tokens',2048),json_mode=settings.get('json_mode',True),max_retries=settings.get('max_retries',1),include_word_links=settings.get('include_word_links',False))
    backend.metadata['revision']=settings.get('revision','unspecified')
    from .windows import DEFAULT_POLICY
    backend.review_policy={k:settings.get(k,v) for k,v in DEFAULT_POLICY.items()}
    backend.metadata['review_policy']=dict(backend.review_policy)
    return backend


def atomic_json(path, result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
        os.replace(temporary,path)
    finally:
        if temporary:temporary.unlink(missing_ok=True)


def analyze_file(candidate_path, output_path, *, backend, scope='candidates', resume=True, window_strategy=None, max_fragments=0, min_priority=0, corpus_path=None, group_indices=None):
    candidate_path,output_path=Path(candidate_path),Path(output_path)
    diagnostics_path=output_path.with_suffix('.diagnostics.json')
    inputs=[candidate_path]+([Path(corpus_path)] if corpus_path is not None else [])
    if any(p.resolve() in (output_path.resolve(),diagnostics_path.resolve()) for p in inputs):
        raise ValueError('MQM output must not overwrite inputs')
    if output_path.resolve()==diagnostics_path.resolve():raise ValueError('MQM report and diagnostics must have different paths')
    payload=candidate_path.read_bytes()
    data=json.loads(payload.decode('utf-8-sig'))
    if window_strategy is not None and data.get('window_config',{}).get('strategy','fixed')!=window_strategy:
        raise ValueError('MQM context strategy differs from the selected candidate mode')
    corpus_payload=None
    if corpus_path is not None:
        from .contracts import attach_corpus
        corpus_payload=Path(corpus_path).read_bytes()
        data=attach_corpus(data,json.loads(corpus_payload.decode('utf-8-sig')))
    previous=None
    if resume and output_path.exists():
        try:previous=restore_report(json.loads(output_path.read_text(encoding='utf-8')),data)
        except (ValueError,OSError,KeyError,TypeError,IndexError):pass
    if previous is not None and diagnostics_path.exists():
        try:
            audit=json.loads(diagnostics_path.read_text(encoding='utf-8'))
            if audit.get('run_fingerprint')==previous.get('run_fingerprint'):
                by_hash={f['request_hash']:f for f in audit['fragments'] if f.get('request_hash')}
                for fragment in previous['fragments']:
                    saved=by_hash.get(fragment.get('request_hash'),{})
                    for key in ('attempts','initial_proposals','confirmation_response','confirmation_error','confirmation_usage'):
                        if key in saved:fragment[key]=saved[key]
        except (ValueError,OSError,KeyError,TypeError,AttributeError):pass
    def save(report):
        report['inputs']={'candidates':{'path':str(candidate_path),'sha256':hashlib.sha256(payload).hexdigest()}}
        if corpus_payload is not None:report['inputs']['corpus']={'path':str(corpus_path),'sha256':hashlib.sha256(corpus_payload).hexdigest()}
        report['updated_at']=datetime.now(timezone.utc).isoformat()
        atomic_json(diagnostics_path,diagnostics(report))
        atomic_json(output_path,compact_report(report))
    result=analyze(data,backend,scope=scope,previous=previous,progress=save,max_fragments=max_fragments,min_priority=min_priority,group_indices=group_indices)
    print(f'[MQM] {result["status"]}; {output_path}',flush=True)
    return compact_report(result)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Local GEMBA-MQM-style review of candidates')
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corpus',type=Path)
    parser.add_argument('--model',required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:1234/v1')
    parser.add_argument('--scope',choices=('candidates','all'),default='candidates')
    parser.add_argument('--no-resume',action='store_true')
    args=parser.parse_args(argv)
    result=analyze_file(args.candidates,args.output,backend=LocalMQMBackend(model=args.model,base_url=args.base_url),scope=args.scope,resume=not args.no_resume,corpus_path=args.corpus)
    if result['status']!='complete':raise SystemExit('MQM completed partially; inspect the saved report')


if __name__=='__main__':main()
