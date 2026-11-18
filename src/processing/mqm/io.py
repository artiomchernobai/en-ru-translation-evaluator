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


def read_settings(root):
    path=Path(root)/'data/mqm_settings.json'
    settings=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    allowed={'model','base_url','timeout','max_tokens','json_mode','scope','resume','revision'}
    if not isinstance(settings,dict) or set(settings)-allowed:raise ValueError('Unsupported MQM settings')
    return settings


def configured_judge(settings):
    backend=LocalMQMBackend(model=settings.get('model',''),base_url=settings.get('base_url','http://127.0.0.1:1234/v1'),
                            timeout=settings.get('timeout',120),max_tokens=settings.get('max_tokens',2048),json_mode=settings.get('json_mode',True))
    backend.metadata['revision']=settings.get('revision','unspecified')
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


def analyze_file(candidate_path, output_path, *, backend, scope='candidates', resume=True, window_strategy=None):
    candidate_path,output_path=Path(candidate_path),Path(output_path)
    if candidate_path.resolve()==output_path.resolve():raise ValueError('MQM output must not overwrite candidates')
    payload=candidate_path.read_bytes()
    data=json.loads(payload.decode('utf-8-sig'))
    if window_strategy is not None and data.get('window_config',{}).get('strategy','fixed')!=window_strategy:
        raise ValueError('MQM context strategy differs from the selected candidate mode')
    previous=None
    if resume and output_path.exists():
        try:previous=json.loads(output_path.read_text(encoding='utf-8'))
        except (ValueError,OSError):pass
    def save(report):
        report['inputs']={'candidates':{'path':str(candidate_path),'sha256':hashlib.sha256(payload).hexdigest()}}
        report['updated_at']=datetime.now(timezone.utc).isoformat()
        atomic_json(output_path,report)
    result=analyze(data,backend,scope=scope,previous=previous,progress=save)
    print(f'[MQM] {result["status"]}; {output_path}',flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='Local GEMBA-MQM-style review of candidates')
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:1234/v1')
    parser.add_argument('--scope',choices=('candidates','all'),default='candidates')
    parser.add_argument('--no-resume',action='store_true')
    args=parser.parse_args(argv)
    result=analyze_file(args.candidates,args.output,backend=LocalMQMBackend(model=args.model,base_url=args.base_url),scope=args.scope,resume=not args.no_resume)
    if result['status']!='complete':raise SystemExit('MQM completed partially; inspect the saved report')


if __name__=='__main__':main()
