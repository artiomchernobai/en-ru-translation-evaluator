"""CLI uses the same runner and settings as the interface."""
import argparse
from pathlib import Path
from .config import ROOT, STAGES
from .settings import read_settings
from .runner import run_stage


def main(argv=None):
    parser=argparse.ArgumentParser(description='Text processing pipeline')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['all',*STAGES],default='all')
    parser.add_argument('--documents',nargs='+')
    parser.add_argument('--max-size',type=int)
    parser.add_argument('--deletion-percentile',type=float)
    parser.add_argument('--candidate-group',type=int)
    parser.add_argument('--target')
    args=parser.parse_args(argv)
    settings=read_settings(args.root)
    if args.candidate_group is not None:
        if args.stage!='mqm' or not args.target:parser.error('--candidate-group requires --stage mqm --target')
        from src.processing.mqm.stages import analyze_candidate
        return analyze_candidate(args.root,args.candidate_group,args.target)
    run_stage(args.root,args.stage,args.documents or settings['documents'],args.max_size,args.deletion_percentile)


if __name__=='__main__':main()
