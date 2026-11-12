"""Run only COMETKiwi in its own Python environment, without UI dependencies."""
import argparse
from pathlib import Path
from src.processing.pipeline.config import SOURCE, TARGETS
from .stages import evaluate_pairs, selected_pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description='Isolated COMETKiwi worker')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--documents', nargs='+', choices=(SOURCE, *TARGETS), required=True)
    args = parser.parse_args(argv)
    evaluate_pairs(args.root, selected_pairs(args.root, args.documents))


if __name__ == '__main__':
    main()
