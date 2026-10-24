"""Export only an audited rebound training run; never relabel a single flip."""
import argparse
from .export_cpu import export

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output, rebound=True)
