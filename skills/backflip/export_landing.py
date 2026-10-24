"""Explicit v2 export: actual config and phase source must match."""
import argparse
from .export_cpu import export

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    export(a.checkpoint,a.output,landing=True)
