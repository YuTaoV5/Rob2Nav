"""Check ONNX Runtime output against the deterministic checkpoint actor."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=Path)
parser.add_argument('onnx', type=Path)
parser.add_argument('vectors', type=Path)
parser.add_argument('report', type=Path)
args = parser.parse_args()
vectors = np.load(args.vectors)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
session = ort.InferenceSession(str(args.onnx), sess_options=opts, providers=['CPUExecutionProvider'])
observations = vectors['observations']
reference = vectors['raw_action']
output = session.run(['raw_action'], {'policy': observations})[0]
assert output.shape == reference.shape and np.isfinite(output).all()
error = float(np.abs(output-reference).max())
report = dict(scope='deterministic raw actor output on synthetic inputs',
              max_abs_error=error, passed=error <= 1e-5, onnxruntime=ort.__version__,
              providers=session.get_providers(), input_shape=list(observations.shape),
              output_shape=list(output.shape), sha256={
                  file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                  for file in (args.checkpoint, args.onnx, args.vectors)})
args.report.write_text(json.dumps(report, indent=2)+'\n')
assert report['passed'], error
print(json.dumps(report))
