#!/usr/bin/env python3
"""Export policy.onnx for ONE checkpoint and prove parity against the torch actor.

This is the parameterised version of the export path that was already validated on the control weight
(byte-identical reproduction of sha 680e0c72...).  It exists because the previously shipped policy.onnx
belonged to the CONTROL weight, not to the selected checkpoint, and a deployment package must ship the
ONNX of the weight it actually deploys.

Key details that must not be changed:
  * dynamo=False  - torch 2.11's default dynamo exporter writes external data (a 1885 byte .onnx plus a
    174080 byte .data sidecar).  With dynamo=False the export is a single self-contained file.
  * opset 17, input "obs", output "action", dynamic batch axis.
  * the deterministic mean is exported; distribution.std_param is sampling-only and is not in the graph.

Usage:
  python export_policy_onnx_for.py --ckpt <path.pt> --out <dir> [--label NAME]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os

import torch
import torch.nn as nn


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="policy")
    a = ap.parse_args()

    ckpt_sha = sha256(a.ckpt)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]

    lin = {}
    for k, v in sd.items():
        if not k.startswith("mlp."):
            continue
        _, idx, param = k.split(".")
        if param in ("weight", "bias"):
            lin.setdefault(int(idx), {})[param] = v
    order = sorted(lin)
    print("linear layers:", [(i, tuple(lin[i]["weight"].shape)) for i in order])

    layers = []
    for n, idx in enumerate(order):
        w, b = lin[idx]["weight"], lin[idx]["bias"]
        m = nn.Linear(w.shape[1], w.shape[0])
        m.weight.data = w.clone()
        m.bias.data = b.clone()
        layers.append(m)
        if n != len(order) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers).eval()

    obs_dim, act_dim = net[0].in_features, net[-1].out_features
    assert (obs_dim, act_dim) == (65, 16), "unexpected interface %dx%d" % (obs_dim, act_dim)

    weights_dir = os.path.join(a.out, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    onnx_path = os.path.join(weights_dir, "policy.onnx")

    torch.manual_seed(0)
    x = torch.randn(4096, obs_dim)
    with torch.no_grad():
        y_torch = net(x)

    torch.onnx.export(
        net, (x[:1],), onnx_path,
        input_names=["obs"], output_names=["action"], opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        do_constant_folding=True, dynamo=False)

    import onnx
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)

    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"obs": x.numpy()})[0]
    max_abs = float(np.abs(y_onnx - y_torch.numpy()).max())

    onnx_sha = sha256(onnx_path)
    rep = {
        "checkpoint": os.path.abspath(a.ckpt),
        "checkpoint_sha256": ckpt_sha,
        "onnx": os.path.abspath(onnx_path),
        "onnx_sha256": onnx_sha,
        "onnx_bytes": os.path.getsize(onnx_path),
        "opset": int(m.opset_import[0].version),
        "input": {"name": "obs", "shape": [1, obs_dim], "dtype": "float32"},
        "output": {"name": "action", "shape": [1, act_dim], "dtype": "float32"},
        "comparison": {"samples": int(x.shape[0]), "max_abs_err": max_abs, "pass": bool(max_abs < 1e-4)},
        "exporter": "torch.onnx.export(dynamo=False), single self-contained file",
    }
    ev = os.path.join(a.out, "evidence")
    os.makedirs(ev, exist_ok=True)
    io.open(os.path.join(ev, "C2_ONNX_PARITY.json"), "w", encoding="utf-8").write(
        json.dumps(rep, indent=2, ensure_ascii=False))
    print("ONNX_SHA", onnx_sha[:16], "bytes", rep["onnx_bytes"], "PARITY_PASS:", rep["comparison"]["pass"],
          "max_abs_err %.3e" % max_abs)
    return 0 if rep["comparison"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
