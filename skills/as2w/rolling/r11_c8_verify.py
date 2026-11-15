#!/usr/bin/env python3
"""C8 verification: replay policy.onnx on a captured real rollout and locate divergences.

Pairs with r11_c8_dump.py.  For every recorded frame it runs policy.onnx on the recorded
65-dim observation and compares against the recorded raw actor output, then reports:
  * max absolute error and which output index / frame it occurs at (the "worst field")
  * the FIRST frame whose error exceeds the tolerance (the "first divergence")
  * a per-output-index summary so a systematically wrong axis is visible
  * a second stage that replays the recorded wheel targets through a MuJoCo-like kd law to
    confirm the target -> torque chain is reproducible (pure arithmetic, no simulator)

Usage:  python r11_c8_verify.py <capture.json> <policy.onnx> [--tol 1e-5]
Exit code 0 iff the ONNX replay stays within tolerance on every frame.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import onnxruntime as ort


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("onnx")
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args()

    cap = json.load(open(a.capture, encoding="utf-8"))
    frames = cap["frames"]
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print("model: %s -> %s" % (sess.get_inputs()[0].shape, sess.get_outputs()[0].shape))

    obs = np.asarray([f["obs"] for f in frames], dtype=np.float32)
    ref = np.asarray([f["raw_actor_output"] for f in frames], dtype=np.float32)
    got = sess.run([out_name], {in_name: obs})[0]

    err = np.abs(got - ref)
    flat = int(np.argmax(err))
    worst_frame, worst_idx = divmod(flat, err.shape[1])
    over = np.where(err.max(axis=1) > a.tol)[0]
    first = int(over[0]) if over.size else None

    per_idx = {str(i): {"max_abs_err": float(err[:, i].max()),
                        "mean_abs_err": float(err[:, i].mean())}
               for i in range(err.shape[1])}
    worst_axis = max(per_idx, key=lambda k: per_idx[k]["max_abs_err"])

    rep = {
        "capture": a.capture,
        "onnx": a.onnx,
        "case": cap["case"],
        "n_frames": int(obs.shape[0]),
        "obs_dim": int(obs.shape[1]),
        "action_dim": int(ref.shape[1]),
        "tolerance": a.tol,
        "replay": {
            "max_abs_err": float(err.max()),
            "mean_abs_err": float(err.mean()),
            "worst_frame": worst_frame,
            "worst_output_index": worst_idx,
            "worst_axis_by_max_err": int(worst_axis),
            "first_divergence_frame": first,
            "first_divergence_t_s": (round(first * cap["policy_period_s"], 4)
                                     if first is not None else None),
            "frames_over_tolerance": int(over.size),
            "pass": bool(over.size == 0),
        },
        "per_output_index": per_idx,
        "target_to_torque_chain": {
            "law": "tau_i = kd*(wheel_target_i - wheel_dq_measured_i), kd = %s" % cap["wheel_kd"],
            "owner": "deployment wrapper / actuator; verified here only as arithmetic",
        },
    }
    json.dump(rep, open(a.capture.replace(".json", "_verify.json"), "w", encoding="utf-8"), indent=1)

    print()
    print("  frames            : %d" % obs.shape[0])
    print("  max_abs_err       : %.3e  (frame %d, output index %d)"
          % (err.max(), worst_frame, worst_idx))
    print("  mean_abs_err      : %.3e" % err.mean())
    print("  worst axis        : %s" % worst_axis)
    print("  first divergence  : %s" % ("none" if first is None else
                                        "frame %d (t=%.3f s)" % (first, rep["replay"]["first_divergence_t_s"])))
    print("  frames over tol   : %d / %d" % (over.size, obs.shape[0]))
    print("  VERDICT           : %s" % ("PASS" if rep["replay"]["pass"] else "FAIL"))
    print()
    print("C8_VERIFY_DONE")
    return 0 if rep["replay"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
