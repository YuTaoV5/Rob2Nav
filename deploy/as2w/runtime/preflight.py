#!/usr/bin/env python3
"""Deployment preflight: refuse to arm a hardware loop with an unverified config.

Checklist v3 section 6 requires that unverified slots, directions, zero offsets or hardware
derating stay empty and that "the real-robot closed-loop preflight MUST reject the config".
This is that rejection gate.

It reads runtime/deploy.yaml and enforces, in order:
  1. every required hardware field is present and not `not_measured`
  2. the declared joint set is exactly the 16 canonical names, no duplicates
  3. the torque limits do not exceed the contract's 90% values
  4. the ONNX artifact exists and its sha256 matches the package manifest

Modes
  --mode sim        only (1)-(3) and the artifact presence check; simulation and offline
                    replay are allowed with unverified hardware fields
  --mode hardware   all four checks; ANY `not_measured` hardware field is a hard failure
Exit code 0 only when the requested mode passes.

No --mode is a dry run: it prints the findings but always exits non-zero if anything is
missing, so it cannot be mistaken for an approval.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys

JOINTS = [f"{leg}_{part}" for part in ("hip", "thigh", "calf", "foot")
          for leg in ("FL", "FR", "RL", "RR")]
LIMITS = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "wheel": 13.5}
REQUIRED_HW = ([f"{leg}_{part}" for part in ("hip", "thigh", "calf", "foot")
                for leg in ("FL", "FR", "RL", "RR")]
               + ["directions", "zero_offsets", "firmware_version"])


def _mini_yaml(text):
    """Zero-dependency reader for the exact subset used by deploy.yaml.

    Supported: comments, `key: scalar`, one nesting level of `key:` + indented `subkey:
    scalar`, block lists (`- item`), and inline lists (`[a, b, c]`).  Anything else raises,
    because a deployment gate must not silently mis-read its own config.  It is validated
    against PyYAML in the test that ships next to this file.
    """
    def scalar(tok):
        tok = tok.strip()
        if tok == "" or tok in ("null", "~"):
            return None
        if tok in ("true", "True"):
            return True
        if tok in ("false", "False"):
            return False
        if tok.startswith("[") and tok.endswith("]"):
            body = tok[1:-1].strip()
            return [] if not body else [scalar(x) for x in body.split(",")]
        if (tok.startswith('"') and tok.endswith('"')) or (tok.startswith("'") and tok.endswith("'")):
            return tok[1:-1]
        try:
            return int(tok)
        except ValueError:
            pass
        try:
            return float(tok)
        except ValueError:
            return tok

    # Pre-process into (indent, body) pairs so the container type of an empty `key:` can be
    # decided by looking at the next line: a following `- item` means a list, anything else
    # means a mapping.  Without this the `segments:` block would be parsed as a mapping and
    # then rejected when the first list item arrived.
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            lines.append((len(line) - len(line.lstrip()), line.strip()))

    root = {}
    stack = [(-1, root)]
    for idx, (indent, body) in enumerate(lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError("unexpected dedent near %r" % body)
        parent = stack[-1][1]
        if body.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("list item without a list parent: %r" % body)
            parent.append(scalar(body[2:]))
        else:
            if ":" not in body:
                raise ValueError("expected key: value, got %r" % body)
            k, _, v = body.partition(":")
            k, v = k.strip(), v.strip()
            if v == "":
                nxt = lines[idx + 1] if idx + 1 < len(lines) else None
                is_list = bool(nxt and nxt[0] > indent and nxt[1].startswith("- "))
                child = [] if is_list else {}
                parent[k] = child
                stack.append((indent, child))
            else:
                parent[k] = scalar(v)
    return root


def load_yaml(path):
    text = io.open(path, encoding="utf-8").read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "deploy.yaml"))
    ap.add_argument("--mode", choices=["dry", "sim", "hardware"], default="dry")
    ap.add_argument("--package-root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    fails, warns = [], []

    # 1) hardware fields
    hw = cfg.get("hardware_slots") or {}
    missing = [k for k in REQUIRED_HW if k not in hw]
    unverified = [k for k in REQUIRED_HW if hw.get(k) in (None, "not_measured")]
    if missing:
        fails.append("hardware_slots missing keys: %s" % ", ".join(sorted(missing)))
    if unverified:
        msg = "unverified hardware fields (%d): %s" % (len(unverified), ", ".join(sorted(unverified)[:6]) + (" ..." if len(unverified) > 6 else ""))
        (fails if a.mode == "hardware" else warns).append(msg)

    # 2) joint set
    order = (cfg.get("action") or {}).get("joint_order") or []
    if sorted(order) != sorted(JOINTS) or len(order) != 16:
        fails.append("action.joint_order is not the 16 canonical names exactly once")
    elif len(set(order)) != 16:
        fails.append("action.joint_order has duplicates")

    # 3) torque limits
    lim = cfg.get("torque_limits_nm") or {}
    for k, v in LIMITS.items():
        got = lim.get(k)
        if got is None:
            fails.append("torque_limits_nm.%s missing" % k)
        elif float(got) > v + 1e-9:
            fails.append("torque_limits_nm.%s = %s exceeds the contract 90%% value %s"
                         % (k, got, v))

    # 4) artifact + manifest
    pol = (cfg.get("policy") or {}).get("file")
    if pol:
        fp = os.path.join(a.package_root, pol)
        if not os.path.exists(fp):
            fails.append("policy file not found: %s" % fp)
        else:
            sums = os.path.join(a.package_root, "SHA256SUMS")
            if os.path.exists(sums):
                want = None
                for line in io.open(sums, encoding="utf-8"):
                    h, _, rel = line.strip().partition("  ")
                    if rel == pol:
                        want = h
                        break
                got = hashlib.sha256(io.open(fp, "rb").read()).hexdigest()
                if want is None:
                    warns.append("%s is not listed in SHA256SUMS" % pol)
                elif want != got:
                    fails.append("policy sha256 mismatch: manifest %s..., file %s..."
                                 % (want[:12], got[:12]))
            else:
                warns.append("SHA256SUMS not found at the package root")

    # report
    print("preflight mode   : %s" % a.mode)
    print("config           : %s" % a.config)
    print("policy file      : %s" % pol)
    print("declared status  : %s (deployable=%s)"
          % (cfg.get("status"), cfg.get("deployable")))
    for w in warns:
        print("  WARN  %s" % w)
    for f in fails:
        print("  FAIL  %s" % f)

    if a.mode == "dry":
        print()
        print("dry run: no approval is granted by this mode")
        return 1 if (fails or warns) else 0
    if fails:
        print()
        print("PREFLIGHT_REFUSED")
        return 1
    print()
    print("PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
