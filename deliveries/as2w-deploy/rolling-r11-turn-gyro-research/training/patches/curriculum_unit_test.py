"""Prove the B-group curriculum actually samples the operator's mixture, without touching the GPU.

The A arm is training, so building a second Isaac environment would contend for the GPU.  Instead this
exercises the exact `_resample` code path of CurriculumFlatCommands on CPU with a minimal stand-in
object, and reports the realised category shares plus the per-category command statistics.

This proves the sampling MATHS and the category definitions.  It does NOT prove the class is wired
into the live env - that is asserted separately by the launcher, which requires the
"AS2W_AB_CURRICULUM ON" startup line to be present before the run is accepted as arm B.
"""
import os
import sys

import torch

sys.path.insert(0, "/root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current/code")

import as2w_ab_curriculum as C  # noqa: E402


class Fake:
    pass


def make(n_envs=200000):
    f = Fake()
    f.device = torch.device("cpu")
    f._frac = torch.tensor([
        C._f("AS2W_AB_TURN_FRACTION", 0.35),
        C._f("AS2W_AB_ARC_FRACTION", 0.25),
        C._f("AS2W_AB_STRAIGHT_FRACTION", 0.15),
        C._f("AS2W_AB_BRAKE_FRACTION", 0.15),
        C._f("AS2W_AB_HOLD_FRACTION", 0.10),
    ], dtype=torch.float32)
    f._frac = f._frac / f._frac.sum()
    f._count = torch.zeros(5, dtype=torch.long)
    f._n_total = 0
    return f


def main():
    torch.manual_seed(0)
    f = make()
    cmd, cat = C.CurriculumFlatCommands._resample(f, 200000)
    n = cmd.shape[0]
    names = ("pure_turn", "arc", "straight", "brake", "hold")
    print("目标占比: %s" % [round(float(v), 3) for v in f._frac])
    ok = True
    for i, nm in enumerate(names):
        m = cat == i
        share = float(m.sum()) / n
        want = float(f._frac[i])
        seg = cmd[m]
        if nm == "pure_turn":
            extra = "vx=%.3f±%.3f wz∈{%s} |wz|=%.3f±%.3f" % (
                seg[:, 0].mean(), seg[:, 0].std(),
                sorted(set(round(float(v), 3) for v in seg[:, 2].abs())),
                seg[:, 2].abs().mean(), seg[:, 2].abs().std())
            good = bool((seg[:, 0].abs() < 1e-6).all()) and \
                bool(set(round(float(v), 3) for v in seg[:, 2].abs()) <= {0.05, 0.10})
        elif nm == "arc":
            extra = "vx=%.3f±%.3f wz abs=%.3f" % (seg[:, 0].mean(), seg[:, 0].std(), seg[:, 2].abs().mean())
            good = bool((seg[:, 0] - 0.25).abs().max() < 1e-6)
        elif nm == "straight":
            extra = "vx∈[%.2f,%.2f] vy max=%.3f" % (seg[:, 0].min(), seg[:, 0].max(), seg[:, 1].abs().max())
            good = bool(seg[:, 0].abs().max() <= 0.25 + 1e-6 and seg[:, 1].abs().max() < 1e-6)
        elif nm == "brake":
            fwd = float((seg[:, 0] > 0).float().mean())
            extra = "vx=%.2f, 前进占比=%.3f" % (seg[:, 0].mean(), fwd)
            good = bool(abs(fwd - 0.5) < 0.01)
        else:
            extra = "全零=%s" % bool((seg.abs() < 1e-9).all())
            good = bool((seg.abs() < 1e-9).all())
        close = abs(share - want) < 0.01
        ok = ok and close and good
        print("  %-10s 实测占比=%.4f 目标=%.4f %s | %s %s"
              % (nm, share, want, "OK" if close else "MISMATCH", extra, "OK" if good else "BAD"))
    print()
    print("CURRICULUM_OK" if ok else "CURRICULUM_BAD")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
