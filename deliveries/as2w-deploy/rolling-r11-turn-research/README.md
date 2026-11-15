# AS2-W rolling · r11 turn research archive

**Status: RESEARCH. Turning does NOT pass on either simulator. Not deployable. R4 not run.**

This package archives the r11 rolling line after: candidate screening, a frozen 11-case
acceptance matrix on both Isaac Lab and MuJoCo, control/measurement chain locking, an
action-delay domain-randomisation fix, and two 2000-iteration continuations.

## Verified capability (frozen gates: hold 0.02, track band [0.70,1.30], support 0.95, upright 0.85)

| capability | range | Isaac Lab | MuJoCo |
|---|---|---|---|
| zero-command hold 10 s | - | PASS | PASS |
| straight forward | 0.25 m/s (track 0.86-0.91) | PASS | PASS (1.11) |
| straight reverse | 0.25 m/s (track 0.86-0.88) | PASS | PASS (1.04-1.08) |
| release braking | 0.25 m/s (|vx| 0.000-0.013) | PASS | reverse PASS / **forward FAIL (one stand state fell)** |
| pure turn left/right | +-0.05, +-0.10 rad/s | **FAIL** (best 0.43 / 0.39) | **FAIL / INVALID_STAGE1** |
| arc left | 0.25 m/s, +0.05 rad/s | **FAIL** (yaw 0.68) | **FAIL** (yaw 0.63-0.69) |
| arc right | 0.25 m/s, -0.05 rad/s | **FAIL** (yaw 0.01) | **FAIL** (yaw -0.04) |

Not run: 0.6 / 1.0 m/s, fixed load 0/+6 kg, latency / friction / push randomisation,
leg-wheel joint design, ONNX export and parity, fixed stand hand-over, link-loss recovery,
R3 read-only preflight, R4 real-robot closed loop.

## Layout

- `weights/` - selected checkpoint (m3998). ONNX not exported yet.
- `runtime/` - action wrapper description, 16-axis mapping, the action-delay-disable patch, eval scripts.
- `training/` - launch recipe with dual sha256 verification, effective config, provenance table.
- `evidence/` - C1-C8 status, Isaac and MuJoCo matrices, dual-sim comparison, chain replay numbers.

## Key finding

`arc_right` has almost no yaw on BOTH simulators (0.01 / -0.04) while `arc_left` is a
near-miss on both (0.63-0.69). A right-side differential-generation / symmetry defect is
the most likely shared cause; wheel-ground physics cannot explain a failure that is
identical on both engines.

## Boundaries

- Isaac side: 2 seeds. MuJoCo side: 3 deterministic stand states. NOT the same protocol.
- No original field logs, no credentials, no raw capture in this archive.
- The action-delay fix is shipped as a patch and is NOT committed to the historical
  `training/as2w-backup/as2w-current` snapshot.

Delivery basis: [AS2-W training rules](../../../training/as2w-backup/as2w-current/AGENTS.md) and
[delivery checklist v3](../../../training/as2w-backup/as2w-current/AS2-W-training-delivery-checklist-v3.md).
