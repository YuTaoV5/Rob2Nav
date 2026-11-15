# AS2-W skills

## 训练与交付手册

- [AS2-W 训练守则](../../training/as2w-backup/as2w-current/AGENTS.md)
- [AS2-W 训练到部署交付清单 v3](../../training/as2w-backup/as2w-current/AS2-W-training-delivery-checklist-v3.md)
- [AS2-W 后双轮直立训练与部署手册](rear-two-wheel/README.md)
- [R4000 转向与 0–1 m/s 续训交接](rolling/r4000-turn-20k.md)

| Skill | Training recipe | Scope |
|---|---|---|
| Rolling | [formal.sh](rolling/recipes/formal.sh) | Straight rolling, turning, braking and fixed stand |
| Rolling, filtered low speed | [filtered.sh](rolling/recipes/filtered.sh) | Opt-in 0.20 m/s controller and reward smoke test |
| Rolling, turn-biased continuation | [turn-2000fix.sh](rolling/recipes/turn-2000fix.sh) | 2000 extra iterations from model_1999 with spin sampling 25%->40%; action-delay DR disabled; research status |
| Stepping | [formal.sh](stepping/recipes/formal.sh) | Omnidirectional stepping and stairs |
| Recovery | [formal.sh](recovery/recipes/formal.sh) | Rising from prone and side tilt |

Run the recipes through the [training entry point](../../training/as2w-backup/README.md). Each output directory records the Git revision, recipe and input hashes, and the effective command. The historical simulator and evaluation scripts remain in the [training snapshot](../../training/as2w-backup/as2w-current) for reproduction.

## Rolling evaluation

Compare reports from the same checkpoint and speed cap with the archived R1199 example below. The output retains each backend's safety and behavior gates, scenario counts, and Isaac Lab turn rows. Inspect the original contracts before comparing pass counts across different protocols.

```bash
D=training/as2w-backup/as2w-current/runs/r290-three-skills
python skills/as2w/rolling/compare_reports.py \
  "$D/rolling-knee-margin-formal/model_1199.pt" \
  "$D/rolling-knee-margin-fixed-exit-trace-1199" \
  "$D/rolling-knee-margin-fixed-exit-trace-isaac-1199"
```

## r11 rolling acceptance (research)

11-case frozen matrix, Isaac Lab side. The MuJoCo counterpart and the full
dual-simulator comparison live in
[rolling-r11-turn-research](../../deliveries/as2w-deploy/rolling-r11-turn-research/).

```bash
python skills/as2w/rolling/r11_acceptance_matrix.py \
  --usd <kit3-as2w.usda> --urdf <as2w.urdf> --out <dir> \
  --checkpoints m3998=<model_3998.pt> --seeds 7393,311 --device cuda:0
```

Status: hold, +-0.25 m/s straight and release braking pass; pure turn and
right arc do not. Do not treat a passing straight/brake row as acceptance.

## r11 gyro yaw feedback (research, 2026-11-16)

Bounded, state-free gyro yaw feedback applied on the wheel differential only. It raises the
Isaac acceptance matrix from 5/11 to a reproducible 6/11 (`arc_right` flips to PASS on both
seeds) without touching hold, straight or braking. It does **not** fix the left arc or the
four pure-turn cases.

- Delivery package: [rolling-r11-turn-gyro-research](../../deliveries/as2w-deploy/rolling-r11-turn-gyro-research/)
- Wrapper + acceptance matrix: [r11_acceptance_ff.py](r11_acceptance_ff.py)
  (`--yaw-kp 60 --ff-cap 4 --ff-gain 0` is the measured optimum; `--yaw-kp 0` restores the
  unmodified controller)
- Execution-chain probe: [r11_isaac_exec_probe.py](r11_isaac_exec_probe.py)
- Dual-caliber summary: [r11_caliber.py](r11_caliber.py)
- C8 replay capture / verify: [r11_c8_dump.py](r11_c8_dump.py), [r11_c8_verify.py](r11_c8_verify.py)

**Wrapper defects that cost real debugging time, documented so they are not reintroduced:**

1. The differential sign must be `S = (-1, +1, -1, +1)` for `(FL, FR, RL, RR)`. The earlier
   `(-1, -1, +1, +1)` is a *front/rear* couple and cannot produce yaw at all.
2. Any early-return guard must check **both** the open-loop gain and the gyro gain, otherwise
   `--yaw-kp` is silently skipped and the run reproduces the baseline exactly.
3. After starting a new arm, verify that the quantity it claims to change is actually non-zero
   (`ff_max > 0`, `caps > 0`, or a different checkpoint sha). Pass counts, printed config values
   and startup logs have each been misleading at least once in this work.
