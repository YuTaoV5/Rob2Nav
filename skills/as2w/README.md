# AS2-W skills

- [工程教训清单（必读）](ENGINEERING_LESSONS.md) —— 本仓库已实际发生过并已修正的 9 类 38 条错误，含提交结论前的 18 项自查表。做仿真、评测、导出或部署前先读。

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
| Rolling, support-weighted continuation | [armE-support2.sh](rolling/recipes/armE-support2.sh) | 600 iterations from model_3998 with rolling_wheel_support -0.65 -> -2.0; MuJoCo 7/11 endpoint; Isaac 5/11; research status |
| Rolling, A/B gyro-loop comparison | [ab-gyro-loop.sh](rolling/recipes/ab-gyro-loop.sh) | Two 1000-iteration arms from the same init with gyro in the training loop: `ARM=A` keeps the armE sampling (equal-budget control), `ARM=B` switches to the 35/25/15/15/10 long-turn curriculum via [as2w_ab_curriculum.py](rolling/as2w_ab_curriculum.py). The recipe pins every input by SHA256 and fails on mismatch; **arm B is only really running the curriculum if the startup line `AS2W_AB_CURRICULUM ON` appears** |
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

## r11 diagnostic and reporting tools (research, 2026-11-16)

Entry points added while chasing the pure-turn gap. All are CPU-side where possible so they can run while a
training job holds the GPU; each writes a JSON next to its log.

| Tool | Purpose |
|---|---|
| [r11_dual_basis_report.py](r11_dual_basis_report.py) | Report an acceptance `ff_arm.json` under BOTH bases: min (every seed passes - the decision basis) and mean (mean tracking in band with the worst seed still supported). A case that passes mean but fails min is seed scatter and must not be promoted. |
| [r11_segmented_report.py](r11_segmented_report.py) | Turning report by window - first second, 3-5 s, last 3 s - with signed yaw, actor differential, measured differential, delivery ratio, common-mode wheel speed and four-wheel support. |
| [r11_joystick_seq.py](r11_joystick_seq.py) | Continuous joystick sequence, fixed stand -> reverse/forward -> release -> stop -> fixed stand, judged on decelerate, come-to-rest and restore separately. |
| [r11_mj_stand_decay.py](r11_mj_stand_decay.py) | Does the post-turn leg offset decay when the legs are just held at fixed stand, or is it a loaded PD equilibrium? |
| [r11_mj_bias_comp.py](r11_mj_bias_comp.py) | Constant posture bias (`target = 2*DEF - q_eq`, iterated) to pull the achieved stance back to fixed stand. |
| [r11_mj_bias_regression.py](r11_mj_bias_regression.py) | The regression the goal demands before any leg-pose change may be adopted: support, rear clearance, torque fraction of the 90% limits, non-wheel contacts, with and without the bias. |
| [r11_mj_estop.py](r11_mj_estop.py) | Instant e-stop - wheel target to zero in one step with the legs frozen - as distinct from a ramped release. |
| [r11_mj_step_reset.py](r11_mj_step_reset.py) | Net-zero step cycle as a way to restore standing; measured NOT to work with the current gait (see the delivery evidence). |

Honest status of this line: the posture bias cuts the post-turn standing residual from about 0.20 rad to
0.06-0.08 rad and passes the MuJoCo hard-protection regression, but the goal requires the same hard-constraint
basis on the Isaac side, which has not been run. Stepping as a reset does not work. The instant e-stop stops the
robot cleanly but saturates the wheel torque cap on every stop from speed, because `tau = -6*dq` exceeds the cap
at any normal speed - that is a pending ruling, not a fixed defect.
