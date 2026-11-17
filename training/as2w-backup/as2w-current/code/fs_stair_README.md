# FS stair line — from-scratch blind stair training (research, deployable: false)

MuJoCo + SB3, **CPU only**. The GPU belongs to the Isaac line and must stay
untouched. Nothing here starts training without an explicit `--run`.

| file | role |
|---|---|
| `fs_stair_env.py` | the only place that defines stages, control wrapper, reward, milestones, failure accounting |
| `fs_stair_contract.py` | control-contract fields, `FIELD_ORDER`, fingerprint, version guard (stdlib only) |
| `fs_stair_replay.py` | window hashing, first-mismatch localisation, field-level diff, re-export |
| `fs_stair_train.py` | entry: dry-run / selfcheck / run / replay |
| `fs_stair_eval_flat.py` | standalone flat low-speed approach + stop evaluation |
| `configs/fs_stair_{1frame,zero_history,6frame}.json` | arms S / H0 / H6 |
| `fs_stair_{1frame,zero_history,6frame}.sh` | one copy-paste command per arm |

## Commands

```bash
# dry run (default; prints the pre-flight report and exits 0)
bash fs_stair_6frame.sh --out <dir>
bash fs_stair_6frame.sh --stage step1 --out <dir>
# selfcheck (PASS/FAIL, non-zero on any failure)
bash fs_stair_6frame.sh --selfcheck --out <dir>
# training, only with the operator's explicit authorization
bash fs_stair_6frame.sh --run --out <dir>
# offline replay diagnostic (never trains)
bash fs_stair_6frame.sh --replay --out <dir> --re-export-first-mismatch
```

`env -u LD_LIBRARY_PATH` is mandatory (the interpreter is
`/opt/isaaclab-venv/bin/python`; otherwise torch/mujoco fail with
`undefined symbol: _PyErr_SetLocaleString`). `torch.set_num_threads(1)` is
enforced at the top of `fs_stair_train.py`; without it the same seed produces a
different trajectory.

## Phase rules

**Phase 1 (`--phase baseline`, default)** is the minimal-variable run:

* deployment-consistent action wrapper: action clip ±6, leg scale 0.35, wheel
  scale 16, torque = PD law clipped to the 90 % budget (hip/thigh ±54, calf ±81,
  wheel ±13.5). Filter / rate-limit / delay are explicit and currently the
  identity (`alpha=1.0`, no rate limit, `delay=0`), which reproduces the pinned
  training env; every parameter is fingerprinted so adding a deployment filter
  later is a versioned change, not a silent drift.
* **nominal mass / inertia**, nominal friction, nominal gains, **no push**, no
  observation noise. Each switch is printed as `off` in the dry-run; `com_offset`
  and `observation_noise` are not implemented and are *rejected* if enabled.
* S / H0 / H6 share the same wrapper, the same initial-state distribution and
  the same reset rule.

**Phase 2 (`--phase disturbance`) is locked by default.** Unlocking requires all
of: `--phase disturbance`, exactly one `--disturbance {none,load,push}`,
`--allow-disturbance`, `--init-checkpoint <pt>`, and — for a formal `--run` —
`--eval-record <json>` whose `checkpoint_sha256` matches the checkpoint and
whose criteria block (`A_s1`, `B_s1`, `C_top`, `D`) is present. `load` and `push`
are never enabled together; each must pass on its own before any merge. Missing
any piece prints `PHASE2_LOCKED` and exits 0 on a dry-run, non-zero under
`--run`.

### Gate from baseline to disturbance (`PENDING_OPERATOR_RULING`)

The baseline does not auto-advance. On the six holdout seeds **28001–28006**
(with 6 seeds the resolution is 1/6, so an improvement counts only if it changes
at least 2 seeds):

| id | capability | measurable definition | command |
|---|---|---|---|
| `A_s1` | flat distance closed | `flat_approach` completed: approach line reached and a stable four-wheel stance held for `TOP_HOLD_S = 0.5 s` | `bash fs_stair_6frame.sh --stage flat_approach --run --out <dir>` |
| `B_s1` | front wheel ascends | at least one front wheel (FL/FR) credited with tread ≥ 1 (≥ 8 N sustained ≥ 60 ms) | `bash fs_stair_6frame.sh --stage step1 --run --out <dir>` |
| `C_top` | rear wheels follow | all four wheels credited with the top tread (`min(ever_levels) ≥ n_levels`) | `bash fs_stair_6frame.sh --stage continuous --run --out <dir>` |
| `D` | top hold | `course_completed` (all four top-loaded ≥ 0.5 s) and not terminated — the `mujoco_screen.py:295` criterion, unchanged | same run as `C_top` |

`TOP_HOLD_S` stays **0.5 s** and `top_stable_hold` stays the default promotion
signal. The alternative signals (`all_wheels_top`, `longest_all4_top_s`) exist
and are off. The diagnostic evidence (0/6 on 1/2/5 steps, 0.26 s max four-wheel
top load) means this signal will stall the ladder at stage 0 — that is a known,
printed consequence, not something the script silently works around.

## Triple-tap logging (one row per control step)

`<out>/tap/triple_tap.worker<rank>.jsonl` (stride 1, capped by `max_rows`).
Units are explicit: `target_wheel_*` rad/s, `target_leg_pos_*` rad, `tau_*` Nm.

* `target_wheel_pre_filter[4]` — `16*action[12:]` after the action clip, before
  the wheel limit/filter.
* `target_wheel_clipped[4]`, `target_wheel_dispatch[4]` — the latter is captured
  at the real `data.ctrl` assignment point (AST-asserted adjacency).
* `target_leg_pos_pre_filter[12]`, `target_leg_pos_dispatch[12]` — proves the
  0.35 leg scale and the `leg_low/leg_high` clip actually take effect.
* `tau_request_est_nm[16]` **ESTIMATED** (PD request),
  `tau_actuator_coord_nm[16]` = `data.actuator_force`,
  `tau_joint_qfrc_nm[16]` = `data.qfrc_actuator` (**simulated joint actuator
  torque**, dof space).
* `torque_budget_nm` = 13.5 is a **budget, never a measurement**.
  `transmission_qfrc_consistency` asserts
  `qfrc_actuator[dof] ≈ Σ gear_a · actuator_force_a` (tolerance 1e-9) and the
  readback reports every actuator's kind / gear / actuators-per-joint. In this
  model all 16 are direct joint motors with gear 1 and one actuator per joint, so
  `actuator_force` equals the joint torque (labelled MEASURED). Real-robot drive
  torque remains `not_measured`.
* `history_newest[60]` + `history_sha256` over the whole float32 window:
  the hash localises the first differing **policy step**; the field needs that
  step's raw window.

### Windows and replay

Windows are saved to `<out>/windows/*.npz` (+ `index.jsonl`) every
`window_every_steps`, on the first/last two steps of an episode, and for ±3 steps
around the first occurrence of each failure kind (max 200). Each window carries
`obs` (frames × 60, float32), `field_order`, `history_sha256`, `step_id`,
`episode_id`, `policy_step`, `scene_seed`, `reason` and the wheel targets.

```bash
# first step whose recorded hash no longer matches its stored window
bash fs_stair_6frame.sh --replay --out <dir> --re-export-first-mismatch
# field-level diff between a recorded and a reference window
bash fs_stair_6frame.sh --replay --out <dir> \
     --replay-window <recorded.npz> --replay-reference <reference.npz> \
     --re-export-first-mismatch
```

`offline_tamper_detection` proves the chain: one observation field of step 3 is
changed by 1e-3, the window hash mismatches, the first mismatching step is 3, the
re-export succeeds, and the diff names `joint_vel_RR_thigh` (index 33).

## Control-contract versioning

`control_contract.version` is `stair-dev-1`; `control_contract.fingerprint` is
the sha256 of the canonical payload covering the control parameters, the history
semantics, the observation `field_order`, the initial-state distribution and the
dispatch units (the fingerprint is per arm because the history definition is
part of it).

If the stored fingerprint no longer matches, the runner prints

```
CONTROL_CONTRACT_CHANGED -> bump control_contract_version and redo replay + baseline regression
```

and **refuses `--run`**. `--accept-contract-change` is accepted only together
with `--dry-run` or `--replay`, and every use writes
`<out>/contract_change_audit.jsonl`; combining it with `--run` is a non-zero
exit. Any change to the filter position/frequency, the rate limit, the delay, the
history fields/order, the action clip/scales, the torque dispatch or the initial
state distribution requires a version bump plus replay and baseline regression.

## Evaluation

* `fs_stair_eval_flat.py` — the stair policy on flat ground only: low-speed
  approach (progress, mean speed, lateral drift, upright, non-wheel contact) and
  the natural stop at the approach line (drift, speed over the last 1.5 s). Full
  flat driving (forward/reverse/turn/arc, braking distance, 10 s dwell) belongs
  to the rolling policy and is not evaluated here.
* After `--run`, `holdout_mujoco.json` is produced by the **unchanged**
  `mujoco_screen.evaluate` on seeds 28001–28006 (criteria: ordered levels,
  `longest*dt ≥ 0.5`, peak non-wheel impact < 50 N). `holdout.json` is the
  per-stage episode accounting used for the promotion gate.

## Evidence

Each experiment directory holds `inputs.sha256`, `run-config.json`, `cmd.sh`,
`dryrun.log` and `selfcheck.json`; a run adds `episodes.jsonl`,
`failures/<kind>.jsonl`, `tap/`, `windows/`, `holdout.json`, `holdout_mujoco.json`
and `ppo.zip`. No videos, no large logs.

Status: `research / deployable: false`. Questions that still need an operator
ruling are listed as `operator_decisions` (OD-1…OD-5) in every config.
