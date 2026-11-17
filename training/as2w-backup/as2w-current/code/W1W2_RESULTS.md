# W1W2 adaptation - delivered code and condensed results

Status: **research / deployable: false**. Nothing here is an accepted weight; no
real-robot work was done. Raw per-frame traces, checkpoints and logs stay outside
the repository (paths below).

## 1. Rolling line - wheel kd loaded comparison

Tooling:
* `w1w2_roll_isaac_loaded.py` - Isaac/PhysX loaded run; intercepts the real dispatch
  point (`set_joint_velocity_target_index`) and records the target each physics substep.
* `w1w2_roll_mujoco_loaded.py` - MuJoCo loaded run consuming the same physical-unit sequence.
* `w1w2_roll_compare.py` - frame-by-frame cross-simulator comparison.
* `w1w2_roll_seq.json` - the shared four-wheel target sequence (identical bytes for both simulators).
* `w1w2_roll_kd_evidence.md` - readback evidence and the torque accounting.

Result (3 kd groups x 2 standing initial states x 2 simulators, 7 cases per arm):
every arm is **4900 frames, cross-simulator target deviation 0, no in-step rewrite**.
Recommended **kd = 1.5** (Isaac common-mode delivery 0.89-0.91, all cases stop within
1.5 s, zero support loss, torque peak 16 % of the 13.5 N*m budget).

Structural finding, NOT fixable by kd: Isaac does not deliver differential wheel
velocity (target +-0.385/+0.769 -> measured diff 0.02-0.04, wz ~ 0) and does not
deliver low-speed common mode (target 0.581 rad/s -> measured ~0.03 while MuJoCo
delivers 0.98). Any acceptance that includes low-speed or differential turning will
disagree between the two simulators.

Evidence root (outside the repo): `/root/gpufree-data/as2w-runs/w1w2-loaded.4Pz917/`

## 2. Stair line - from-scratch single frame vs six frame

Tooling: `fs_stair_env.py`, `fs_stair_train.py`, `fs_stair_contract.py`,
`fs_stair_replay.py`, `fs_stair_eval_flat.py`, `configs/fs_stair_*.json`,
`fs_stair_{1frame,6frame,zero_history}.sh`, `fs_stair_README.md`.

Design: three arms share one wheel contract, reward, initial-state distribution and
step budget; only the observation width differs - S = 60 (single frame),
H0 = 360 (six-frame shape with the history slots pinned to zero), H6 = 360
(six real consecutive frames, 100 ms). H0 exists because zero-padding a narrow actor
is not bit-identical (BLAS reassociation, ~1e-7), so S vs H6 alone would confound
architecture with arithmetic.

Run: 200,000 environment steps per arm, CPU only (MuJoCo + SB3, 4 workers each),
`flat_approach` stage, evaluated on the six held-out seeds 28001-28006.

| arm | input | max_forward per seed (m) | mean | complete | terminated | worst upright |
|-----|-------|--------------------------|------|----------|------------|---------------|
| S  | 60  | 1.54 / 1.00 / 1.38 / 1.66 / 1.91 / 0.77 | 1.38 | 0/6 | 5/6 | 0.793 |
| H0 | 360 | 4.49 / 4.91 / 1.25 / 5.43 / 5.65 / 2.82 | 4.09 | 0/6 | 2/6 | 0.897 |
| H6 | 360 | 4.29 / 5.81 / 0.43 / 4.67 / 2.37 / 2.07 | 3.27 | 0/6 | 3/6 | 0.836 |

Prior 16,384-step baseline for the same arms: S 0.0004-0.0016 m, H6 0.09-0.25 m.
So at 200k the "stands still" degeneracy is broken by three orders of magnitude.

At the 1/6 holdout resolution (an arm counts as better only when it improves at
least two seeds): H0 beats H6 on 5/6 seeds, H0 beats S on 5/6, H6 beats S on 5/6.
**There is no evidence that the six-frame history helps.** All three arms still fail
the flat-approach criterion (`complete = 0/6`, best `top_hold_s` 0.02 s) because of
lateral drift (`lateral_out` in 4/6, 2/6, 2/6; `max_lateral_m` 0.44-0.66).

Evidence root (outside the repo): `/root/gpufree-data/as2w-runs/w1w2-stair-run/`
with `{1frame,zero_history,6frame}/{ppo.zip,holdout.json,episodes.jsonl,tap/,windows/,failures/,inputs.sha256,run-config.json}`
and `eval_selfcheck.json`.

## 3. Evaluation-path defects fixed before the run

`mujoco_screen.evaluate` builds a 58-dim observation internally; the caller must
append the two gait-clock dims through `observation_suffix`, otherwise a 60-dim
actor raises `1x58 vs 60x128`. The same function runs the 60-dim `state` actor to
produce `previous_action` even when `native_planner` overrides the targets, so a
valid 60-dim actor must always be supplied. It also models only the five-step stair
terrain and therefore cannot score `flat_approach` at all.

Fixes in `fs_stair_train.py`:
* `--eval-stage` now defaults to `same` (the training stage).
* `evaluate_with_sim` appends the gait-clock suffix and uses the **trained** actor
  for the single-frame arm, recording `policy_source`.
* `evaluate_with_sim` refuses `flat_approach` loudly (`FROZEN_SCORER_NOT_APPLICABLE`)
  instead of crashing or silently scoring the wrong terrain.

Verification: `fs_stair_eval_selfcheck.py` (run separately) -
`three_arms_same_schema` (8 summary keys / 45 record keys identical across arms),
`s_arm_uses_frozen_scorer` (real one-seed run on `step1`, evaluator string recorded),
`history_slots_actually_used` (zeroing the five older slots changes the action by 1.05).

## 4. Known caveats

* `approach_min_gap_m` is `None` on all three arms - the gap tracker never populated.
* Flat completion is derived from `promotion_complete(flat_stage=True) -> top_stable`;
  the assignment that sets `course_completed` on flat ground was not traced line by line.
* Nothing here is dual-simulator accepted, ONNX exported, or hardware validated.
