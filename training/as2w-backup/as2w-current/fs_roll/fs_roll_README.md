# fs_roll -- AS2-W rolling from-scratch, arms A/B/C

Status: **research / deployable: false**. Prepared 2026-11-16/17. NOTHING was trained;
`--run` was never executed in the preparation phase.

Execution snapshot + evidence: `/root/gpufree-data/as2w-runs/fs-roll-20261116-2345/`
(`dryrun.log`, `selfcheck.json` 56 checks, `run-config.json`, `inputs.sha256`, `per_arm/`,
`triple_tap_sample.jsonl`, `history_window_sample.npz`, `offline_forward_samples.npz`).
The copies here are byte-identical to that snapshot; sha256 is in `inputs.sha256`.

## Experiment

One wheel contract / reward set / command source / env-step budget; only observation and
network differ. A=65, B=390, C=66 (65 + vx_hat).

| arm | observation | actor input | actor params | critic | estimator | total |
|-----|-------------|-------------|--------------|--------|-----------|-------|
| A | single 65-D frame | 65 | 43,536 | 42,241 | - | 85,793 |
| B | last 6 frames concatenated | 390 | 85,136 | 83,841 | - | 168,993 |
| C | 6 frames -> MLP -> vx_hat; actor sees frame + vx_hat | 66 | 43,664 | 42,369 | 66,689 | 152,738 |

Parameter counts are counted from the built modules (not hand-computed). Hidden size
[128,128,128] ELU for all three actors.

## Control contract (versioned)

* `control_contract_version = dev-1`, `contract_fingerprint` = canonical hash of
  filter / rate-limit / delay / reset / history-field semantics (field order participates).
* Deployment-consistent wrapper: no filter, no rate limiter, zero policy-step delay, reset
  rules, 13.5 N*m pre-dispatch clamp. Same instance/parameters for A/B/C.
* **Rule: any change to filter position, filter frequency, delay or history field semantics
  -> bump `control_contract_version` + redo replay + baseline regression.** If the fingerprint
  changes without a version bump, `--run` refuses with
  `CONTROL_CONTRACT_CHANGED -> bump control_contract_version and redo replay + baseline regression`.
* `--accept-contract-change` is a **dry-run / replay diagnostic only**; combined with `--run`
  it exits non-zero (use only after a version bump, with replay + baseline regression attached).

## Gates run before Isaac

The `--run` path resolves every authorisation gate on pure CPU **before** importing `isaaclab`:
contract-change guard, then phase-2 (checkpoint + `--allow-disturbance` + `--eval-record` sha
match). A refused `--run` prints `RUN_GATE_REFUSED ...`, `ISAAC_IMPORTED=no` and `KIT_LAUNCHED=no`
and exits non-zero; `KitLauncher initialization complete` must never appear. The CLI self-check
`refusal_never_imports_isaac` runs four refused `--run` combinations in subprocesses and asserts
that property.

## Phases

`--phase baseline` (default): from-zero training with the deployment-consistent wrapper, nominal
mass, no push, and **every other randomisation off** (friction / motor / observation noise /
action delay are printed item by item as phase-2 candidates).

`--phase disturbance` is **locked by default**. `--run` requires all of:
`--init-checkpoint`, `--allow-disturbance` (operator authorisation) **and** `--eval-record <json>`
whose `checkpoint_sha256` matches the checkpoint and which carries the held-out gate fields
(`tracking`, `attitude`, `contact`, `brake`, `seeds`, `evaluator`). Missing any -> refuses and
prints `PHASE2_LOCKED` with the missing item; a sha mismatch prints both shas. Dry-run prints
`PHASE2_LOCKED` and exits 0.

`--disturbance load` (deterministic `--load-kg` on base_link) and `--disturbance push`
(velocity push sampling) are **one at a time**; both together is an error.

**Merge rule: load and push must each be run separately and pass before a merged (load+push)
configuration is allowed. Never start with load+push together.**

## Baseline -> disturbance gate (PENDING_OPERATOR_RULING)

Not automatic. After the three `baseline` arms finish, run the existing held-out evaluator --
`deliveries/as2w-deploy/rolling-armD-turn65-research/runtime/r11_acceptance_ff.py` or
`skills/as2w/rolling/r11_acceptance_matrix.py` (**no new evaluator, no changed scoring**) -- and
only a pass allows a load/push continuation from that checkpoint. The exact pass thresholds and
seed set are **not frozen**; status `PENDING_OPERATOR_RULING`.

## Curriculum

`--curriculum armE_mixture` (default): armE mixture, vx cap 0.25 m/s.
`--curriculum staged_0_to_3`: 8 stages 0.05/0.10/0.25/0.50/1.00/1.50/2.00/3.00 m/s with
per-stage dwell/share; each stage must pass track + attitude + contact + brake before the next
unlocks; hard failure demotes one stage and clears the streak; the top stage never advances.
High-speed arcs (stages 5..7) are marked `needs_separate_validation` and blocked until opened.
Gate thresholds are `PLACEHOLDER_PENDING_OPERATOR_RULING` (AGENTS.md gives qualitative gates only).

## Triple-tap logging

One JSON object per policy step: `env_id, episode_id, policy_step, target_wheel_pre_filter[4]
rad/s, target_wheel_dispatch[4] rad/s, target_leg_pos_pre_filter[12] rad, target_leg_pos_dispatch
[12] rad, history_newest[65], history_sha256, dtype, field_order, window_shape, torque_budget_nm,
torque_budget_location, tau_request_est_nm[4] ESTIMATED, tau_solver_nm (not_measured)`.

* `target_*_dispatch` is read at the real dispatch point (`last_dispatch` set immediately before
  `set_joint_velocity_target_index` / `set_joint_position_target_index`).
* Torque is a separate group: 13.5 N*m budget (pre-dispatch clamp + `effort_limit_sim`), an
  ESTIMATED request `kd*(dispatch-dq)`, and `tau_solver_nm` which is `not_measured` (PhysX
  implicit drive does not expose applied joint effort). Estimated torque is never written into
  the measured column.
* Raw 6x65 windows are sampled (every `--history-sample-stride` steps + episode ends) to
  `history_windows.npz`; `--replay --re-export-first-mismatch <run_dir>` re-exports the window at
  the first ledger mismatch with a field-level diff (field name + index + delta).

## Commands

```bash
bash fs_roll_A.sh                 # dry-run (default): pure CPU, no Isaac, no GPU
bash fs_roll_B.sh
bash fs_roll_C.sh
bash fs_roll_A.sh --run           # training: OPERATOR ONLY
```

## Pending validation (do not read as verified)

* contract group selection `pending_validation`; USD drive gains `not_measured`.
* 13.5 N*m is a software clamp + `effort_limit_sim`; solver torque is `not_measured`.
* 16384 envs GPU memory is a linear extrapolation (27,724 / 46,068 MiB, 60.2%), not measured.
* staged curriculum runtime path and disturbance runtime path were never executed.
* phase-2 pass thresholds/seeds are `PENDING_OPERATOR_RULING`.
* wheel radius / odometry not used.
