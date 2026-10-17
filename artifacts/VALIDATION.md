# Remote simulation validation

Date: 2026-10-17

Host: Ubuntu 24.04 / NVIDIA L40; isolated Python 3.11 environment.

RoboGauge revision: `add2c7ee0185ea8fd93d242f86e1c8cac6e9913b`.

| Run | Success | Collision | Fall | Timeout | Mean successful duration (s) |
|---|---:|---:|---:|---:|---:|
| empty | 1/1 | 0 | 0 | 0 | 17.40 |
| slalom_final | 3/3 | 0 | 0 | 0 | 19.93 |
| random_final | 10/10 | 0 | 0 | 0 | 24.85 |
| blocked_check | 0/1 | 0 | 0 | 1 | - |
| ppo_initial_eval | 0/3 | 3 | 0 | 0 | - |

The blocked scene intentionally has no feasible path; timeout is the expected outcome.

PPO initial checkpoint: 32,768 environment steps, 4 environments, seed 42. This is an initial training run, not a convergence claim. Held-out PPO evaluation seeds start at 200; reactive random-scene evaluation seeds start at 100.

Three simulation checks plus Stable-Baselines3 Gymnasium API/checkpoint reload checks were executed. See tests_final.log and the saved summaries for evidence.

Sensor geometry is idealized and mounting extrinsics are assumed. Odometry is supplied by simulator state. No real robot commands are sent.
