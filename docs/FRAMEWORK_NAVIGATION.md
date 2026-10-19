# Deployment core and navigation integration

Updated 2026-10-20. Rob2Nav is being migrated to a deployment framework with Studio as a client. This release provides an independently importable Python core, executable source-policy MuJoCo backends, a lifecycle/command-gate abstraction, and separate deployment/navigation pages. It is not yet equivalent to LeggedSkillDeploy's complete hardware stack.

## Structure and executable commands

* `rob2nav/contracts.py`: reviewed policy observation, joint, action, PD and frequency contracts. PIE uses FL/FR/RL/RR per-leg joint order, 50Hz policy and 200Hz physics. LSD tracking uses its distinct joint permutation and resolves gains from each pinned configuration.
* `rob2nav/assets.py` + `deployment/upstream_lock.json`: fixed external-cache paths and SHA256 validation. Do not redistribute upstream code/checkpoints without the necessary license.
* `rob2nav/backends/source_mujoco.py`: real PIE and LSD source-policy inference, video and joint-state recording, plus resolved `io.json`. Studio's old inference entry point forwards to this core. No motor or DDS bridge is provided.
* `rob2nav/runtime.py`: transport protocol, lifecycle and body-frame timed velocity gate. Stale/nonfinite commands enter fault and request transport stop. This abstract gate is tested but is not yet the scheduler for every existing inference/training loop; do not claim an integrated hardware watchdog.
* `rob2nav/navigation.py`: a bounded static SE(2) geometry probe and independent instantaneous fixed-alpha CBF projection. This is original reference code, not SCAN or SEA reproduction.

Run from the repository with the backend's isolated Python environment:

```bash
python -m rob2nav profiles
python -m rob2nav simulate --policy pie --scene stairs --seconds 10 --output outputs/deployment/pie
python -m rob2nav simulate --policy lsd_backflip --scene empty --seconds 6 --output outputs/deployment/backflip
python -m pytest studio/tests -q
cd studio/frontend && npm ci && npm run build
```

PIE uses `../envs/go2-pie/bin/python`; LSD uses `../envs/go2-mapless/bin/python`. Preserve historical Isaac training dependencies. Serve Studio with the existing loopback-only API and SSH tunnel. Pages `#deployment` and `#navigation` consume shared-core APIs. Navigation parameters are bounded server-side; reference planning does not start ROS or hardware.

## Navigation source gates

Full immutable source audit: [navigation audit](project_audits/navigation.md).

SCAN main `348e8a590a50a5a6bbab8d8c6dcfd171f009be26` is Apache2, ROS Noetic/Ubuntu20.04. Its occupancy map, trajectory optimizer and route interface require a native process/ROS bridge. Its source simulator's trotting visualization is not dynamic Go2 contact validation. Native build, topic mapping, odometry frames and command tracking remain pending.

SEA main `560de73c094a750a6a358bc1920ebc9ef1d9f3c5` has no root license in the audited tree. Its 550-value navigation history and released low-level JITs do not provide a trained navigation checkpoint. RPLIDAR 41-ray 240-degree observations differ from D435i/Mid360. The audited CBF constructor defaults to180 degrees while its caller supplies only ray count: correct and regression-test the angular contract before any safety claim. Minute-scale training belongs to the authors' setup.

SCAN trajectory tracking and SEA navigation policy are alternative higher-level backends. Both must output timestamped body-frame velocities to an independently validated locomotion backend. Neither planner supplies autonomous stair-climbing or jumping joint skills merely by producing a path.

## Current optimization gates

Stairs-retention job `be9053facb374533` and platform-retention job `f1478e77469949e1` each completed 200 iterations and nine-case evaluations: flat/stairs/platform32/32 each; all six original narrow-stake cases0/32. These preserve tested dense-terrain skills; they do not show higher generalized success or optimal gait. Old V3/V4/V7 runs stay stopped unless a justified new design explicitly supersedes them.

The recurring `go2` thread task was reactivated at20-minute intervals. It must inspect actual jobs before any launch, evaluate typically500-iteration chunks, extend while matched independent evaluation improves, and assess the design after an initial5000-iteration budget. Three matched plateaus or a material retention regression trigger diagnosis and a changed design, not blind continuation or abandonment of the whole project. Record successes, falls, collision/foot contacts, tracking/energy/foot slip where available, independent seeds/layouts, selected checkpoints and failed attempts. Unknown gait metrics must remain unknown until implemented. Never claim global optimality.

Next priorities: deploy and verify this foundation; add independent-seed evaluation of retained models and gait metrics; implement native navigation contracts/bridges and bounded training adapters; connect actual sensor observations and locomotion; validate dynamic navigation with collision/stuck/timeout and no-shield baselines. Publish selected own weights and reproducible evidence, not all intermediate checkpoints. Simulation only.
