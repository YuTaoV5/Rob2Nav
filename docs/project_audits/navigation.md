# Navigation source audit

Audited 2026-10-20. Read-only source review; neither upstream stack was built or executed. Source checkout, a launch template, a browser demo, and a verified dynamic Go2 navigation integration are separate milestones.

## Source locks and roles

| Project | Audited main commit | License boundary | Role |
|---|---|---|---|
| [SCAN-Planner](https://github.com/wuyi2121/SCAN-Planner) | `348e8a590a50a5a6bbab8d8c6dcfd171f009be26` | Apache-2.0 root; preserve dependency notices | Classical spatial local planner + route guidance + local 3D occupancy |
| [SEA-Nav-Code](https://github.com/11chens/SEA-Nav-Code) | `560de73c094a750a6a358bc1920ebc9ef1d9f3c5` | No root license found. `training/legged_gym/LICENSE` and inherited file headers are BSD-3-Clause, not blanket permission for all novel modules/weights. Keep external until clarified. | Learned local velocity navigation, differentiable LSE-CBF, collision replay, frozen locomotion |

Pins came from live GitHub API recursive tree SHA. Cached web commit listings were older, especially SEA. Never silently resolve mutable `main` for a packaged deployment.

## SCAN actual contract

[README](https://github.com/wuyi2121/SCAN-Planner/blob/348e8a590a50a5a6bbab8d8c6dcfd171f009be26/README.md) tests Ubuntu 20.04 / ROS Noetic / catkin; Armadillo required. ROS2 is a separately maintained `ros2-community` branch, not evidence that main natively runs in a ROS2 Humble stack. Build is `catkin_make`; launch `scan_planner/run.launch` plus RViz.

[run.launch](https://github.com/wuyi2121/SCAN-Planner/blob/348e8a590a50a5a6bbab8d8c6dcfd171f009be26/src/planner/plan_manage/launch/run.launch) accepts lidar OR depth, rather than a verified joint D435i/Mid360 fusion interface. Inputs include timestamped body odometry, sensor pose, PointCloud2 or depth image and calibrated intrinsics/extrinsics. Real defaults: `/LIO/odom_vehicle`, `/LIO/odom_imu`, `/LIO/clouds_lidar`, or `/camera/aligned_depth_to_color/image_raw`. `cloud_is_world` differs between simulation and real operation; frame and timestamp conversion must be explicit. MID360 support means accepting its transformed clouds through the localization pipeline, not a bundled MID360 driver.

Navigation modes: 1 interactive goal, 2 multi-floor keypoints, 3 supplied reference path. Thus it can build a local map online, while long-range tasks still need localization and route/keypoint guidance; do not describe it as zero-map/zero-route autonomy.

[advanced_param.xml](https://github.com/wuyi2121/SCAN-Planner/blob/348e8a590a50a5a6bbab8d8c6dcfd171f009be26/src/planner/plan_manage/launch/advanced_param.xml) defaults: 5 cm voxel resolution; 10×10×5 m sliding map; body height 0.4 m; double-cylinder radius 0.25 m, offset 0.18 m; planning horizon 3.5 m; max velocity 0.75 m/s, acceleration 0.5 m/s². These are useful configurable profiles, not universal safety dimensions.

[closed_loop_controller.cpp](https://github.com/wuyi2121/SCAN-Planner/blob/348e8a590a50a5a6bbab8d8c6dcfd171f009be26/src/planner/plan_manage/src/closed_loop_controller.cpp) subscribes `planning/bspline` + body odometry and publishes `geometry_msgs/Twist` on `cmd_vel`, plus `planning/go2_execution_frozen`. Default tracking bounds vx 0.75, vy 0.35, yaw 1 rad/s. The provided `go2_kinematic_sim` / gait publisher and open-loop trajectory odometry do NOT validate contact dynamics, stairs locomotion or a learned Go2 policy. It contains no locomotion training weights. MoE/PIE must implement a separate command-tracking backend and independent dynamic tests.

[Project page](https://wuyi2121.github.io/SCAN-Planner/) / [paper](https://arxiv.org/abs/2606.19555) contain upstream demonstrations. Label them as upstream evidence, not Rob2Nav results.

## SEA actual contract

[Training README](https://github.com/11chens/SEA-Nav-Code/blob/560de73c094a750a6a358bc1920ebc9ef1d9f3c5/README.md): Python 3.8 + Isaac Gym Preview 4, custom `training/rsl_rl`, custom `legged_gym`. Training entry `python training/legged_gym/legged_gym/scripts/train.py --headless`; play equivalent `play.py`. This is incompatible with simply importing it into the current Isaac Lab/MuJoCo environment. An isolated legacy runtime is needed.

[Go2 config](https://github.com/11chens/SEA-Nav-Code/blob/560de73c094a750a6a358bc1920ebc9ef1d9f3c5/training/legged_gym/legged_gym/envs/go2/go2_pos_config.py) uses 2048 environments, 60 s episodes, navigation output 3 (vx, vy, yaw). Input is 10×55=550 values: 12 proprioceptive fields (gravity3, last locomotion command3, body linear velocity3, body angular velocity3), 41 log2 range beams, 2 delayed local-goal fields. It models perception delay and a 10-step history. It is not raw RGB/depth navigation.

The repository contains three lower-level JIT files in `ctrl_model/` (`body_latest.jit`, `encoder_latent.jit`, `encoder_vel.jit`), loaded by the environment to generate 12 joint actions. No trained navigation `.pt`/`.onnx` was found in the pinned tree. The deployment README requires users to separately supply nav and loco ONNX models. Three JIT locomotion components must not be advertised as a ready trained SEA navigation policy.

[DifferentiableSafeActorCritic](https://github.com/11chens/SEA-Nav-Code/blob/560de73c094a750a6a358bc1920ebc9ef1d9f3c5/training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py) combines a 16-dimensional history latent, nominal velocity and learned alpha with [ExactLSECBFLayer](https://github.com/11chens/SEA-Nav-Code/blob/560de73c094a750a6a358bc1920ebc9ef1d9f3c5/training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py). Shield changes planar vx/vy while passing yaw through; it uses range-based clearance with default radius0.15+margin0.05 m and damping1. It is not SCAN's full orientation-sensitive 3D body collision model. Safety guarantees must be scoped to the model and perception assumptions.

**Important source discrepancy:** Go2 training rays span ±120° in 6° steps (41 beams), and deployment documents the same range. `ExactLSECBFLayer` defaults to `fov_deg=180`, while its actor constructor passes only `num_rays`. Its assumed beam vectors therefore do not match the configured 240° observation sector. This needs a controlled fix and regression tests before connecting a shield to real or simulated dynamic navigation. Also observation construction clips ranges to5m, while sensor/deployment nominal max is3m; make normalization explicit.

[Deployment](https://github.com/11chens/SEA-Nav-Code/blob/560de73c094a750a6a358bc1920ebc9ef1d9f3c5/deployment/README.md) is ROS2 (Foxy/Python3.8 constraints documented), RPLIDAR A2M12 + BreezySLAM + external [quad_deploy](https://github.com/11chens/quad_deploy), `ros_base`, Unitree SDK2. `/rays`:41 beams, ±120°,0.1–3m,base_link,~10Hz. `/pose`:world x/y/yaw, startup origin. Control waits for fresh perception, then external nav/loco/FSM drives lowcmd. Perception stale>1s enters zero-command safe stop and requires explicit resume. Hardware launch is deliberately outside this audit and Rob2Nav simulation integration scope.

[Project page](https://11chens.github.io/sea-nav/) reports minute-scale training and demonstrations. These are author results; they do not imply equivalent speed on our runtime, with MoE substitution, or on stair/pit terrain.

## Actionable Rob2Nav integration design

1. **Runtime core owns execution; website is a client.** Typed robot state, sensor frames, nav goals, body-frame twist, policy observation/action schema, timed reset/history, model hashes, PD/joint order, command arbitration, lifecycle and safety state machine. No browser directly emits joint/SDK commands.
2. **Separate planner and skill ports.** `NavigationBackend.step(state, perception, goal) -> NavigationCommand(vx,vy,yaw,valid_until,frame,diagnostics)` feeds a validated locomotion `PolicyBackend`. SCAN trajectory tracker and SEA learned policy are interchangeable higher-level backends, not one concatenated neural network.
3. **SCAN adapter milestone:** pinned external source/container, ROS1 bridge, calibrated cloud/odom/goal mappings, trajectory/voxel telemetry, bounded twist into MuJoCo MoE. Verify kinematic protocol first, then dynamic collision/goal tests. Export launch/config without calling it executed.
4. **SEA adapter milestone:** legacy Gym environment check, 550-value contract test, ray angular and range consistency fix, source nav training/export, ONNX inference parity, MuJoCo command-tracking transfer with frozen MoE. Changing locomotion changes dynamics and invalidates an untested safety/efficiency claim.
5. **Shared navigation page:** route/goal, local map and body envelope, planned vs executed path, raw vs shielded twist, clearance/CBF intervention, sensor age/dropouts, replanning latency, success/collision/stuck/timeout/energy and seed. Show mode/source/pin, runtime availability, prerequisites, artifact/video and independent validation status. Simulation/dry-run/export are distinct actions.
6. **Evaluation gates:** narrow passage at multiple headings, table overhang (3D), clutter, dead end/backtrack, dynamic crossing, stale/noisy sensor, stairs with actual locomotion, route completion across independent seeds. Include no-shield and circle-envelope baselines. Do not count ideal-map/kine trajectories as sensor-driven contact-dynamics success.

Current audit delivers contracts and integration plan only. Native SCAN build, SEA navigation checkpoint training, ROS bridging, MoE closed loop, depth/lidar calibration and dynamic navigation evaluation remain unverified.
