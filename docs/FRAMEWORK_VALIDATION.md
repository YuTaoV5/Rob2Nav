# Framework and navigation page validation

2026-10-20. Tests:34 Studio/core tests passed; TypeScript and Vite production build passed. Deployed API serves `/api/deployment`, `/api/navigation`, bounded `/api/navigation/probe`; browser confirmed separate navigation page, computed path, task submission and completed video. No hardware control was attempted.

The server is Ubuntu24.04 with ROS Jazzy; the audited SCAN main targets Noetic/Ubuntu20.04. Native SCAN and SEA integrations remain pending and are labeled accordingly. The core CLI executed a2-second LSD backflip inference smoke after migration (`outputs/framework_smoke/backflip_v2`), producing real physics/video and resolved IO metadata. This short smoke is a compatibility check, not a full-skill trial.

The navigation page submitted task `09f6f49cf6114151`: historical Rob2Nav rule navigation with RoboGauge MoE, simulated depth/lidar, slalom, seed801, max25s. It completed1/1 according to the existing task definition. This is one integration smoke, not a general navigation success estimate or SCAN/SEA result. Video and raw summary are in `artifacts/framework_navigation/native_moe_slalom`.

Static geometry reference probes for clutter, narrow passage, overhang and dead end are recorded separately in `artifacts/framework_navigation/*.json`. They use exact supplied boxes and SE(2) A*, not physical robot dynamics or native SCAN. The separately displayed fixed-alpha CBF output is not used to execute that path, is not SEA's differentiable adaptive shield and establishes no real-robot safety guarantee.

Platform-retention job `f1478e77469949e1`:200 iterations, completed marker present, no failed marker; nine cases completed with flat/stairs/platform each32/32 and six original stake cases0/32. Continued mixed task `72f9bb2d17384900` starts from this selected checkpoint with frozen base, unchanged observations/rig and500 iterations, seed74. Its results are pending. Independent evaluation seeds and gait-quality metrics are still needed before interpreting changes as generalized improvement. The suite now accepts explicit `--seed` while preserving801 as default.

The existing `go2` recurring thread task is ACTIVE at20-minute intervals, with evidence-based continuation, retention and plateau diagnosis instructions. It must continue substantive work rather than merely monitor a stopped process. It cannot establish or promise a global optimum.
