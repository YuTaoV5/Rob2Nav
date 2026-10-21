# Unified deployment verification snapshot — 2026-10-21

These are offline protocol and real free-base MuJoCo results. No robot was connected. All shipping profiles remain hardware_qualified=false.

* `unified_atomic_trial_9304`: frozen v9, source robot asset, hybrid_pd; stand → MoE → sideflip5200 → MoE → stand, five commits and settled. Integrated roll -6.3597 rad. Handover torque step ≤0.3001 Nm; native dynamic action torque step 44.6787 Nm. Joint speed reached 1.08188 times the rated envelope. This is one research cycle, NOT a motor-safety or robustness pass.
* `unified_switch_trial_9209`: same v9, RoboGauge asset; stand_soft entry timed out, zero commits. Do not infer all profile pairs work from the source-asset cycle.
* `unified_switch_trial_9205`: older controller; five profile transitions passed, but the active reference filter suppressed native gait. It is historical integration evidence only.
* `unified_sdk_codec_hybrid.json`: real pinned SDK IDL/CRC encoding in an isolated environment, no SDK DDS initialization/publisher. This is not a real-time or robot certification.

Each run includes its summary, raw 500 Hz control trace and video. `verified_v9_sources/` preserves the exact deployment source/configuration exercised by 9304 and 9209, plus the full verification workspace hash manifest. Model binaries are referenced by SHA and are stored in the main release archive/vendor paths, not duplicated here. The external reference NPZ must be prepared from the pinned source as documented; it is not included.

Current editable framework contains subsequent changes (policy cadence, velocity command expiry, cache guards and sequence helper); the v9 positive result must not be attributed to these later changes. Latest local verification: core 9/9 tests, studio 62/62 tests, frontend TypeScript/Vite production build passed. Sequence service has no physical-device validation. Browser deployment-plan UI is source/build verified; this release does not assert that the server was redeployed with it.

Historical summary field `root_writes` incorrectly says torque-only even when `wire_mode=hybrid_pd`; preserve raw JSON. Interpret it as initialization only, then actuator commands without root stabilization forces or welds. Hybrid PD uses motor gains plus residual feedforward, avoiding double PD.
