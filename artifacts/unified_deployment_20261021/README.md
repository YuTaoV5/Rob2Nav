## Final cache-lifecycle regression: v14

`unified_repeated_trial_9315` uses the exact current deployment source, including CLI (AST comparison for every module in `release_acceptance.json`). It completes seven activations: stand → MoE → sideflip → MoE → sideflip → MoE → stand. Two explicit triggers and two separate inverted-to-upright motion windows are checked from raw traces. Total integrated roll -12.7138977 rad; handover torque step ≤0.300001 Nm; retained Prepared entries = 3. Evicted references are now closed off the control thread, not retained for the lifetime of the service. A separate test verifies closure and bounded metadata.

Current core tests: **18 passed**. SDK/geometry/offline records and v12/v13 heading tests below remain applicable; v14 does not modify their observation/control mathematics. The v14 full sequence regression covers the cache lifecycle change. Maximum speed ratio 1.08440 remains above the rated envelope, so no hardware qualification is granted. Long-duration hardware operation, timing and broad disturbance robustness are still unverified.

Reproduce from the documented CPU environment with `python -m rob2nav.deployment.verify_mujoco --asset source --atomic-cycle --sequence deployment/sequences/repeated_flip_return.json --max-seconds 120 --seed 9515 --output outputs/repeated_new --video`. Per-entry deadlines remain unchanged; 120 seconds only provides a total budget for the longer seven-activation sequence.

---

## Latest verification: v12/v13 supersedes earlier development results

The actual control modules in the working release match frozen v13 semantically (AST comparison in `release_acceptance.json`). Only subsequent CLI plan presentation differs; device execution is unchanged. Raw frozen sources preserve original bytes, including line endings. Model binaries remain referenced by their original SHA.

| Frozen version / run | Scenario | Result |
|---|---|---|
| v12 / 9212 | soft stand → stiff stand → MLP → MoE → stand | passed |
| v12 / 9312 | shared sequencer, stand → MoE → sideflip → MoE → stand, yaw 0° | passed |
| v13 / 9313 | same shared sequencer, initial yaw +30° | passed |
| v13 / 9314 | same shared sequencer, initial yaw -30° | passed |
| v13 / 9213 | five-profile gait/PD transition regression | passed |

All above handover command torque steps ≤0.300001 Nm and Kp steps ≤0.030001 per 2 ms control tick. Native learned gait executes without a permanent transition low-pass. Atomic flips rotate approximately one full turn and then return to MoE and stand without resetting the physical robot state. Initial heading tests are finite cases, not a general robustness certificate. Do not count their differing seeds as randomized terrain or payload trials.

The three heading trials reach joint speed ratios approximately 1.0842, 1.0851 and 1.0993. These violate the rated envelope: **research motion cycles passed, motor-speed qualification failed**. Native dynamic torque changes are separate from controlled handover changes. Sideflip remains quarantined by default; all hardware qualification flags remain false.

The gain `support_model_gain` is now explicit in the skill contract and qualification digest. Generic transitions retain bounded measured-torque/integral support (gain 0); the certified sideflip research entry uses model regularization gain 2. This fixes the low-PD equilibrium bias without removing the atomic entry's load stabilizer. v10 integral-only and v11 universally regularized failures are retained with full traces/videos and frozen sources.

17 core regression tests cover supported integral behavior, atomic trigger/contact requirements, velocity expiry, stale/obsolete output, pending atomic return, failed preparation retaining current policy, entry yaw alignment after turning, calibrated mappings and SDK send failure. Studio 62 tests passed. SDK v13 performs real IDL/CRC and rejects firmware identity changes without opening DDS. Model fingerprint test proves deterministic hashing and detection of compiled mass changes even with identical root XML bytes.

`verified_v10_sources/` through `verified_v13_sources/` contain the exact control/profile/sequence sources and complete workspace hash manifests. External motion references and binaries are identified by hashes; they are not redundantly copied. Qualification uses the compiled model plus MuJoCo version and root XML, not root XML alone. The SDK source checkout and imported Python package must agree; a False publisher result fails the sender.

Device testing, full hardware timing, sensor calibration, independent watchdog and motor-safe stunt weights remain outside the verified simulation evidence. No robot was contacted. See the unified deployment manual for the staged device procedure and rejection gates.


---

# Unified deployment verification snapshot — 2026-10-21

These are offline protocol and real free-base MuJoCo results. No robot was connected. All shipping profiles remain hardware_qualified=false.

* `unified_atomic_trial_9304`: frozen v9, source robot asset, hybrid_pd; stand → MoE → sideflip5200 → MoE → stand, five commits and settled. Integrated roll -6.3597 rad. Handover torque step ≤0.3001 Nm; native dynamic action torque step 44.6787 Nm. Joint speed reached 1.08188 times the rated envelope. This is one research cycle, NOT a motor-safety or robustness pass.
* `unified_switch_trial_9209`: same v9, RoboGauge asset; stand_soft entry timed out, zero commits. Do not infer all profile pairs work from the source-asset cycle.
* `unified_switch_trial_9205`: older controller; five profile transitions passed, but the active reference filter suppressed native gait. It is historical integration evidence only.
* `unified_sdk_codec_hybrid.json`: real pinned SDK IDL/CRC encoding in an isolated environment, no SDK DDS initialization/publisher. This is not a real-time or robot certification.

Each run includes its summary, raw 500 Hz control trace and video. `verified_v9_sources/` preserves the exact deployment source/configuration exercised by 9304 and 9209, plus the full verification workspace hash manifest. Model binaries are referenced by SHA and are stored in the main release archive/vendor paths, not duplicated here. The external reference NPZ must be prepared from the pinned source as documented; it is not included.

Current editable framework contains subsequent changes (policy cadence, velocity command expiry, cache guards and sequence helper); the v9 positive result must not be attributed to these later changes. Latest local verification: core 9/9 tests, studio 62/62 tests, frontend TypeScript/Vite production build passed. Sequence service has no physical-device validation. Browser deployment-plan UI is source/build verified; this release does not assert that the server was redeployed with it.

Historical summary field `root_writes` incorrectly says torque-only even when `wire_mode=hybrid_pd`; preserve raw JSON. Interpret it as initialization only, then actuator commands without root stabilization forces or welds. Hybrid PD uses motor gains plus residual feedforward, avoiding double PD.
