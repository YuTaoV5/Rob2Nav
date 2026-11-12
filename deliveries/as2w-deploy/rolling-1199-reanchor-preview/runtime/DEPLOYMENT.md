# R1199 rolling preview deployment plan

## Model and control contract

- Policy runs at 50 Hz. ONNX uses float32 `obs [1,65]` and `actions [1,16]`. See the parity manifest for export details.
- The action order is FL, FR, RL, RR within each group: hips, thighs, calves, wheels. The first 12 values are converted to joint position targets as default position + 0.35 × wrapped action. The last four are converted to wheel speed targets as 16 × wrapped action. The reference simulator applies position, wheel-speed and total-torque limits.
- Preserve the 65-element observation construction and use previous **wrapped** action as history. Reset policy history, stand bias, wheel reference and latch counters on mode entry. See the evaluator for exact indexing and filtering.
- Intended rolling commands are longitudinal velocity and yaw rate. Lateral command is ignored. Only the 0.05 m/s straight-line cases have evidence for the revised wrapper.
- Start with fixed stand, zero command and calibration. Once measured joint error is below 0.14 rad after 12 s, freeze the calibrated offset and reset the wheel hold counter once. The reference latch requires quiet wheel and leg feedback. Permit motion after the fixed-stand checks hold continuously for 0.40 s. A timeout keeps the requested command at zero. On release, brake, hold wheel position and verify return to fixed stand.
- Keep continuous and straight stand bias, parking odometry, and the joint-settling check enabled. Parking common gain is 0.5. This revision uses zero additional wheel-speed damping.
- Use the joint names and limits in `as2w-official-reference.json` for comparison. Hardware slots, directions, zero offsets and available torque must be verified against the target robot before a hardware adapter can be qualified. Do not infer these values from policy order.

## Offline reproduction

From an AS2-W training runtime containing the pinned simulator assets and dependencies used for R1199, set `AS2W_RUNTIME` to that runtime root and execute `bash runtime/replay_mujoco.sh` from this package directory. The script runs the 12-case pushed diagnostic by default. The checkpoint and simulation wrapper can also be used independently for offline inspection.

Integration sequence is fixed-input replay of observation, raw ONNX output, wrapped action and final 16-axis targets, followed by matching MuJoCo and Isaac Lab tests and then read-only hardware shadow validation. The current preview does not establish hardware parity or clear the full rolling gate. Rollback is to the site's last verified fixed-stand/stop configuration, with policy entry disabled.
