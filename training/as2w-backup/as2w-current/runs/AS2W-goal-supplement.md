# AS2-W acceptance goal supplement
Keep the existing three-policy objective and all acceptance gates. Use runs/rolling, runs/stepping, and runs/recovery for independent evidence.
Rolling: isolate release, brake, lock, stance transition and hold. Compare the same scenario with and without a parking disturbance before changing the transition. Replay the required joystick cases in both simulators.
Stepping: establish reference feasibility with residual zero, compare support transfer and joint clearance, then validate a learned candidate on flat ground and stairs.
Recovery: reproduce mirrored deep-side failures, inspect early support and rotation, and evaluate first stand and post-push recovery as separate timing measurements.
For reward changes, compare stage-level activation and weighted contributions, especially immediately before first failure. Count formal acceptance only with both simulators, matching policy identity, and the existing safety and delivery requirements.

## Rolling curriculum update (2026-11-12)

- The 1,200-iteration R1199 fine-tune is historical evidence. The current rolling recipe uses 1,500 iterations per run and starts from R1199 at 0.20 m/s with 384 environments.
- Forward and reverse share the speed tiers 0.20, 0.35, 0.60, 1.0, 1.5, 2.0, 2.5 and 3.0 m/s. Retain low-speed left and right turns and arcs. Unlock a higher tier only after the same candidate and runtime contract pass the current tier in both simulators, including joystick, safety, braking, 10-second hold and fixed-stand transitions.
- Use the 5 Hz first-order leg-residual filter in training, evaluation and eventual deployment, with matching reset behavior. Wheel targets bypass it. Keep the continuous low-yaw and post-release braking, parked drift and fixed-stand costs. Quantify raw actor saturation and jitter separately with the real-robot replay.
- Run source: GitHub 9dffc8c9055f9df3783af3ed6901da782d950f6f and the immutable snapshots and input hashes under runs/rolling/filtered-1500-stage020.dWUv9Y. The first assessment is queued on the terminal checkpoint. Continue the existing v3 delivery, hardware authorization and progress reporting requirements.

## Current formal rolling run (2026-11-12)

The operator requested 4,000 iterations and 4,096 training environments. This supersedes the previous 1,500-iteration run budget for the current run. The active run is runs/rolling/filtered-4000-env4096-stage020.ovvRL3 at the 0.20 m/s tier. The 3 m/s goal, stage unlock requirements, and independent simulator evaluation counts still apply.
