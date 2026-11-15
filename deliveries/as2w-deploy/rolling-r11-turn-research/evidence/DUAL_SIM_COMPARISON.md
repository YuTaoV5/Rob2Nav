# Dual-simulator comparison (same candidate m3998, same 11-case matrix, same gates)

| case | Isaac Lab | MuJoCo | agreement |
|---|---|---|---|
| hold 10 s | PASS (vx -0.0002, wz +0.0012) | PASS | yes |
| straight_fwd_0.25 | PASS (vx 0.91) | PASS (vx 1.11) | yes |
| straight_bwd_0.25 | PASS (vx 0.88) | PASS (vx 1.04-1.08) | yes |
| brake_bwd_0.25 | PASS (|vx| 0.012) | PASS (|vx| 0.04) | yes |
| **brake_fwd_0.25** | PASS | **FAIL: one stand state fell (upright -0.988, support 0.000, 27 non-wheel contacts)** | **NO** |
| turn_left_0.05 | FAIL 0.19 | FAIL 0.55-0.68 | both fail |
| turn_right_0.05 | FAIL 0.09 | FAIL 0.29-0.34 | both fail |
| turn_left_0.10 | FAIL 0.20 | yaw 0.99-1.15 IN BAND but support 0.49-0.69 | different failure mode |
| turn_right_0.10 | FAIL 0.08 | yaw 0.67-0.74 borderline but support 0.77-0.79 | different failure mode |
| **arc_left_0.25** | FAIL (0.68 on 1 of 2 seeds) | **FAIL 0.63-0.69** | **both near-miss at ~0.65** |
| **arc_right_0.25** | FAIL 0.01 | **FAIL -0.04** | **both essentially no yaw** |

## Conclusions

1. `arc_right` has almost no yaw on BOTH simulators (0.01 / -0.04). This is the strongest
   cross-simulator failure feature and cannot be explained by wheel-ground physics
   differences alone; it points at the right-side differential generation / symmetry.
2. MuJoCo delivers far more wheel differential (exec 0.86-1.04) than Isaac (0.43-0.57),
   confirming the Isaac execution chain as the short board; but MuJoCo converts it to yaw
   at the cost of support collapse (0.10 tier support 0.49-0.79).
3. `brake_fwd` falling in one MuJoCo stand state is a safety finding and blocks any claim
   that braking is safe.

## Reporting caveat

Isaac side uses 2 random seeds; MuJoCo side uses 3 deterministic stand states. The two are
NOT the same statistical protocol and must not be compared as if they were.
