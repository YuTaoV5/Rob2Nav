# Training

## Recipe (2000-iteration turn-biased continuation, action-delay DR disabled)

```bash
bash training/launch2000fix.sh
```

The script verifies BOTH hashes before starting (fails closed):
- init checkpoint `model_1999.pt` sha256 `f54751faddc211b4cfa7498ac35e95e4f2617779bcdb8b3b5de5904c9da72985`
- environment `code/as2w_isaac_robust_env.py` sha256 `1af438db2ff3babc064a3a56bc8825880a5d3640b2d0f76174b7faa35233e24f`

## Effective config

| item | value |
|---|---|
| envs | 4096 |
| iterations | 2000 (1999 -> 3999) |
| save interval | 100 |
| command mix | hold 0.20 / straight 0.15 / spin 0.40 / arc 0.25 |
| policy period | 20 ms (dt 0.005 x decimation 4) |
| episode length | 32 s |
| leg kp / kd | 55 / 2 |
| wheel kp / kd | 0 / 6 |
| wheel effort limit | 13.5 Nm (0.9 x 15) |
| leg scale | 0.35, wheel scale 16.0, action clip 6.0 |
| action-delay DR | **DISABLED** (`as2w_action_delay_max_steps = 0`) |
| observation lag | retained, {0,1} |
| rewards | track_lin 1.5, track_yaw 1.5 (std .08), wheel_support -0.65, knee_margin -3 (0.10 m) |

## Source change shipped as a patch

`runtime/action-delay-disable.patch` is the ONLY source change. Rationale: `apply_actions()`
runs once per PHYSICS substep (4x per 20 ms policy step) under PhysX, so the depth-5
`DelayedTarget` ring was written 4 times per policy step and aliased; the nominal
"0-4 policy step (0-80 ms)" delay actually collapsed to a 0-20 ms substep-level delay with
zero delay whenever lag < 4. The patch defaults the delay to 0 and makes the maximum an
explicit, recorded config value.

**The change is NOT committed to `training/as2w-backup/as2w-current` in this package.**
The historical snapshot stays as-is; the patch must be applied by the recipe or the value
set explicitly, so that any reproduction states which environment source it used.

## Provenance

| item | sha256 |
|---|---|
| repo HEAD | `abde5f49330a467db800e7cdcfe6e991c88b4c9b` |
| AGENTS.md | `a6a4e5e09c5218741e2b4a6f627d9b16ebf757027bec9725dafec211459faad5` |
| train_r11.py | `27d6f9644de6fdc4741e60a4f764346759e44767bb11bcd68432ee64c549ee9a` |
| URDF | `696aec2c0915ec1e06bcc91a63104d8234c51c0eb94331fa5496bf426c51e926` |
| USD | `0c5786834c5deda135d3195062033dfe8312c9abf7daad689070925e89d3a55c` |
| MJCF (original, unmodified) | `bd86aae1d8047a7be67aef8340c1303950f136af5fb5506dfeb451a1c94d0ce2` |
| MJCF (r11 copy w/ exclude) | `dc8a4d33a687327e021fc6f655164aad67b1c25fa997a94fd81277cec38e8bd9` |
| output with delay DR ON (turn-pilot) | `3bef0e1496f2aa65c1b73b7fa2912d3ec6a81b140eab334a3490976a0f4e3494` |
| output with delay DR OFF (turn-fix) | `617b252ca3b3520ff4e77dcfbb03adcc9d9f61c6f8a45c85c61d24675ffd2339` |
