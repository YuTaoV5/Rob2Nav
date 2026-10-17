# Third-party notices

## NJU quadrupedal-agility

- Source: https://github.com/NJU-RLC/quadrupedal-agility
- Revision: `15d16ea99bc23e5d401b0b0e3b88a8edcc28f0ed`
- Subset location: `vendor/NJUAgility/`
- License: MIT; see [upstream license](vendor/NJUAgility/LICENSE) and notices retained in individual files.
- The subset includes unchanged BBC networks, a normalization utility, and two upstream deployment checkpoints.
- [Provenance](vendor/NJUAgility/PROVENANCE.md) and SHA-256 snapshot identify the files; `python scripts/check_nju.py` verifies them.
- Rob2Nav's sensing, IK, navigation and switching code are independent additions. No NJU checkpoint weights were trained or updated in this integration.

## RoboGauge

- Source: https://github.com/wty-yy/RoboGauge
- Revision: `add2c7ee0185ea8fd93d242f86e1c8cac6e9913b`
- Location: `vendor/RoboGauge/`
- License: MIT; see [the original license](vendor/RoboGauge/LICENSE).
- Copyright: 2025 tianyang wu.

The source snapshot includes upstream robot meshes, model checkpoints, documentation and assets as distributed by RoboGauge. Original files and notices are preserved. The root project license does not replace any third-party notices. Additional dependencies retain their respective licenses.

The checkpoint in `artifacts/outputs/ppo_initial/navigation.zip` was produced by this project's initial navigation training run. It is distinct from the upstream low-level locomotion checkpoints.

The checkpoint in `artifacts/terrain/residual_model/locomotion_residual.zip` was produced by this project's joint-residual training run above the frozen RoboGauge prior; it is distinct from both the upstream checkpoints and the navigation PPO.
