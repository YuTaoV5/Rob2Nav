# Third-party notices

## Original criticaldamped Go2 AME reproduction

`critical_ame` executes the external repository
https://github.com/criticaldamped/AME_Locomotion at revision
15d02c891273d3f4c9b55c9afe3073bcd2d9152f through isolated source paths.
The external repository and its robot assets retain their original notices;
they are not relicensed by Rob2Nav. This run starts from random Go2 network
initialization, not the upstream G1 weights, RoboGauge, or the NVIDIA rough
initializer used by the separate perceptive experiments. The bounded runtime
ray re-query and current reproduction limits are documented in
docs/CRITICAL_AME_REPRODUCTION.md and CRITICAL_AME_VALIDATION.md.

## AME-inspired experiment

`perceptive_rl/attention.py` is an independent implementation informed by the
architecture described in He et al., “Attention-Based Map Encoding for Learning
Generalized Legged Locomotion” (https://arxiv.org/abs/2506.09588).
SII-FUSC/AME_Locomotion and the official leggedrobotics/ame2_minimal were inspected
as references. Their encoder source and G1 checkpoints are not redistributed in
this project. The experiment differs from both implementations and is not an
official reproduction. New Go2 checkpoints retain the initializer provenance
and asset restrictions described below.

## Isaac Lab perceptive locomotion experiments

- Framework: [Isaac Lab 2.3.2](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2), BSD-3-Clause; installed externally rather than vendored.
- Optimizer: [RSL-RL 3.1.2](https://github.com/leggedrobotics/rsl_rl), BSD-3-Clause.
- The Go2 USD and external initial rough checkpoint are downloaded from NVIDIA's Isaac Lab asset server. NVIDIA assets and checkpoints retain their original terms; the root source-code license does not relicense them or derived weights.
- `artifacts/perceptive/manifest.json` identifies the external initializer URL and SHA-256, and separately identifies Rob2Nav's fine-tuned teacher and student checkpoints.
- The from-scratch teacher, fine-tuned teacher, student imitation and student PPO are distinct training stages. They do not update the vendored RoboGauge or NJU checkpoints.
- Extreme Parkour, Robot Parkour Learning and Vision Locomotion are methodological references, linked in [the experiment documentation](docs/PERCEPTIVE_RL.md). This extension does not claim to reproduce their full training methods or published performance.

## NJU quadrupedal-agility

- Source: https://github.com/NJU-RLC/quadrupedal-agility
- Revision: `15d16ea99bc23e5d401b0b0e3b88a8edcc28f0ed`
- Subset location: `vendor/NJUAgility/`
- License: MIT; see [upstream license](vendor/NJUAgility/LICENSE) and notices retained in individual files.
- The subset includes unchanged BBC networks, a normalization utility, and two upstream deployment checkpoints.
- [Provenance](vendor/NJUAgility/PROVENANCE.md) and SHA-256 snapshot identify the files; `python scripts/check_nju.py` verifies them.
- Rob2Nav's sensing, IK, navigation and switching code are independent additions. No NJU checkpoint weights were trained or updated in this integration.

## RoboGauge

The `moe_perceptive` experiment reconstructs the actual network embedded in the
vendored 137k deployment checkpoint and strictly reloads its actor/encoder
tensors before staged fine-tuning. It does not recover the upstream critic or
optimizer and is not a from-scratch run. The original artifact is unchanged.
The AME-inspired visible-map encoder and training integration are Rob2Nav
additions. Additional inspected go2_rl_gym revision and inherited notices are
recorded in [MoE provenance](vendor/moe_reference/PROVENANCE.md) and
[license](vendor/moe_reference/LICENSE).

- Source: https://github.com/wty-yy/RoboGauge
- Revision: `add2c7ee0185ea8fd93d242f86e1c8cac6e9913b`
- Location: `vendor/RoboGauge/`
- License: MIT; see [the original license](vendor/RoboGauge/LICENSE).
- Copyright: 2025 tianyang wu.

The source snapshot includes upstream robot meshes, model checkpoints, documentation and assets as distributed by RoboGauge. Original files and notices are preserved. The root project license does not replace any third-party notices. Additional dependencies retain their respective licenses.

The checkpoint in `artifacts/outputs/ppo_initial/navigation.zip` was produced by this project's initial navigation training run. It is distinct from the upstream low-level locomotion checkpoints.

The checkpoint in `artifacts/terrain/residual_model/locomotion_residual.zip` was produced by this project's joint-residual training run above the frozen RoboGauge prior; it is distinct from both the upstream checkpoints and the navigation PPO.
