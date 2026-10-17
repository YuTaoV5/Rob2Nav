# MoE reference and notices

The upstream training source inspected for this extension is
https://github.com/wty-yy/go2_rl_gym at
`30e74dc507bec7a642a8c98be26081f2c6f0822d`.
Its full license and inherited notices are retained in `LICENSE`.

The actual initializer is the older deployment artifact already distributed in
`vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt`, SHA256
`2206659a6a23886446dcfc05d86888d6cdc9b8a4e31f44cbbc4e2d17ba3d6ead`.
It contains actor and student MoE parameters, with 5x45 history, 8 latent experts,
32 latent values and a shared actor. It differs from current upstream module
names. `moe_perceptive/model.py` reconstructs this artifact's exact architecture
and loads every actor/encoder tensor strictly. `moe_perceptive/check.py` checks
multi-step and reset equivalence against the original artifact.

The original teacher encoder, critic, optimizer, RNG and training environment
state are not present in the deployment artifact. No claim of restoring its
original training run is made. The new critic and terrain branch are initialized
separately; all pre-existing actor/encoder weights are reused, then selectively
fine-tuned. The original artifact is never overwritten.
