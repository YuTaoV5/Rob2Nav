# NJU Agility subset

Source: https://github.com/NJU-RLC/quadrupedal-agility

Commit: 15d16ea99bc23e5d401b0b0e3b88a8edcc28f0ed

Unchanged files: go2_deploy/module/{net_ll,estimator}.py,
bbc/rsl_rl/utils/utils.py (stored as module/utils.py), and
go2_deploy/models/ll_policy/{3_model,2_model}/model.pt (renamed bbc3.pt/bbc2.pt).
The upstream MIT license is retained. This is not the box-specific policy.
Adapters, switching logic and MuJoCo validation are Rob2Nav code, not upstream results.

The adapter uses the deployment hip scale (0.3), actual Go2 thigh ranges from
the NJU URDF, and a compatibility unpickler for the training-only Normalizer
objects. No checkpoint weights are updated. Experimental BBC results must not
be presented as reproduction of NJU's Isaac Gym / real-robot results.
