# Go2 stairs: simulation reproduction

Run [training](TRAINING.md) first, or explicitly point CKPT at an archived model and record its hash. This manual runs the actual native PhysX dynamics evaluator; the navigation page is the separate live MuJoCo goal-navigation skill.

```bash
cd /root/gpufree-data/Rob2Nav
export MOE="$PWD/vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt"
export CKPT="$PWD/outputs/stairs_reproduction/model_199.pt"
test -f "$CKPT"
sha256sum "$CKPT"
/isaac-sim/python.sh -m moe_perceptive.evaluate --headless --device cuda:0 \
  --moe "$MOE" --checkpoint "$CKPT" --kind stairs --difficulty .5 \
  --num-envs 32 --seed 801 --memory-lifetime 4 --sensor-rig original --course original \
  --output outputs/stairs_evaluation/stairs_seed801
```

Add --video --enable_cameras to an invocation with a fresh output directory to record the first real episode. Repeat with seeds 802 and 803 and new output paths. Read every summary.json: successes/total, per-episode route_complete/failure, source and policy hashes. A native process exiting zero or a video with one successful moment is insufficient. Unknown-map, shallow-probe, and teacher-map diagnostics do not count as sensor-policy acceptance.

Mixed terrain is evaluated as the three explicit layouts above, not an invented --kind mixed evaluator flag. Keep failures and regressions. Historical evidence is under artifacts/capability_integration/retention_stairs_v2, retention_platform_v2 and artifacts/retention_seed802/803; compare the actual checkpoint/hash before borrowing a score. No arbitrary terrain or hardware reliability claim follows from these fixed layouts.

[Training](TRAINING.md) / [SDK deployment plan](DEPLOYMENT.md).
