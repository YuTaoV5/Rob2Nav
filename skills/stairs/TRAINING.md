# Go2 stairs: training reproduction

This skill uses the native Rob2Nav MoE terrain PPO implementation. It is not a new upstream scratch-training result. The shipped 137k RoboGauge MoE is frozen during the value/fusion stages; the retained recipe initializes from the V2 sensor-policy checkpoint. Completion and terrain success are separate checks.

Use Bash from the repository root on the configured server. Required runtime: Isaac Sim 5.1, IsaacLab 2.3.2, RSL-RL 3.1.2, PyTorch 2.7/CUDA 12.8. Follow [environment setup](../../docs/DEPLOYMENT.md) first. The native asset and MoE weight are shipped under vendor/RoboGauge; no G1 or Go1 weights are used.

```bash
cd /root/gpufree-data/Rob2Nav
export MOE="$PWD/vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt"
export INIT="$PWD/artifacts/moe_memory_v2_end/training/visible_moe_memory4_v2_seed42/model_999.pt"
test -f "$MOE"
test -f "$INIT"
sha256sum "$MOE" "$INIT"
/root/gpufree-data/envs/go2-mapless/bin/python -m moe_perceptive.check --checkpoint "$MOE" --output outputs/stairs_model_check.json
/isaac-sim/python.sh -m moe_perceptive.train --headless --device cuda:0 \
  --moe "$MOE" --initialize-from "$INIT" --kind stairs --course original \
  --num-envs 512 --iterations 200 --warmup 20 --fusion 180 --actor 0 \
  --seed 71 --memory-lifetime 4 --sensor-rig original --numeric-debug \
  --studio-config skills/stairs/studio_config.json \
  --output outputs/stairs_reproduction
cat outputs/stairs_reproduction/completed.json
ls -lh outputs/stairs_reproduction/model_199.pt
```

Output directories must not already exist. The per-skill studio_config.json explicitly fixes learning_rate=1e-5; preserve its hash with each run. Historical Studio runs can contain additional overrides, so this command does not claim bit-for-bit reproduction of their weights. Keep source manifests, phases.jsonl, completed.json, TensorBoard events, checkpoints, seed, and the actual invocation. The original pretrained MoE SHA256 is 2206659a6a23886446dcfc05d86888d6cdc9b8a4e31f44cbbc4e2d17ba3d6ead.

For a from-base research run omit --initialize-from and use 1000 iterations with warmup 50, fusion 300 and actor 300. That changes the experiment, and its final result must be scored independently. Never substitute a teacher-map policy for the sensor policy.

[Simulation](SIMULATION.md) / [SDK deployment plan](DEPLOYMENT.md).
