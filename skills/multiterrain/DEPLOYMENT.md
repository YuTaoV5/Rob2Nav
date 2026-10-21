# Go2 multiterrain: SDK2 deployment plan

Plan only. No robot was connected, no hardware commands were sent, and no physical validation is claimed. This plan concerns the Go2 body and SDK2; no arm is required. The terrain perception adapter and low-level sending loop still need implementation before hardware use.

First pass [simulation](SIMULATION.md), fix the checkpoint/source hashes and retain failures. Install the official SDK on the robot control computer (installation only):

```bash
python3.11 -m venv ~/venvs/rob2nav-go2
source ~/venvs/rob2nav-go2/bin/activate
python -m pip install numpy scipy torch
mkdir -p ~/src
cd ~/src
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
git checkout --detach 814556d15970dd2ecf1c9984e845ca02ab07e206
git rev-parse HEAD > ../sdk2-reviewed-commit.txt
python -m pip install -e .
ip addr
```

Record a reviewed SDK revision, Go2 model, firmware, local network interface and low-level control permission. SDK installation does not establish compatibility. Do not run a vendor trick menu as a substitute for deploying this learned model.

Implement this exact chain locally on the robot computer:

1. Decode LowState, timestamps, IMU and 12 motor states. Actor joint order is FL, FR, RL, RR, each hip/thigh/calf. SDK indices are [3,4,5,0,1,2,9,10,11,6,7,8]. Validate signs and quaternion convention against recorded states before any motor output.
2. Reproduce moe_perceptive.mdp.history (5x45) and observed-return elevation memory (33x21 heights, masks and ages). Actor dimension is 2304. Memory lifetime is 4 seconds. Use measured Go2 sensor returns transformed by calibrated extrinsics and synchronized odometry. No privileged map, fictional depth data, or zero-filled substitution counts as this model's observation contract. Built-in sensor coverage may differ; measure it and retrain if the observation contract cannot be reproduced on the Go2 body.
3. Load the complete PerceptiveActor state with the frozen MoE base and trained fusion branch; do not export only a new final linear layer. Use the exact source normalization and temporal-history initialization. Compare recorded-input inference against the simulation implementation before using output.
4. At 50 Hz produce 12 joint offsets scaled by 0.25 rad around [0.1,0.8,-1.5,-0.1,0.8,-1.5,0.1,1,-1.5,-0.1,1,-1.5]. Use the training PD contract KP20/KD0.5 in a bounded local control loop; map LowCmd motors by name/index, compute CRC and enforce torque, position and velocity limits. A 500 Hz sender must hold the last valid 50 Hz target, not rerun history updates at 500 Hz.
5. Ensure one low-level writer and release the sport motion service using the reviewed SDK mechanism. Add independent state/inference deadline checks, power/temperature checks, posture/foot-contact checks and an operator emergency stop. Choose and verify a protective standing/damping transition; the web server does not close the motor loop.

Deployment stages: read-only state recording -> offline inference parity -> suspended joint-order/sign checks -> flat-ground low-speed standing/walking -> one small terrain element -> fixed-layout repetitions -> sensor/connection fault injection. This task stops at the plan; it does not execute these hardware stages. Save model, SDK and calibration hashes, time-aligned states/targets, constraint violations and videos. A reliable terrain crossing needs independent real-world validation after the missing adapters are built.

Official SDK source: [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python). [Training](TRAINING.md) / [simulation](SIMULATION.md).

本次方案的官方SDK源码固定为 `814556d15970dd2ecf1c9984e845ca02ab07e206`；初次为源码核对，随后已在独立go2-sdk-check环境验证真实SDK消息与CRC；未初始化DDS、未连接机器人。统一部署框架仍在物理验证中，见最新handover及docs/UNIFIED_GO2_DEPLOYMENT.md。实机型号/固件兼容性仍按各技能方案在后续实施时审查。
