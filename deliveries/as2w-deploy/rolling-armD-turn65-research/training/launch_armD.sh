#!/bin/bash
set -u
R=/root/gpufree-data/Rob2Nav; BASE=/root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current; TRAIN=/root/gpufree-data/Rob2Nav/skills/as2w/rolling/train_armE_support2p0.py; INIT=/root/gpufree-data/as2w-runs/r11_fanziqi_lab/yaw-arms/armE-support-20261116-034514/r11-final.pt; OUT=/root/gpufree-data/as2w-runs/as2w-rolling-goal/results/train/armD-turn65-1k-20261116-145902/run; PARENT=/root/gpufree-data/as2w-runs/as2w-rolling-goal/results/train/armD-turn65-1k-20261116-145902
printf '%s  %s\n' 4bdf4d26b076d7c717e9c151946dd42f8b3cf9dbbf592c5d6aac1ed5f6615866 "$INIT" | sha256sum -c - || exit 9
export AS2W_R11_SAVE_INTERVAL=200
export AS2W_GYRO_KP=60 AS2W_GYRO_CAP=4 AS2W_GYRO_FF_GAIN=0
# operator 2026-11-16 table: TURN 40 / ARC 25 / (hold+straight+brake) 35 -> 15/10/10
export AS2W_AB_CURRICULUM=1
export AS2W_AB_TURN_FRACTION=0.40
export AS2W_AB_ARC_FRACTION=0.25
export AS2W_AB_HOLD_FRACTION=0.15
export AS2W_AB_STRAIGHT_FRACTION=0.10
export AS2W_AB_BRAKE_FRACTION=0.10
export AS2W_AB_TURN_DWELL_MIN_S=8 AS2W_AB_TURN_DWELL_MAX_S=12 AS2W_AB_TURN_DWELL_S=10
export AS2W_YAW_WEIGHT=3.0 AS2W_YAW_STD=0.05 AS2W_SYM_W=0.0 AS2W_SUPPORT_W=-2.0
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages:$BASE/code"
unset LD_LIBRARY_PATH
cd "$BASE"
cat > "$PARENT/resolved_config.txt" <<CFG
run=armD-turn65-1k
user_decision=2026-11-16 operator: save armC, stop it, start a fresh group from 4bdf4d26 for 1000 rounds with turning exposure restored to 65 percent
init_checkpoint=$INIT
init_sha256=4bdf4d26b076d7c717e9c151946dd42f8b3cf9dbbf592c5d6aac1ed5f6615866
num_envs=4096
iterations=1000
save_interval=200
mixture: turn=0.40 arc=0.25 hold=0.15 straight=0.10 brake=0.10  (turning total 0.65)
gyro: kp=60 cap=4 ff_gain=0 (in training loop, as A/B)
reward: yaw_weight=3.0 yaw_std=0.05 sym_w=0.0 support_w=-2.0
posture_bias=off ; normal_exit=four-wheel grounded return
KNOWN_DEFECT: the AS2W_AB_TURN_DWELL_* variables are PRINTED but never used by as2w_ab_curriculum.py - "long turns" were never actually implemented in armB or armC
CFG
exec /isaac-sim/python.sh "$TRAIN" --viz none --device cuda:0   --usd "$BASE/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"   --urdf "$R/training/as2w-backup/dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"   --out "$OUT" --num-envs 4096 --iterations 1000 --init-checkpoint "$INIT"
