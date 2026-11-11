set -eu
p=$1
out=$2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
assets=assets/r237-train-root
"$p" code/train_as2w_scratch_bootstrap.py "$assets" runs/r267-hind-envelope/model_307200.pt "$out" \
  --random-actor --history --privileged-critic --envs 24 --steps 614400 --seed 271 \
  --stop-training --low-speed-course --stair-approach-curriculum --learn-omni --directional-flat \
  --flat-fraction .5 --flat-posture-weight .35 --flat-hind-envelope .55 --flat-pitch-weight .22 \
  --flat-contact-penalty .5 --flat-command-interval 1.6 --flat-command-max .15 \
  --stair-rises .04 .06 .06 .07 .08 --stair-hold-probability .55 --stair-hold-stage-max 3 \
  --rear-transfer-weight .6 --top-support-shaping .12 --stair-pair-balance .6 \
  --terminal-penalty 4 --learning-rate 3e-5 --exploration-std .25 --gamma .997 \
  --benchmark-demos runs/r257-learn-omni/teacher-train.npz runs/r262-asymmetric-critic/interrupted-teacher.npz \
  --flat-demos runs/r260-directional-omni/yaw-teacher-train.npz runs/r271-flat-signed-teacher/teacher-train.npz runs/r272-flat-corrections/teacher-train.npz --reference-reset-probability .25 \
  --bootstrap-replay-updates 1500 --replay-every 2 --replay-learning-rate 5e-5 --replay-epochs 3 --stop-demo-weight 3 &
train_pid=$!
for steps in 0 307200 614400; do
  while [ ! -s "$out/model_$steps.pt" ]; do
    kill -0 "$train_pid" 2>/dev/null || { wait "$train_pid"; exit 1; }
    sleep 10
  done
  "$p" code/eval_as2w_joint_posture.py "$assets" "$out/dynamic-$steps.json" "r272-$steps:raw:$out/model_$steps.pt"
  "$p" code/eval_as2w_platform_support.py "$assets" "$out/model_$steps.pt" "$out/eval-$steps.json" --no-motion-prior --count 4
done
wait "$train_pid"
