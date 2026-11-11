set -eu
p=$1
out=$2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
assets=assets/r237-train-root
base=runs/r277-flat-no-checkpoint
while [ ! -s "$base/yaw-teacher-v2.npz" ]; do
  kill -0 "$(cat "$base/yaw-teacher-v2.pid")" 2>/dev/null || exit 2
  sleep 5
done
python3 - "$base/yaw-teacher-v2.json" <<'PYGATE'
import json,sys
rows=json.load(open(sys.argv[1]))
for direction in (-1,1):
    assert sum(r["passed"] for r in rows if r["split"]=="heldout" and direction*r["target"][2]>0)>=4
PYGATE
"$p" code/train_as2w_flat_heading.py "$assets" "$out" --scratch-widths 128 128 128 \
  --envs 12 --steps 614400 --seed 276 \
  --stop-training --learn-omni --directional-flat --flat-fraction 1 --flat-sagittal-only --flat-turn-fraction .35 \
  --flat-static-fraction .20 --flat-command-max .12 --flat-command-interval 2.2 \
  --flat-posture-weight .35 --flat-hind-envelope .55 --flat-pitch-weight .22 --flat-contact-penalty .5 \
  --terminal-penalty 4 --learning-rate 3e-5 --exploration-std .12 --gamma .99 \
  --flat-demos "$base/teacher/teacher-train.npz" "$base/yaw-teacher-v2.npz" \
  --bootstrap-replay-updates 1200 --replay-every 2 --replay-learning-rate 5e-5 --replay-epochs 3 --stop-demo-weight 3 &
train_pid=$!
for steps in 0 30720 153600 307200 614400; do
  while [ ! -s "$out/model_$steps.pt" ]; do
    kill -0 "$train_pid" 2>/dev/null || { wait "$train_pid"; exit 1; }
    sleep 5
  done
  "$p" code/eval_as2w_flat_heading.py "$assets" "$out/sagittal-$steps.json" --turn "r278-$steps:raw:$out/model_$steps.pt"
  "$p" code/eval_as2w_static_zero.py "$assets" "$out/static-$steps.json" "r278-$steps=$out/model_$steps.pt"
done
wait "$train_pid"
