set -eu
p=$1
out=$2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
assets=assets/r237-train-root
"$p" code/train_as2w_flat_scratch.py "$assets" runs/r267-hind-envelope/model_307200.pt "$out" \
  --random-actor --envs 12 --steps 614400 --seed 276 \
  --stop-training --learn-omni --directional-flat --flat-fraction 1 --flat-sagittal-only \
  --flat-static-fraction .20 --flat-command-max .12 --flat-command-interval 2.2 \
  --flat-posture-weight .35 --flat-hind-envelope .55 --flat-pitch-weight .22 --flat-contact-penalty .5 \
  --terminal-penalty 4 --learning-rate 3e-5 --exploration-std .12 --gamma .99 \
  --flat-demos runs/r271-flat-signed-teacher/teacher-train.npz \
  --bootstrap-replay-updates 1200 --replay-every 2 --replay-learning-rate 5e-5 --replay-epochs 3 --stop-demo-weight 3 &
train_pid=$!
"$p" code/eval_as2w_flat_sagittal.py "$assets" "$out/sagittal-baseline.json" --sagittal "r267:raw:runs/r267-hind-envelope/model_307200.pt"
for steps in 0 30720 153600 307200 614400; do
  while [ ! -s "$out/model_$steps.pt" ]; do
    kill -0 "$train_pid" 2>/dev/null || { wait "$train_pid"; exit 1; }
    sleep 5
  done
  "$p" code/eval_as2w_flat_sagittal.py "$assets" "$out/sagittal-$steps.json" --sagittal "r276-$steps:raw:$out/model_$steps.pt"
  "$p" code/eval_as2w_static_zero.py "$assets" "$out/static-$steps.json" "r276-$steps=$out/model_$steps.pt"
done
wait "$train_pid"
