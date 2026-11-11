set -eu
out=$1
launcher=$2
while kill -0 "$(cat "$out/job.pid")" 2>/dev/null; do
 sleep 5
done
grep -q 'AS2W_ISAAC_PPO_COMPLETE 6144000' "$out/job.log"
test -s "$out/model_249.pt"
exec bash "$launcher" --out "$out" --num-envs 1024 --iterations 2500 --command-stage sagittal --zero-memory --extend-from "$out/model_249.pt"
