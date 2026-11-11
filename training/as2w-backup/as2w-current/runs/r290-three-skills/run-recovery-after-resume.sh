set -euo pipefail
d=runs/r290-three-skills
run="$d/getup-symmetric-formal"
resume_pid=$(cat "$d/getup-resume-825-scheduler.pid")
model="$run/model_1199.pt"
until test -s "$model" && unzip -tq "$model" >/dev/null 2>&1; do
  kill -0 "$resume_pid" || exit 1
  sleep 10
done
while kill -0 "$resume_pid" 2>/dev/null; do sleep 10; done
wait_room() {
  local limit
  limit=$(cat /sys/fs/cgroup/memory.max)
  if test "$limit" != max; then
    while test "$((limit-$(cat /sys/fs/cgroup/memory.current)))" -lt 8589934592; do sleep 20; done
  fi
}
wait_room
bash "$d/run-symmetric-recovery-final-only.sh" || echo 'Standard holdout evaluation requires review'
wait_room
bash "$d/run-recovery-side-holdout.sh" || echo 'Side holdout evaluation requires review'
wait_room
exec bash "$d/run-mixed-recovery-formal.sh"
