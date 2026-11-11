"""Run the strict velocity evaluator with additional read-only gait metrics."""
import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mjlab.envs as envs
from rob2nav_gait_metrics import GaitRecorder

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--gait-output", required=True)
parser.add_argument("--stage-seconds", type=float, default=10.)
known, _ = parser.parse_known_args()
args = sys.argv[1:]
i = args.index("--gait-output")
del args[i:i+2]
sys.argv = [str(Path(__file__).with_name("eval_velocity.py"))] + args
if "--audit-body-contact" not in sys.argv:
    sys.argv.append("--audit-body-contact")
OriginalEnv = envs.ManagerBasedRlEnv


class MeasuredEnv(OriginalEnv):
    def step(self, action):
        if not hasattr(self, "_gait_recorder"):
            self._gait_recorder = GaitRecorder(self, known.gait_output, known.stage_seconds)
            original = self.metrics_manager.compute_substep
            def observe():
                original()
                self._gait_recorder.sample()
            self.metrics_manager.compute_substep = observe
        result = super().step(action)
        self._gait_recorder.mark_resets(result[2] | result[3])
        return result

    def close(self):
        if hasattr(self, "_gait_recorder"):
            self._gait_recorder.close()
        super().close()


envs.ManagerBasedRlEnv = MeasuredEnv
runpy.run_path(sys.argv[0], run_name="__main__")
