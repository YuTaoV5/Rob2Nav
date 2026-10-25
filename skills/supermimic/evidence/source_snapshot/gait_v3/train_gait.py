"""Train a separate, auditable gait refinement run; old velocity task is unchanged."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_gait_v2 as gait
from go2_supermimic.supermimic import train as training, state as st

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--gait-config", required=True)
extra, remaining = parser.parse_known_args()
settings = gait.GaitSettings(**json.loads(Path(extra.gait_config).read_text()))
if not (.22 <= settings.body_height_target <= .34 and .1 <= settings.mapped_target_rate_margin <= 1.):
    raise ValueError("Invalid gait contract")
if any(value < 0 for name, value in asdict(settings).items() if name.endswith("weight")):
    raise ValueError("Penalty weights must be nonnegative")
if settings.mapped_target_rate_reduction not in ("mean", "max"):
    raise ValueError("Target-rate reduction must be mean or max")
sys.argv = [sys.argv[0]] + remaining
gait.register(settings)
training.Imagination = gait.imagination_class(settings)
OriginalTrainer = training.Trainer


class GaitTrainer(OriginalTrainer):
    def __init__(self, cfg):
        if cfg.task != gait.TASK or cfg.driver_policy or cfg.expert_policy:
            raise ValueError("This trainer only supports the blind gait task without external drivers")
        super().__init__(cfg)
        if "initialize_hold_targets" not in self.env_cfg.events:
            raise ValueError("Training reset-hold event missing")
        self.imag.gait_velocity_limits = gait.velocity_limits(self.env)
        original = self.env.step
        self.gait_steps = 0
        def step(action):
            # Snapshot pre-transition physical state only for the shared reward;
            # nothing is added to actor observations or written to physics.
            self.env._gait_previous_state = st.robot_state(self.env).clone()
            self.env._gait_previous_pose = st.planar_pose(self.env).clone()
            result = original(action)
            self.gait_steps += 1
            if self.gait_steps % 50 == 0:
                height = float(st.robot_state(self.env)[:, st.Z].mean())
                self.writer.add_scalar("gait/base_height_m", height, self.gait_steps)
                print("GAIT_PHYSICS", self.gait_steps, "height", round(height, 5), flush=True)
            return result
        self.env.step = step
        source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(gait.__file__), Path(__file__), Path(gait.velocity.__file__)]}
        self.record = {"task": cfg.task, "cfg": asdict(cfg), "gait_settings": asdict(settings),
                       "source_sha256": source, "initialize_hold_targets": True, "actor_dim": self.obs.shape[-1],
                       "actor_terms": self.terms, "actor_exteroception": False,
                       "velocity_limits": self.imag.gait_velocity_limits.cpu().tolist(), "capability_qualified": False}
        (self.log_dir/"gait_contract.json").write_text(json.dumps(self.record, indent=2))

    def run(self):
        super().run()
        env = self.env
        c = st.robot_consts(env)
        s1, pose1 = st.robot_state(env), st.planar_pose(env)
        s0, pose0 = env._gait_previous_state, env._gait_previous_pose
        analytic, _ = gait.reward(s0, s1, pose0, pose1, self.term.gen.chase_vel,
                                  env.action_manager.action, env.action_manager.prev_action,
                                  c.soft_limits, c.default_q, c.action_scale, self.imag.gait_velocity_limits,
                                  env.step_dt, settings)
        error = float((analytic-gait.real_reward(env, settings)).abs().max())
        assert error < 1e-5, error
        target = self.obs[:, -28*self.history:].reshape(-1, self.history, 28)
        assert torch.count_nonzero(target[:, :, 3:]) == 0
        report = {"training_completed": True, "shared_transition_reward_max_error": error,
                  "command_padding_zero": True, "initialize_hold_targets": True,
                  "capability_qualified": False}
        (self.log_dir/"gait_audit.json").write_text(json.dumps(report, indent=2))
        print("GAIT_AUDIT", json.dumps(report), flush=True)


training.Trainer = GaitTrainer
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    training.main()
