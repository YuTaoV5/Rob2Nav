"""Exact batched RoboGauge walk_moe adapter, distinct from go2_rl_lab exports."""
import hashlib
import json
from pathlib import Path
import torch


class RoboGaugeHead:
    def __init__(self, policy, num_envs, device):
        self.policy = torch.jit.load(str(policy), map_location=device).eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.history_length = int(self.policy.history_length)
        self.mask = self.policy.obs_no_goal_mask.to(device)
        if self.mask.numel() != 45 or self.history_length != 5:
            raise ValueError("Unexpected RoboGauge history contract")
        self.history = torch.zeros(num_envs, self.history_length, 45, device=device)

    @torch.no_grad()
    def __call__(self, observation):
        x = self.policy.normalizer(observation)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        no_goal = self.history[:, :, self.mask].reshape(len(x), -1)
        latent, _ = self.policy.student_moe_encoder(self.history.flatten(1), no_goal)
        return self.policy.actor(torch.cat([latent, x], dim=1))

    def reset(self, done):
        self.history[done] = 0


class RoboGaugeTeacher:
    def __init__(self, env, profile):
        from go2_supermimic.supermimic import state as st
        self.env = env
        self.profile_path = Path(profile).resolve()
        self.contract = json.loads(self.profile_path.read_text())
        c = self.contract
        self.path = self.profile_path.parents[2]/c["policy"]
        actual = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if actual != c["sha256"] or c["observation"] != "robogauge45" or c["policy_hz"] != round(1/env.step_dt):
            raise ValueError("Teacher hash/observation/control contract mismatch")
        self.robot = env.scene["robot"]
        names = list(self.robot.joint_names)
        self.ids = torch.tensor([names.index(name) for name in c["joints"]], device=env.device)
        self.default = torch.tensor(c["default"], device=env.device)
        self.head = RoboGaugeHead(self.path, env.num_envs, env.device)
        self.last = torch.zeros(env.num_envs, 12, device=env.device)
        self.c = st.robot_consts(env)
        self.calls = 0
        self.mapping_error = 0.
        self.metadata = {"policy": str(self.path), "sha256": actual, "history_length": 5,
                         "history_layout": "time-major normalized 5x45, same as original TorchScript",
                         "command_scale": [2.,2.,.25], "teacher_joints": c["joints"],
                         "native_kp": c["kp"], "native_kd": c["kd"],
                         "collection_kp": self.c.kp.cpu().tolist(), "collection_kd": self.c.kd.cpu().tolist(),
                         "collection_dynamics": "unchanged SuperMimic actuators; native MoE gains are metadata only"}

    @torch.no_grad()
    def act(self, command):
        d = self.robot.data
        clipped = torch.maximum(torch.minimum(command, command.new_tensor([1.,1.,2.])), command.new_tensor([-1.,-1.,-2.]))
        obs = torch.cat([d.root_link_ang_vel_b*.25, d.projected_gravity_b,
                         clipped*command.new_tensor([2.,2.,.25]), d.joint_pos[:,self.ids]-self.default,
                         d.joint_vel[:,self.ids]*.05, self.last], dim=-1)
        raw = self.head(obs)
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("Nonfinite teacher output")
        self.last = raw
        target = d.default_joint_pos.clone()
        target[:,self.ids] = self.default + self.contract["scale"]*raw
        action = (target-d.default_joint_pos)/self.c.action_scale
        error = float((d.default_joint_pos+action*self.c.action_scale-target).abs().max())
        self.mapping_error = max(self.mapping_error,error)
        self.calls += 1
        return action

    def reset(self, done):
        self.head.reset(done)
        self.last[done] = 0
