"""Gyro-1 wheel feedback as a TRAINING-side action term, so training and evaluation share one
closed loop.

Why this file exists: gyro-1 was implemented only inside the evaluation wrapper
(`r11_acceptance_ff.py`) and never appeared in any training code, so every policy to date was
trained WITHOUT it and then evaluated WITH it.  The operator flagged that as a control-contract
inconsistency and required both training arms to use the same closed-loop implementation, recorded
as a control-contract change.

The evaluation wrapper computes, per policy step:

    raw  = ff_gain*wz_cmd + yaw_kp*(wz_cmd - wz_meas)
    ff   = S * clip(raw, +-ff_cap)                    S = (-1, +1, -1, +1) for FL, FR, RL, RR
    tau  = KD*(target - dq) + ff
    sp   = dq + tau/KD  =  target + ff/KD             <- equivalent velocity set-point

This action term applies exactly the same set-point offset, so the wheel loop is identical in
training and in evaluation.  Gains come from the environment variables AS2W_GYRO_KP / AS2W_GYRO_CAP
and default to 0, which reproduces the previous behaviour byte for byte when unset.

State-free by construction (no integrator, no filter), so being invoked once per physics sub-step
rather than once per policy step cannot compound.
"""
from __future__ import annotations

import os

import torch

from isaaclab.envs.mdp.actions import JointVelocityAction
from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg
from isaaclab.utils import configclass

WHEEL_KD = 6.0
S_WHEEL = (-1.0, 1.0, -1.0, 1.0)     # FL, FR, RL, RR -> left-versus-right differential


@configclass
class GyroWheelVelocityActionCfg(JointVelocityActionCfg):
    """JointVelocityActionCfg plus the gyro-1 parameters."""
    as2w_yaw_kp: float = 0.0
    as2w_ff_cap: float = 4.0
    as2w_ff_gain: float = 0.0
    as2w_wheel_kd: float = WHEEL_KD
    as2w_command_name: str = "base_velocity"


class GyroWheelVelocityAction(JointVelocityAction):
    """Velocity action with the gyro-1 yaw feedback folded into the set-point."""

    cfg: GyroWheelVelocityActionCfg

    def __init__(self, cfg: GyroWheelVelocityActionCfg, env):
        super().__init__(cfg, env)
        # environment variables win, so a launcher can switch the loop on without editing code
        kp = float(os.environ.get("AS2W_GYRO_KP", cfg.as2w_yaw_kp))
        cap = float(os.environ.get("AS2W_GYRO_CAP", cfg.as2w_ff_cap))
        gain = float(os.environ.get("AS2W_GYRO_FF_GAIN", cfg.as2w_ff_gain))
        self._as2w_kp, self._as2w_cap, self._as2w_gain = kp, cap, gain
        self._as2w_on = bool(kp != 0.0 or gain != 0.0)
        self._as2w_kd = float(cfg.as2w_wheel_kd)
        self._as2w_cmd_name = cfg.as2w_command_name
        self._as2w_s = torch.tensor(S_WHEEL, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
        if self._as2w_on:
            print("AS2W_GYRO_TRAIN_LOOP ON  yaw_kp=%.1f  ff_cap=%.2f  ff_gain=%.1f  kd=%.1f"
                  % (self._as2w_kp, self._as2w_cap, self._as2w_gain, self._as2w_kd), flush=True)
        else:
            print("AS2W_GYRO_TRAIN_LOOP OFF (gains are zero; baseline behaviour)", flush=True)

    def apply_actions(self):
        tgt = self.processed_actions
        if not self._as2w_on or tgt.shape[-1] != 4:
            self._asset.set_joint_velocity_target_index(target=tgt, joint_ids=self._joint_ids)
            return
        # measured body-frame yaw rate, one sample per call, same quantity the eval wrapper used
        wz_meas = self._asset.data.root_ang_vel_b.torch[:, 2]
        try:
            wz_cmd = self._env.command_manager.get_command(self._as2w_cmd_name)[:, 2]
        except Exception:
            wz_cmd = torch.zeros_like(wz_meas)
        raw = self._as2w_gain * wz_cmd + self._as2w_kp * (wz_cmd - wz_meas)
        mag = torch.clamp(raw, -self._as2w_cap, self._as2w_cap)
        ff = self._as2w_s.to(tgt.dtype) * mag.unsqueeze(-1)          # (N, 4)
        sp = tgt + ff / self._as2w_kd
        self._asset.set_joint_velocity_target_index(target=sp, joint_ids=self._joint_ids)
