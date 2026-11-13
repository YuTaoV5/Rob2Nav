"""Patch a private copy of the pinned R4000 recipe for rolling turns."""
import hashlib
import sys
from pathlib import Path


def edit(name, digest, changes):
    path = Path(sys.argv[1]) / name
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected baseline: {path}")
    source = data.decode()
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError(f"Expected one match in {path}: {old}")
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8", newline="\n")


edit("as2w_isaac_speed_range.py",
     "b7b35e6d2a37c72be5fbfb83adf01f0dee80f5cd544f862c635357d348924701", [
    ("yaw_scale = ((.18 - planar.norm(dim=1)) / .08).clamp(0., 1.)",
     """yaw_limit = .10 - .06 * ((planar[:, 0].abs() - .20) / .80).clamp(0., 1.)
        self.vel_command_b[:, 2].clamp_(-yaw_limit, yaw_limit)
        yaw_scale = torch.ones_like(yaw_limit)"""),
    ("draw = torch.rand((shape[0], 3), device=self.device)\n            sign",
     "draw = torch.rand((shape[0], 4), device=self.device)\n            sign"),
    ("cmd[:, 0] = torch.where(turn, 0., sign * (.03 + .17 * draw[:, 1]))",
     """cap = self._env.cfg.teleop_speed_cap
            speed = .03 + (cap - .03) * draw[:, 1]
            speed = torch.where(draw[:, 3] < .30, speed.clamp(max=.20), speed)
            cmd[:, 0] = torch.where(turn, 0., sign * speed)"""),
    ("cmd[:, 2] = torch.where(turn | arc, sign * (.035 + .085 * draw[:, 1]), 0.)\n            cmd[arc, 0] = .05",
     """yaw_sign = torch.where(draw[:, 3] < .5, -1., 1.)
            cmd[:, 2] = torch.where(turn | arc, yaw_sign * (.04 + .06 * draw[:, 1]), 0.)"""),
    ("resampling_time_range = (1.1, 3.4)", "resampling_time_range = (6., 15.)"),
    ("yaw_error = (yaw-command[:, 2]).abs()/.20",
     """term = env.command_manager.get_term("base_velocity")
    late = ((term.segment_duration - term.time_left - 3.) / 2.).clamp(0., 1.)
    yaw_error = (1. + late) * (yaw-command[:, 2]).abs()/.20"""),
])
edit("as2w_rolling_fixed_exit.py",
     "54b60cc9c0285770f347dde463b1b09b4178f6d1120e6c04fc7054d2ecfa3b24", [
    ("    wheels = linear / (.086 * 16.) + .035 * motion * (1. - turning) * (wheel - common)",
     """    straight_residual = wheel - common - steering * side
    yaw_gate = (yaw.abs() / .04).clamp(max=1.)
    wheels = linear / (.086 * 16.) + .035 * motion * (1. - turning) * straight_residual
    yaw_residual_limit = (.33 * yaw_drive.abs() / (.086 * 16.)).clamp(max=.06)
    wheels = wheels - yaw_gate * yaw_residual_limit * yaw.sign() * steering * side
    # Keep the original wheel map on the straight and neutral paths.
    wheels = wheels + .035 * motion * (1. - turning) * (1. - yaw_gate) * steering * side"""),
])
edit("train_as2w_isaac_robust.py",
     "fdd01444528b82b31b3a301e3375fe4718f55721da4599c1530f10d0cddb6609", [
    ("runner_cfg.save_interval = min(25, args.iterations)",
     "runner_cfg.save_interval = min(1000, args.iterations)"),
    ("runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)",
     """runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
            runner.save(str(args.out / "stage-final.pt"))"""),
])
edit("eval_as2w_mujoco_rolling.py",
     "87b51dea6999e446637f51adc3752362fd756e3233ca491f8d13486b7d645999", [
    ("pure_turn_common_scale=args.pure_turn_common_scale)",
     """pure_turn_common_scale=args.pure_turn_common_scale,
                                             gyro_z=frame[None,5], wheel_dq=frame[None,38:42])"""),
])
