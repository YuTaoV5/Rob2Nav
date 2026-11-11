"""Evaluate the R13 reward expressions on one AS2-W MuJoCo recording."""
import numpy as np
import mujoco as mj
from mujoco_telemetry import PhysicalTelemetry
from reward_expression import SIGNALS, expression, reward_specs


class ReplayScore:
    def __init__(self, model, contract, weights):
        self.model = model
        self.weights = weights
        self.extra = [(spec, expression(spec["formula"])) for spec in reward_specs(weights.get("custom_rewards", []))]
        self.dt = contract["control_dt"]
        joints = [model.joint(name) for name in contract["joint_names"]]
        self.base = model.body("base_link").id
        self.telemetry = PhysicalTelemetry(model, self.base, joints)
        self.wheels = [model.jnt_bodyid[j.id] for j in joints[12:]]
        self.leg_qpos = [j.qposadr[0] for j in joints[:12]]
        self.default = np.array(contract["default_joint_pos"][:12])
        self.streak = 0
        self.frames = []
        self.force = np.zeros(6)

    def sample(self, data, action, active, ground_z):
        model = self.model
        rotation = data.xmat[self.base].reshape(3, 3)
        upright = float(np.clip(rotation[2, 2], -1., 1.))
        height = float(data.qpos[2] - ground_z)
        pose_error = float(np.mean(np.square(data.qpos[self.leg_qpos] - self.default)))
        standing = float(np.exp(-pose_error / .25 - ((height - .36) / .12)**2))
        loads = np.zeros((model.nbody, 3))
        for index, contact in enumerate(data.contact):
            mj.mj_contactForce(model, data, index, self.force)
            force = contact.frame.reshape(3, 3).T @ self.force[:3]
            one, two = model.geom_bodyid[contact.geom]
            loads[one] -= force
            loads[two] += force
        vertical = loads[self.wheels, 2]
        other = [body for body in range(1, model.nbody) if body not in self.wheels]
        impact = float(np.linalg.norm(loads[other], axis=1).max())
        support = np.clip(vertical / 8., 0., 1.)
        linear, angular = float(np.linalg.norm(data.qvel[:3])), float(np.linalg.norm(data.qvel[3:6]))
        righted = float(np.clip((upright - .65) / .35, 0., 1.))
        braking = 2. / (1. + linear / .2) + 2. / (1. + angular / .4)
        stance = (braking + 2. * support.mean() + 2. * support.min()) * np.clip((height - .23) / .10, 0., 1.)
        stable = bool(active and upright >= .9396926208 and height >= .29 and (vertical > 8.).all()
                      and linear <= .2 and angular <= .4 and impact < 1.)
        self.streak = self.streak + 1 if stable else 0
        stance += 4. * min(self.streak * self.dt, 1.)
        raw = (upright, standing, min(impact, 2000.), float(righted * stance), float(np.mean(action**2)))
        p = self.weights
        weight = (p["upright_weight"], p["pose_weight"], -p["impact_weight"], p["stance_weight"],
                  -(p["effort_base"] + p["effort_righted"] * righted))
        terms = {key:dict(raw=value,weight=float(scale),gate=bool(active),weighted=float(value*scale) if active else 0.)
                 for key,value,scale in zip(("upright","pose","impact","stance","effort"),raw,weight)}
        signals = dict(zip(SIGNALS, raw))
        for i, (spec, fn) in enumerate(self.extra):
            value = float(np.clip(fn(signals), -10., 10.))
            terms[f"custom_{i}"] = dict(raw=value, weight=spec["weight"], gate=bool(active), weighted=value*spec["weight"] if active else 0., label=spec["name"])
        self.frames.append(dict(step=len(self.frames), time=len(self.frames)*self.dt, sim_time_s=float(data.time),
            telemetry=self.telemetry.sample(data),
            terms=terms,
            signals=dict(height_m=height,wheel_min_n=float(vertical.min()),roll_rad=float(np.arctan2(rotation[2,1],upright)),
                         angular_speed=angular,speed=linear,stable_streak_s=self.streak*self.dt,
                         active=float(active),righted_gate=righted,stable_gate=float(stable), nonwheel_impact_n=impact,
                         **{f"wheel_{leg}_n":float(load) for leg,load in zip(("FL","FR","RL","RR"),vertical)})))

    def result(self, profile):
        return dict(schema_version=2, backend="MuJoCo", video_fps=round(1/self.dt), control_dt_s=self.dt,
                    label="MuJoCo 复算 · " + profile + " · 使用本回合 MuJoCo 接触数据，贡献未乘步长",
                    frames=self.frames)
