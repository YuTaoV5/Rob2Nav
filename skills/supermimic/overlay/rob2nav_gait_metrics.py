"""Read-only, physical-substep gait measurements; never change controls or state.

Duty factor uses complete touchdown-to-touchdown cycles, not an arbitrary-window
contact percentage. Raw collision occupancy is reported separately. Contacts are
geometric (active dist <= 0), not a claim of measured load-bearing force.
"""
import json
from pathlib import Path

import numpy as np
import torch

from go2_supermimic.supermimic import state as st
from rob2nav_contact_audit import BodyContactAudit

STAGES = ("stand", "forward", "backward", "left", "right", "turn_left", "turn_right", "stop")


def complete_cycles(contact, valid, dt, minimum_period=0.15, maximum_period=2.0):
    contact = np.asarray(contact, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    touchdowns = np.flatnonzero(contact[1:] & ~contact[:-1]) + 1
    cycles = []
    for start, end in zip(touchdowns[:-1], touchdowns[1:]):
        period = (end - start) * dt
        # A one-substep separation is collision chatter, not a meaningful swing.
        swing = ~contact[start:end]
        edges = np.diff(np.r_[False, swing, False].astype(int))
        longest_swing = max((b-a for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))), default=0)*dt
        if valid[start:end + 1].all() and minimum_period <= period <= maximum_period and longest_swing >= .04 and contact[start:end].sum()*dt >= .04:
            cycles.append({"period_s": float(period), "duty_factor": float(contact[start:end].mean())})
    return cycles


def summarize(data, dt, stage_seconds, nominal_mass):
    rows = []
    nstage = round(stage_seconds / dt)
    for stage_id, label in enumerate(STAGES):
        sl = slice(stage_id * nstage, min((stage_id + 1) * nstage, len(data["feet_contact"])))
        contact = data["feet_contact"][sl]
        if len(contact) < 2:
            continue
        valid = data["valid"][sl].astype(bool)
        # Settling affects summary gait estimates, never the separate joint/contact gates.
        valid[:round(min(1., stage_seconds / 4) / dt)] = False
        feet = data["feet_world"][sl]
        velocity = np.diff(feet, axis=0) / dt
        pair_valid = valid[1:] & valid[:-1]
        stance = contact[1:] & contact[:-1] & pair_valid[..., None]
        slip = np.linalg.norm(velocity[..., :2], axis=-1)
        leg_rows = []
        for leg, name in enumerate(st.LEGS):
            cycles = []
            for env_id in range(contact.shape[1]):
                cycles.extend(complete_cycles(contact[:, env_id, leg], valid[:, env_id], dt))
            leg_rows.append({"leg": name, "complete_cycles": len(cycles),
                             "duty_factor": float(np.mean([c["duty_factor"] for c in cycles])) if cycles else None,
                             "period_s": float(np.mean([c["period_s"] for c in cycles])) if cycles else None,
                             "raw_contact_occupancy": float(contact[..., leg][valid].mean()) if valid.any() else None})
        gravity = data["gravity"][sl]
        tilt = np.degrees(np.arccos(np.clip(-gravity[..., 2] / np.maximum(np.linalg.norm(gravity, axis=-1), 1e-8), -1, 1)))
        roll = np.degrees(np.arctan2(-gravity[..., 1], -gravity[..., 2]))
        pitch = np.degrees(np.arctan2(gravity[..., 0], np.linalg.norm(gravity[..., 1:], axis=-1)))
        height = data["height"][sl]
        power = data["absolute_power"][sl]
        speed = np.linalg.norm(data["velocity"][sl][..., :2], axis=-1)
        dfs = [r["duty_factor"] for r in leg_rows if r["duty_factor"] is not None]
        rows.append({"stage": label, "legs": leg_rows,
                     "duty_factor_leg_spread": float(max(dfs)-min(dfs)) if len(dfs) == 4 else None,
                     "named_foot_support_count_mean": float(contact.sum(-1)[valid].mean()) if valid.any() else None,
                     "stance_slip_rms_m_s": float(np.sqrt(np.mean(slip[stance]**2))) if stance.any() else None,
                     "tilt_rms_deg": float(np.sqrt(np.mean(tilt[valid]**2))) if valid.any() else None,
                     "roll_rms_deg": float(np.sqrt(np.mean(roll[valid]**2))) if valid.any() else None,
                     "pitch_rms_deg": float(np.sqrt(np.mean(pitch[valid]**2))) if valid.any() else None,
                     "foot_clearance_p95_m": np.quantile(feet[..., 2][valid]-.022, .95, axis=0).tolist() if valid.any() else None,
                     "height_mean_m": float(height[valid].mean()) if valid.any() else None,
                     "height_std_m": float(height[valid].std()) if valid.any() else None,
                     "angular_rate_xy_rms_rad_s": float(np.sqrt(np.mean(np.sum(data["angular_velocity"][sl][..., :2][valid]**2, axis=-1)))) if valid.any() else None,
                     "absolute_mechanical_power_mean_W": float(power[valid].mean()) if valid.any() else None,
                     "nominal_mass_cost_of_transport": float(power[valid].mean() / (nominal_mass*9.81*speed[valid].mean())) if valid.any() and speed[valid].mean() > .05 else None,
                     "calf_collision_occupancy": float(data["calf_contact"][sl][valid].mean()) if valid.any() else None})
    return {"stages": rows, "physics_dt": dt, "physical_samples": len(data["height"]),
            "contact_definition": "active raw terrain/named-foot geometric contact dist<=0; not force-qualified load duty",
            "cycle_definition": "complete touchdown-to-touchdown; 0.15..2s; stance and continuous swing >=40ms; no reset or stage-boundary crossing",
            "slip_definition": "world site finite difference while named foot contact persists; reset pairs excluded",
            "cost_definition": "sum abs(applied_joint_torque*joint_velocity)/(nominal mass*g*actual speed), not electrical energy",
            "capability_qualified": False}


class GaitRecorder:
    def __init__(self, env, output, stage_seconds):
        self.env, self.output, self.stage_seconds = env, Path(output), stage_seconds
        self.contact = BodyContactAudit(env)
        self.foot_map = torch.full((len(self.contact.names),), -1, device=env.device, dtype=torch.long)
        self.calf_map = self.foot_map.clone()
        for geom_id, name in enumerate(self.contact.names):
            for leg_id, leg in enumerate(st.LEGS):
                if name.endswith(leg + "_foot_collision"):
                    self.foot_map[geom_id] = leg_id
                elif "/" + leg + "_calf_collision" in name:
                    self.calf_map[geom_id] = leg_id
        self.samples = []
        self.valid = []
        self.nominal_mass = float(env.sim.mj_model.body_mass.sum())
        self.closed = False

    def contacts(self, mapping):
        audit = self.contact
        geom, worlds = audit.geom.long(), audit.world.long()
        a, b = geom[:, 0].clamp(0, len(mapping)-1), geom[:, 1].clamp(0, len(mapping)-1)
        leg = torch.where(audit.ground[b], mapping[a], torch.where(audit.ground[a], mapping[b], -1))
        valid = (torch.arange(len(geom), device=self.env.device) < audit.nacon[0]) & (worlds >= 0) & (worlds < self.env.num_envs) & (leg >= 0) & (audit.dist <= 0) & (geom.min(-1).values >= 0)
        counts = torch.zeros(self.env.num_envs*4, device=self.env.device, dtype=torch.long)
        ids = worlds[valid]*4 + leg[valid]
        counts.scatter_add_(0, ids, torch.ones_like(ids))
        return (counts > 0).reshape(self.env.num_envs, 4)

    def sample(self):
        env = self.env
        s = st.robot_state(env)
        robot = env.scene["robot"]
        tau = env.sim.data.qfrc_actuator[:, robot.indexing.joint_v_adr]
        row = {"feet_contact": self.contacts(self.foot_map), "calf_contact": self.contacts(self.calf_map),
               "feet_world": st.limb_positions_w(robot)["feet"], "height": s[:, st.Z].squeeze(-1),
               "gravity": s[:, st.GRAV], "angular_velocity": s[:, st.W_B], "velocity": s[:, st.V_YAW],
               "absolute_power": (tau*s[:, st.QD]).abs().sum(-1)}
        self.samples.append({k: value.detach().cpu().numpy().copy() for k, value in row.items()})
        self.valid.append(np.ones(env.num_envs, dtype=bool))

    def mark_resets(self, terminated):
        ids = terminated.detach().cpu().numpy().astype(bool)
        if ids.any():
            for v in self.valid[-self.env.cfg.decimation:]:
                v[ids] = False

    def close(self):
        if self.closed or not self.samples:
            return
        self.closed = True
        data = {k: np.stack([r[k] for r in self.samples]) for k in self.samples[0]}
        data["valid"] = np.stack(self.valid)
        report = summarize(data, self.env.physics_dt, self.stage_seconds, self.nominal_mass)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output.with_suffix(".npz"), **data)
        self.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

