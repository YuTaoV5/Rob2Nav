"""Free-base Go2 paw-to-palm contact skill. No robot pose teleportation or welds."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x*x*x*(10+x*(-15+6*x))


def limited(current, target, speed, dt):
    delta = np.asarray(target)-current
    return current+delta*min(1., speed*dt/max(np.linalg.norm(delta), 1e-9))


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.]])


@dataclass
class HandObservation:
    position: list
    confidence: float = 1.
    open_palm: bool = True
    visible: bool = True
    sequence: int = 0


class PawSimulation:
    DT = .002
    CONTROL_DT = .01
    LEGS = ('FL', 'FR', 'RL', 'RR')

    def __init__(self, root, seed=0, friction=.8, mass_scale=1.):
        self.root = Path(root)
        self.rng = np.random.default_rng(seed)
        xml = ET.parse(self.root/'vendor/RoboGauge/resources/robots/go2/go2.xml').getroot()
        xml.find('compiler').set('meshdir', str(self.root/'vendor/RoboGauge/resources/robots/go2/assets'))
        xml.find('option').attrib.update(timestep=str(self.DT), gravity='0 0 -9.81', iterations='80', cone='elliptic')
        for motor in xml.find('actuator'):
            motor.set('class', 'knee' if 'calf' in motor.attrib['name'] else 'go2')
        for inertial in xml.findall('.//inertial'):
            inertial.set('mass', str(float(inertial.attrib['mass'])*mass_scale))
            inertial.set('diaginertia', ' '.join(str(float(x)*mass_scale) for x in inertial.attrib['diaginertia'].split()))
        world = xml.find('worldbody')
        base = world.find('body')
        base.set('pos', '0 0 .34')
        ET.SubElement(world, 'light', pos='0 -1 3', dir='0 0 -1', diffuse='.85 .85 .85')
        ET.SubElement(world, 'light', pos='1 2 2', dir='-1 -1 -1', diffuse='.5 .5 .5')
        ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 .1', rgba='.82 .85 .89 1', friction=f'{friction} .01 .001', condim='4')
        palm = ET.SubElement(world, 'body', name='human_hand', mocap='true', pos='.8 -.12 .18')
        ET.SubElement(palm, 'geom', name='palm', type='box', size='.055 .045 .012', rgba='.90 .64 .47 1', friction='1 .02 .002', condim='4', solref='.025 1', solimp='.9 .95 .002', mass='.2')
        for i in range(4):
            ET.SubElement(palm, 'geom', name=f'finger_{i}', type='capsule', fromto=f'-.045 {-.038+i*.025} 0 -.115 {-.038+i*.025} .005', size='.009', rgba='.90 .64 .47 1', contype='0', conaffinity='0', mass='.01')
        ET.SubElement(palm, 'geom', name='thumb', type='capsule', fromto='.01 -.05 -.002 -.04 -.09 .002', size='.014', rgba='.90 .64 .47 1', contype='0', conaffinity='0', mass='.01')
        ET.SubElement(palm, 'geom', name='forearm', type='capsule', fromto='.07 0 -.01 .30 0 -.02', size='.026', rgba='.30 .48 .65 1', contype='0', conaffinity='0', mass='.01')
        visual = ET.SubElement(xml, 'visual')
        ET.SubElement(visual, 'global', offwidth='1280', offheight='800')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.kin = mujoco.MjData(self.model)
        self.base = self.model.body('base_link').id
        self.feet = [self.model.geom(n).id for n in self.LEGS]
        self.foot_bodies = self.model.geom_bodyid[self.feet]
        self.palm_geom = self.model.geom('palm').id
        self.floor = self.model.geom('floor').id
        self.joints = self.model.actuator_trnid[:, 0]
        self.qids = self.model.jnt_qposadr[self.joints]
        self.dofs = self.model.jnt_dofadr[self.joints]
        self.bounds = self.model.jnt_range[self.joints]
        self.mass = self.model.body_subtreemass[self.base]
        self.qhome = np.tile([0., .75, -1.5], 4)
        self.data.qpos[self.qids] = self.qhome
        mujoco.mj_forward(self.model, self.data)
        self.home = self.data.geom_xpos[self.feet].copy()
        self.home[:, 2] = .022
        self.base_home = np.array([0., 0., .337])
        self.qtarget = self.qhome.copy()
        self.target = self.home.copy()
        self.state = 'SETTLE'
        self.entered = 0.
        self.leg = 1
        self.obs = None
        self.obs_at = -100.
        self.obs_sequence = -1
        self.filtered = np.array([.37, -.13, .19])
        self.hand_actual = np.array([.8, -.13, .19])
        self.hand_candidate_since = None
        self.last_good = -100.
        self.contact_since = None
        self.contact_lost = None
        self.force = 0.
        self.ground_force = np.zeros(4)
        self.peak_force = 0.
        self.peak_tilt = 0.
        self.events = []
        self.last_tau = np.zeros(12)
        self.renderer = None
        self.frame = None
        self.last_reason = ''
        self.armed = True
        self.requested_leg = None
        self.posture_kp = 35.
        self.force_offset = 0.
        self.settle_anchor = self.home.copy()
        self.environment_hand = None

    def observe(self, observation: HandObservation):
        p = np.asarray(observation.position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or not 0 <= observation.confidence <= 1:
            raise ValueError('Invalid hand observation')
        if observation.sequence <= self.obs_sequence:
            return False
        self.obs_sequence = observation.sequence
        self.obs = observation
        self.obs_at = float(self.data.time)
        return True

    def change(self, state, reason=''):
        if state == self.state:
            return
        self.events.append(dict(time=round(float(self.data.time), 4), previous=self.state, state=state, reason=reason))
        self.state, self.entered, self.last_reason = state, float(self.data.time), reason
        self.stage_start = self.data.geom_xpos[self.feet[self.leg]].copy()
        self.contact_since = None
        if state == 'RETRACT':
            self.retract_start = self.stage_start.copy()
            self.armed = False
        if state == 'HOLD':
            self.force_offset = 0.

    @property
    def tilt(self):
        return float(np.arccos(np.clip(self.data.xmat[self.base].reshape(3,3)[2,2], -1, 1)))

    def contacts(self):
        self.force = 0.
        self.ground_force[:] = 0
        self.other_hand_contact = False
        self.other_hand_force = 0.
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 not in self.feet and c.geom2 not in self.feet and self.palm_geom not in (c.geom1, c.geom2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            pair = (c.geom1, c.geom2)
            if self.palm_geom in pair:
                if self.feet[self.leg] in pair:
                    self.force += max(0., force[0])
                else:
                    self.other_hand_force += max(0.,force[0])
                    self.other_hand_contact = self.other_hand_contact or force[0] > 2.
            if self.floor in pair:
                for leg, foot in enumerate(self.feet):
                    if foot in pair:
                        self.ground_force[leg] += max(0., force[0])
        self.peak_force = max(self.peak_force, self.force)
        self.peak_tilt = max(self.peak_tilt, self.tilt)

    def valid_hand(self):
        t = float(self.data.time)
        if self.obs is None or t-self.obs_at > .28 or not self.obs.visible or not self.obs.open_palm or self.obs.confidence < .75:
            return False
        p = np.array(self.obs.position)
        if not (.29 <= p[0] <= .46 and .055 <= abs(p[1]) <= .23 and .12 <= p[2] <= .27):
            return False
        chosen = (0 if p[1] >= 0 else 1) if self.requested_leg is None else self.requested_leg
        if (chosen == 0 and p[1] < .04) or (chosen == 1 and p[1] > -.04):
            return False
        if self.state not in ('WAIT', 'SETTLE', 'COOLDOWN') and chosen != self.leg:
            return False
        if self.state in ('WAIT', 'SETTLE', 'COOLDOWN'):
            self.leg = chosen
        # Bound innovations; a switched hand or a discontinuous detector should not move the paw abruptly.
        delta = p-self.filtered
        self.filtered += np.clip(delta, -.012, .012)*.30
        self.last_good = t
        return True

    def ik(self, targets, desired_base):
        self.kin.qpos[:] = self.data.qpos
        self.kin.qpos[:3] = desired_base
        self.kin.qpos[3:7] = [1.,0.,0.,0.]
        self.kin.qpos[self.qids] = self.qtarget
        for _ in range(6):
            mujoco.mj_forward(self.model, self.kin)
            worst = 0.
            for leg in range(4):
                jac = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.kin, jac, None, self.kin.geom_xpos[self.feet[leg]], int(self.foot_bodies[leg]))
                cols = self.dofs[leg*3:leg*3+3]
                j = jac[:, cols]
                error = targets[leg]-self.kin.geom_xpos[self.feet[leg]]
                worst = max(worst, float(np.linalg.norm(error)))
                delta = j.T @ np.linalg.solve(j@j.T+np.eye(3)*1e-5, error)
                ids = self.qids[leg*3:leg*3+3]
                low, high = self.bounds[leg*3:leg*3+3].T
                self.kin.qpos[ids] = np.clip(self.kin.qpos[ids]+np.clip(delta,-.15,.15), low+.02, high-.02)
            if worst < .0005:
                break
        self.qtarget[:] = self.kin.qpos[self.qids]

    def support_forces(self, support, desired_base):
        # Centroidal force/moment feedback; the free base is never actuated directly.
        rmat = self.data.xmat[self.base].reshape(3,3)
        orientation = .5*sum(np.cross(rmat[:,i], np.eye(3)[:,i]) for i in range(3))
        force = self.mass*(np.array([28.,28.,65.])*(desired_base-self.data.qpos[:3])-np.array([9.,9.,12.])*self.data.qvel[:3]+[0,0,9.81])
        torque = np.array([60.,80.,35.])*orientation-np.array([8.,10.,5.])*self.data.qvel[3:6]
        a = np.concatenate([np.vstack((np.eye(3),skew(self.data.geom_xpos[self.feet[i]]-self.data.subtree_com[self.base]))) for i in support], axis=1)
        weights = np.diag([1.,1.,1.,3.,3.,2.])
        aw, b = weights@a, weights@np.r_[force,torque]
        f = aw.T@np.linalg.solve(aw@aw.T+np.eye(6)*.015,b)
        # Projected least-squares under unilateral contact and a friction pyramid.
        h = aw.T@aw+np.eye(len(f))*.01
        step = 1./max(np.linalg.norm(h,2),1.)
        for _ in range(18):
            f -= step*(h@f-aw.T@b)
            f = f.reshape(-1,3)
            f[:,2] = np.clip(f[:,2],0,120)
            f[:,:2] = np.clip(f[:,:2],-.55*f[:,2,None],.55*f[:,2,None])
            f = f.ravel()
        return f.reshape(-1,3)

    def control(self):
        self.contacts()
        t = float(self.data.time)
        age = t-self.entered
        good = self.valid_hand()
        if good:
            if self.hand_candidate_since is None:
                self.hand_candidate_since = t
        else:
            self.hand_candidate_since = None
        hand_visible = self.obs is not None and self.obs.visible and t-self.obs_at < .35
        hand_goal = np.clip(np.asarray(self.obs.position), [.18,-.35,.06],[.70,.35,.35]) if hand_visible else np.array([.8, self.filtered[1], .25])
        if self.environment_hand is not None:
            physical_position, physical_visible = self.environment_hand
            hand_goal = np.asarray(physical_position) if physical_visible else np.array([.8,self.filtered[1],.25])
        self.hand_actual = limited(self.hand_actual, hand_goal, .5 if not hand_visible else .22, self.CONTROL_DT)
        self.data.mocap_pos[0] = self.hand_actual-[0,0,.012]
        unsafe = self.tilt > .36 or self.data.qpos[2] < .23 or self.force > 18. or self.other_hand_contact
        active = self.state in ('SHIFT','LIFT','LOWER','HOLD')
        if active and (unsafe or t-self.last_good > .55 or not self.armed):
            self.change('RETRACT', 'force_or_balance_guard' if unsafe else ('user_stop' if not self.armed else 'hand_lost_or_outside_workspace'))
            age = 0.
        if self.state == 'SETTLE' and t > 1.5:
            self.change('WAIT')
        elif self.state == 'WAIT' and self.armed and self.hand_candidate_since is not None and t-self.hand_candidate_since > .35 and self.tilt < .12:
            self.change('SHIFT', 'stable_open_palm')
        elif self.state == 'SHIFT' and age > .9:
            self.change('LIFT')
        elif self.state == 'LIFT' and age > 1.2:
            self.change('LOWER')
        elif self.state == 'LOWER':
            if self.force > 1.2:
                if self.contact_since is None:
                    self.contact_since = t
                elif t-self.contact_since > .08:
                    self.change('HOLD', 'measured_contact')
            else:
                self.contact_since = None
            if age > 3.5:
                self.change('RETRACT', 'contact_timeout')
        elif self.state == 'HOLD':
            self.force_offset = float(np.clip(self.force_offset+.00010*(self.force-3.),-.012,.018))
            if self.force < .4:
                if self.contact_lost is None:
                    self.contact_lost = t
                elif t-self.contact_lost > .4:
                    self.change('LOWER', 'reacquire_contact')
            else:
                self.contact_lost = None
        elif self.state == 'RETRACT' and age > 1.6 and self.ground_force[self.leg] > 5:
            self.change('COOLDOWN')
        elif self.state == 'RETRACT' and age > 3:
            self.change('COOLDOWN', 'retract_deadline')
        elif self.state == 'COOLDOWN' and age > 1:
            self.change('WAIT')
        age = t-self.entered
        self.target[:] = self.home
        shift = np.array([-.055, -.055 if self.leg == 0 else .055, -.008])
        blend = smooth(age/.9) if self.state == 'SHIFT' else (1.-smooth(age) if self.state == 'COOLDOWN' else (1. if self.state in ('LIFT','LOWER','HOLD','RETRACT') else 0.))
        desired_base = self.base_home+shift*blend
        support = list(range(4))
        if self.state in ('LIFT','LOWER','HOLD','RETRACT'):
            support.remove(self.leg)
        if self.state == 'LIFT':
            hover = self.filtered+[0,0,.09]
            lift = self.stage_start.copy(); lift[2] = max(hover[2],.25)
            if age < .4:
                self.target[self.leg] = self.stage_start+(lift-self.stage_start)*smooth(age/.4)
            else:
                self.target[self.leg] = lift+(hover-lift)*smooth((age-.4)/.8)
        elif self.state in ('LOWER','HOLD'):
            if self.state == 'LOWER':
                target = self.filtered+[0,0,max(.016,.09-age*.045)]
            else:
                target = self.filtered+[0,0,.019+self.force_offset]
            self.target[self.leg] = limited(self.target_last, target, .16, self.CONTROL_DT)
        elif self.state == 'RETRACT':
            hover = self.retract_start.copy();hover[2] = max(.29,hover[2]+.06)
            home_hover = self.home[self.leg].copy();home_hover[2] = hover[2]
            if age < .35:
                self.target[self.leg] = self.retract_start+(hover-self.retract_start)*smooth(age/.35)
            elif age < .95:
                self.target[self.leg] = hover+(home_hover-hover)*smooth((age-.35)/.6)
            else:
                self.target[self.leg] = home_hover+(self.home[self.leg]-home_hover)*smooth((age-.95)/.65)
        self.target_last = self.target[self.leg].copy()
        self.ik(self.target, desired_base)
        q, dq = self.data.qpos[self.qids], self.data.qvel[self.dofs]
        tau = self.posture_kp*(self.qtarget-q)-2.2*dq+self.data.qfrc_bias[self.dofs]
        forces = self.support_forces(support, desired_base)
        for i, f in zip(support,forces):
            jac = np.zeros((3,self.model.nv))
            mujoco.mj_jac(self.model,self.data,jac,None,self.data.geom_xpos[self.feet[i]],int(self.foot_bodies[i]))
            tau -= jac[:,self.dofs].T@f
        if self.state in ('LIFT','LOWER','HOLD','RETRACT'):
            jac = np.zeros((3,self.model.nv))
            mujoco.mj_jac(self.model,self.data,jac,None,self.data.geom_xpos[self.feet[self.leg]],int(self.foot_bodies[self.leg]))
            vel = jac@self.data.qvel
            error = self.target[self.leg]-self.data.geom_xpos[self.feet[self.leg]]
            stiffness = np.array([300.,300.,150.])
            damping = np.array([12.,12.,7.])
            task = stiffness*error-damping*vel
            if self.state in ('LOWER','HOLD'):
                task[2] -= 3.
            task = np.clip(task,-18,18)
            cols = slice(3*self.leg,3*self.leg+3)
            tau[cols] = jac[:,self.dofs[cols]].T@task+self.data.qfrc_bias[self.dofs[cols]]
        lo, hi = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = np.clip(np.clip(tau,self.last_tau-3.,self.last_tau+3.),lo,hi)
        self.last_tau = self.data.ctrl.copy()

    def step(self, count=1):
        for _ in range(count):
            self.control()
            for _ in range(5):
                mujoco.mj_step(self.model,self.data)
            self.contacts()

    def snapshot(self):
        return dict(time=float(self.data.time),state=self.state,leg=self.LEGS[self.leg],
            force=float(self.force),peak_force=float(self.peak_force),tilt_deg=math.degrees(self.tilt),
            base=self.data.qpos[:3].tolist(),hand=self.hand_actual.tolist(),paw=self.data.geom_xpos[self.feet[self.leg]].tolist(),
            ground_force=self.ground_force.tolist(),observation_age=float(self.data.time)-self.obs_at,
            reason=self.last_reason,torque_peak=float(np.abs(self.last_tau).max()),armed=self.armed,
            unexpected_hand_force=float(self.other_hand_force))

    def render(self, width=960, height=640):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model,height=height,width=width)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [.12,0,.23]
        cam.distance,cam.azimuth,cam.elevation = 1.45,125,-18
        opt = mujoco.MjvOption();opt.geomgroup[3]=0
        self.renderer.update_scene(self.data,camera=cam,scene_option=opt)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
