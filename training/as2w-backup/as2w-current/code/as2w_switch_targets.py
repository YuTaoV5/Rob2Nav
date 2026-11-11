"""Common normalized observation and target mapping for physical switch diagnostics."""
import numpy as np
import torch

from as2w_mode_switch import Phase
from as2w_rolling_fixed_exit import rolling_fixed_exit, rolling_wheel_hold, rolling_precalibrated_stand
from as2w_stepping_control import guarded_sequence_action
from as2w_zero_memory import ZeroWheelMemory


def request_at(time_s,reverse=False):
    if time_s<4.:
        return None
    if reverse:
        return "stepping" if time_s<20. else "rolling" if time_s<29. else None
    return "rolling" if time_s<11. else "stepping" if time_s<28. else None


def diagnostic_duration(reverse=False):
    # The observation window covers the request, both existing timeouts,
    # and two complete seconds of verified fixed stand.
    release = 29. if reverse else 28.
    return release+5.+4.+2.


class SwitchTargets:
    def __init__(self,policies,q0,lo,hi,wheel_limit,dt,wheel_angle):
        self.policies = policies
        self.q0,self.lo,self.hi = q0,lo,hi
        self.wheel_limit,self.dt = wheel_limit,dt
        self.wheel_anchor = wheel_angle.copy()
        self.memory = ZeroWheelMemory()
        self.bias = torch.zeros((1,12))
        self.reference = torch.zeros((1,4))
        self.held = torch.zeros(1,dtype=torch.long)
        self.last_action = None
        self.filtered = np.zeros(3,dtype=np.float32)
        self.age = 0.

    @torch.inference_mode()
    def update(self,q,dq,gyro,gravity,decision,stick_moving,t,*,fade_stand_bias=False, recalibrate_fixed_exit=False):
        mode = decision.policy
        if decision.reset_policy_history:
            self.last_action = np.zeros(16 if mode=="rolling" else 12,dtype=np.float32)
            self.memory = ZeroWheelMemory()
            self.wheel_anchor = q[12:].copy()
            self.age = 0.
        target = np.zeros(3,dtype=np.float32)
        if stick_moving and decision.joystick_enabled:
            target[0] = .22 if mode=="rolling" else .10
        self.filtered += .12*(target-self.filtered)
        if not np.any(target) and np.max(np.abs(self.filtered))<.005:
            self.filtered[:] = 0.
        cmd = torch.from_numpy(self.filtered[None])
        q_error = q[:12]-self.q0
        wheel = (q[12:]+2.*np.pi)%(4.*np.pi)-2.*np.pi
        sensor = np.r_[gyro,gravity,q_error,dq[:12]].astype(np.float32)
        sense = torch.from_numpy(sensor[None])
        wheel_angle = torch.as_tensor(wheel[None],dtype=torch.float32)
        wheel_rate = torch.as_tensor(dq[None,12:],dtype=torch.float32)
        self.reference,self.held,hold = rolling_wheel_hold(
            self.reference,self.held,cmd,wheel_angle,wheel_rate,sense[:,:3],self.dt)
        recalibrate = recalibrate_fixed_exit and mode=="rolling" and decision.phase==Phase.EXIT and decision.weight==0.
        self.bias,_ = rolling_precalibrated_stand(
            self.bias,cmd,sense[:,6:18],sense[:,:3],-sense[:,5],wheel_rate,
            self.held,cmd.new_tensor([t]),self.dt,recalibrate=recalibrate)
        fixed = np.clip(self.q0+self.bias[0].numpy(),self.lo+.05,self.hi-.05)
        fixed_wheel = (16.*hold[0]).numpy()
        legs_target,wheel_target = fixed.copy(),fixed_wheel.copy()
        if mode=="rolling":
            phase = 2.*np.pi*self.age/.62
            obs = np.r_[self.filtered-np.array((.4,0.,0.)),sensor[:6],.4,q_error,wheel,dq,
                        self.last_action,np.sin(phase),np.cos(phase)].astype(np.float32)
            raw = self.policies[mode].run(self.memory.push(obs),name=mode,centered=False,reset=False)
            legs,wheels = rolling_fixed_exit(cmd.new_tensor([raw]),cmd)
            self.last_action = torch.cat((legs,wheels),dim=1)[0].numpy()
            legs_target = np.clip(fixed+.35*self.last_action[:12],self.lo+.05,self.hi-.05)
            wheel_target = np.clip(16.*self.last_action[12:]+wheel_target,-self.wheel_limit,self.wheel_limit)
        elif mode=="stepping":
            phase = 2.*np.pi*max(self.age-2.,0.)/8.
            obs = np.r_[self.filtered,sensor,self.last_action,np.sin(phase),np.cos(phase)]
            raw = self.policies[mode].run(obs,name=mode,centered=False,reset=False)
            self.last_action = guarded_sequence_action(cmd,cmd.new_tensor([self.age]),
                cmd.new_tensor([raw]),sense[:,12:14],27.,period=8.,stride_gain=.35)[0].numpy()
            reference = fixed
            if fade_stand_bias:
                scale = np.clip((2.4-self.age)/.9,0.,1.)
                reference = self.q0+scale*(fixed-self.q0)
            legs_target = np.clip(reference+.30*self.last_action,self.lo+.05,self.hi-.05)
        if mode:
            self.age += self.dt
        return fixed,fixed_wheel,legs_target,wheel_target
