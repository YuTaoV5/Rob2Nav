"""A same-side neutral stance for straight rolling and standing."""
import numpy as np


class StraightRoll:
    def __init__(self, base, dt, geometry, profile):
        self.base = base
        self.dt = dt
        self.inverse = geometry.inverse
        self.height, self.yaw_p, self.yaw_d, self.slew = profile
        self.heading = self.forward = 0.
        self.active = False

    def plan(self, model, data, obs, command, tick, advance):
        a = self.base.plan(model,data,obs,command,tick,advance)
        strength = float(np.clip(1.-max(abs(command[1])/.045,abs(command[2])/.085),0.,1.))
        if strength == 0.:
            self.active = False
            return a
        if not self.active:
            self.heading = 0.
            self.forward = float(np.clip(.086*np.median(obs[38:42]),-.6,.6))
            self.active = True
        up = -obs[6:9]
        yaw = obs[3:6] @ up
        self.heading = np.clip(self.heading+self.dt*(yaw-command[2]),-.50,.50)
        step = self.slew*self.dt
        self.forward += np.clip(command[0]-self.forward,-step,step)
        side = np.array((-1.,1.,-1.,1.))
        ends = np.array((1.,1.,-1.,-1.))
        dz = -self.height + np.clip(.075*obs[6]*ends+.07*obs[7]*side,-.025,.025)
        delta = np.column_stack((np.zeros(4),np.zeros(4),dz))
        legs = np.einsum("lij,lj->il",self.inverse,delta).reshape(12)/.35
        steer = np.clip(.16*command[2] - self.yaw_p*self.heading - self.yaw_d*(yaw-command[2]),-.20,.20)
        wheels = (self.forward+side*steer)/(.086*16.)
        return a+strength*(np.r_[legs,wheels]-a)
