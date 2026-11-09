"""Proprioceptive corrections for the initial rear support and summit stop."""
import numpy as np

SIDE=np.array((1.,-1.,1.,-1.))
HIP_X=np.array((.2119,.2119,-.2119,-.2119))


class Clearance:
    def __init__(self,profile,inverse,defaults,brake=None):
        self.rear,self.summit=profile
        self.brake=brake
        self.inverse=inverse
        self.defaults=np.asarray(defaults[:12]).reshape(3,4)
        self.distance=0.
        self.level_ticks=0
        self.hold=0

    def displacement(self,z):
        return np.einsum("lij,lj->il",self.inverse,np.column_stack((np.zeros(4),np.zeros(4),z))).reshape(12)/.35

    def apply(self,action,obs,age):
        if self.rear:
            pitch=-obs[6]
            rising=max(0.,-obs[4])
            trim=np.clip(self.rear*max(pitch-.06,0.)+.012*rising,0.,.061)
            fade=np.clip((62-age)/25.,0.,1.)
            action=action.copy()
            action[:12]+=self.displacement(fade*trim*np.array((.30,.30,-1.,-1.)))
        if not self.summit:
            return action
        self.distance+=.02*np.clip(.086*np.median(obs[38:42]),0.,.85)
        hip,thigh,calf=self.defaults+obs[10:22].reshape(3,4)
        long=-.212*(np.cos(thigh)+np.cos(thigh+calf))
        points=np.column_stack((HIP_X-.212*(np.sin(thigh)+np.sin(thigh+calf)),
                               .06*SIDE+.1054*SIDE*np.cos(hip)-long*np.sin(hip),
                               .1054*SIDE*np.sin(hip)+long*np.cos(hip)))
        up=-obs[6:9]
        spread=np.ptp(points@up)
        level=(self.distance>=self.summit and spread<.034 and up[2]>.985
               and np.abs(obs[38:42]).mean()<7.)
        confirmation=6
        if self.brake is not None and not self.hold:
            level=self.brake.ready(obs,self.distance,age)
            confirmation=3
        self.level_ticks=self.level_ticks+1 if level else 0
        if self.level_ticks>=confirmation or self.hold:
            self.hold+=1
            u=min(1.,self.hold/10.)
            u=u*u*(3.-2.*u)
            neutral=self.displacement(np.full(4,-.018))
            wheel=.16*obs[5]*SIDE
            action=action+u*(np.r_[neutral,wheel]-action)
        return action
