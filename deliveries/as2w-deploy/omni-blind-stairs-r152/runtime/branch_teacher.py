"""Short-horizon MuJoCo corrections for collecting safer student-state labels."""
import numpy as np
import torch

from train_student import sensor_frame


def actions(inverse):
    yield np.zeros(16)
    for legs in ((2,),(3,),(2,3),(0,1)):
        for dz,dx,dw in ((-.05,.0,-.10),(-.075,.025,-.22),(.065,.034,-.20),(.060,-.035,-.25),
                         (.0,.0,-.35)):
            target=np.zeros((4,3))
            target[list(legs),0]=dx
            target[list(legs),2]=dz
            q=np.einsum("lij,lj->il",inverse,target).reshape(12)/.35
            w=np.full(4,.35*dw)
            w[list(legs)]=dw
            yield np.r_[q.clip(-1.,1.),w]
    yield np.r_[np.zeros(12),np.full(4,-.45)]


class BranchTeacher:
    def __init__(self, sim, model, controller, defaults, limits, geometry, contract, width):
        self.mj=sim.mj
        self.model=model
        self.scratch=self.mj.MjData(model)
        self.controller=controller
        self.defaults=defaults
        self.limits=limits
        self.base=model.body("base_link").id
        self.wheels=[model.body(n).id for n in contract["joint_names"][12:]]
        joints=[model.joint(n) for n in contract["joint_names"]]
        self.qa=[j.qposadr[0] for j in joints]
        self.va=[j.dofadr[0] for j in joints]
        self.force=np.zeros(6)
        self.width=width
        self.candidates=tuple(actions(geometry.inverse))
        self.calls=self.changes=0

    def targets(self,action):
        leg=np.clip(self.defaults[:12]+.35*action[:12],*self.limits[:12,:2].T)
        wheel=np.clip(16.*action[12:],-self.limits[12:,2],self.limits[12:,2])
        return leg,wheel

    @torch.inference_mode()
    def rollout(self,data,obs,tick,advance,action,delta):
        scratch=self.scratch
        self.mj.mj_copyData(scratch,self.model,data)
        hidden=self.controller.stairs.state
        peak=self_peak=0.
        margin=upright=1.
        for j in range(4):
            if j>0 and self.controller.active_tick is not None:
                frame=obs.copy()
                frame[3:6]=scratch.qvel[3:6]
                frame[6:9]=-scratch.xmat[self.base,6:9]
                q=scratch.qpos[self.qa].copy()
                q[12:]=(q[12:]+2.*np.pi)%(4.*np.pi)-2.*np.pi
                frame[10:26]=q-self.defaults
                frame[26:42]=scratch.qvel[self.va]
                inp=torch.from_numpy(sensor_frame(frame,self.defaults))[None,None]
                guess,hidden=self.controller.stairs.actor(inp,hidden)
                u=min(1.,(tick+j-self.controller.active_tick+1.)/8.)
                u=u*u*(3.-2.*u)
                action=(1.-u)*self.controller.last_flat+u*guess[0,0].numpy()
            gain=1. if j==0 else .45 if j==1 else 0.
            leg,wheel=self.targets(action+gain*delta)
            if not advance(scratch,leg,wheel):
                return (float("inf"),float("inf"))
            for k,c in enumerate(scratch.contact):
                a,b=(int(self.model.geom_bodyid[g]) for g in c.geom)
                if a and b:
                    self.mj.mj_contactForce(self.model,scratch,k,self.force)
                    self_peak=max(self_peak,self.force[0])
                elif (a or b) not in self.wheels:
                    self.mj.mj_contactForce(self.model,scratch,k,self.force)
                    peak=max(peak,self.force[0])
            rot=scratch.xmat[self.base].reshape(3,3)
            loc=(scratch.xpos[self.wheels[:2]]-scratch.xpos[self.base])@rot
            margin=min(margin,loc[0,1],-loc[1,1])
            upright=min(upright,rot[2,2])
        score=(.018*peak+.036*max(peak-30.,0.)+.10*self_peak+20.*max(.065-margin,0.)
               +7.*max(.90-upright,0.)-3.5*(scratch.qpos[0]-data.qpos[0])+.10*np.square(delta).sum())
        return score,peak

    def choose(self,data,obs,tick,advance,action):
        self.calls+=1
        zero=self.candidates[0]
        best=self.rollout(data,obs,tick,advance,action,zero)
        if best[1]<32.:
            return action
        selected=zero
        options=self.candidates[1:]
        shift=(tick*5)%len(options)
        for delta in (options[shift:]+options[:shift])[:self.width]:
            check=self.rollout(data,obs,tick,advance,action,delta)
            if check[0]<best[0]-.02:
                selected,best=delta,check
        self.changes+=int(selected is not zero)
        return action+selected
