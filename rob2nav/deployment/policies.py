"""Distinct observation adapters and a model process outside the motor loop."""
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from .types import Skill,vector


def rotation(q):
    w,x,y,z=vector(q,4)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*(3.-2.*x)


def reference_action_frame(reference,phase):
    count=len(reference['joint_pos']);frame=min(round(max(phase,0.)*float(reference['fps'][0])),count-1)
    return dict(frame=frame,count=count,joint_pos=np.asarray(reference['joint_pos'][frame]).tolist())


def history_action(skill,raw,phase=0.):
    """Training history, expressed in SDK order for the observation adapter."""
    raw=vector(raw)
    if skill.action_parameterization in ('backflip_entry_v2','backflip_entry_v3'):
        hold=1.-_smooth((phase-1.)/.2)
        raw=np.clip((1.-hold)*raw+hold*skill.entry_hold_scale*np.tanh(raw),-8.,8.)
    elif skill.action_parameterization!='joint_offset_v1':raw=np.clip(raw,-100.,100.)
    return skill.to_sdk(raw)


def action_target(skill,raw,phase=0.,reference=None):
    """Apply the exact time-dependent action mapping used by FIXSTAND training."""
    raw=vector(raw)
    mode=skill.action_parameterization
    target=vector(skill.default)+skill.scale*raw
    if mode in ('backflip_entry_v2','backflip_entry_v3'):
        target=vector(skill.default)+skill.scale*skill.from_sdk(history_action(skill,raw,phase))
    elif mode in ('upright_endpoint_v2','upright_endpoint_v3'):
        weight=max(1.-_smooth((phase-1.)/.5),_smooth((phase-5.)/1.5))
        target=(1.-weight)*target+weight*(vector(skill.default)+skill.endpoint_residual_limit_rad*np.tanh(raw))
    elif mode in ('reference_residual_v2','reference_residual_v3','reference_residual_v4'):
        if reference is None:raise ValueError('Sideflip residual requires the pinned reference')
        if 'frame' in reference:
            frame=int(reference['frame']);count=int(reference['count']);nominal=reference['joint_pos']
        else:
            frame_data=reference_action_frame(reference,phase)
            frame=frame_data['frame'];count=frame_data['count'];nominal=frame_data['joint_pos']
        if frame<90 or frame>=count-90 or skill.boundary_blend_frames:
            distance=min(frame,count-1-frame)
            blend=_smooth((distance-50.)/40.)
            limit=skill.residual_limit_rad+(skill.bridge_residual_limit_rad-skill.residual_limit_rad)*blend
            residual=limit*np.tanh(raw)
            if skill.terminal_hip_taper_frames:
                taper=_smooth((count-1-frame)/(skill.terminal_hip_taper_frames-1))
                residual=np.where([name.endswith('_hip_joint') for name in skill.joints],residual*taper,residual)
            bounded=vector(nominal)+residual
            if skill.boundary_blend_frames:
                width=skill.boundary_blend_frames
                def quintic(x):
                    x=np.clip(x,0.,1.)
                    return x*x*x*(10.+x*(-15.+6.*x))
                weight=quintic((frame-(90-width))/(2.*width))*quintic(((count-90+width)-frame)/(2.*width))
                target=(1.-weight)*bounded+weight*target
            else:target=bounded
    return skill.to_sdk(target)


def observation(skill,s,command,last,previous=None,phase=0.,reference=None,reference_yaw=0.):
    q=skill.from_sdk(s.q)-vector(skill.default);dq=skill.from_sdk(s.dq)
    last=skill.from_sdk(last);previous=last if previous is None else skill.from_sdk(previous)
    command=vector(command,3)
    if skill.observation=='stand':return np.zeros(1,dtype=np.float32)
    if skill.observation=='robogauge45':
        out=np.r_[s.gyro*.25,s.gravity,np.clip(command,[-1,-1,-2],[1,1,2])*[2,2,.25],q,dq*.05,last]
    elif skill.observation=='backflip60':
        time=phase-1. if skill.action_parameterization in ('backflip_entry_v2','backflip_entry_v3') else max(phase,0.)
        theta=np.pi*min(time,2.)/2
        features=np.ravel([[np.sin(theta/d),np.cos(theta/d)] for d in [1,2,4]])
        out=np.clip(np.r_[s.gyro*.25,s.gravity,q,dq*.05,last,previous,features],-100,100)
    elif skill.observation in ('upright48','upright52'):
        if s.velocity is None:raise ValueError('upright48 requires measured/estimated BODY COM linear velocity; SDK LowState alone is insufficient')
        out=np.r_[s.velocity,s.gyro,s.gravity,command,q,dq,last]
        if skill.observation=='upright52':out=np.r_[out,[float(phase<1.),float(1.<=phase<5.),float(phase>=5.),np.clip(phase/8.,0.,1.)]]
    else:
        if reference is None or s.quaternion is None:raise ValueError('mimic69 requires pinned motion and measured orientation')
        index=min(round(max(phase,0.)*float(reference['fps'][0])),len(reference['joint_pos'])-1)
        anchor=list(reference['body_names']).index('base') if 'body_names' in reference else 0
        z=reference_yaw;align=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
        ori=rotation(s.quaternion).T@align@rotation(reference['body_quat_w'][index,anchor])
        out=np.r_[reference['joint_pos'][index],reference['joint_vel'][index],ori[:,:2].reshape(-1),s.gyro,q,dq,last]
    expected={'robogauge45':45,'backflip60':60,'upright48':48,'upright52':52,'mimic69':69}[skill.observation]
    return vector(out,expected).astype(np.float32)


def _worker(pipe,manifest,root):
    try:
        skill=Skill.load(manifest,root)
        if not skill.policy:raise ValueError('Stand controller does not need a model process')
        path=Path(root)/skill.policy
        if path.suffix=='.onnx':
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            actor=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            def infer(obs):return actor.run(None,{actor.get_inputs()[0].name:obs[None].astype(np.float32)})[0][0]
        else:
            import torch
            torch.set_num_threads(1);actor=torch.jit.load(str(path),map_location='cpu').eval()
            if hasattr(actor,'reset'):actor.reset()
            def infer(obs):
                with torch.inference_mode():value=actor(torch.from_numpy(obs.astype(np.float32))[None])
                if isinstance(value,tuple):value=value[0]
                return value[0].numpy()
        pipe.send(dict(ready=True,sha256=skill.sha256))
        while True:
            request=pipe.recv()
            if request is None:return
            if request.get('reset'):
                if hasattr(actor,'reset'):actor.reset()
                pipe.send(dict(reset_done=True));continue
            obs=vector(request['observation'],{'robogauge45':45,'backflip60':60,'upright48':48,'upright52':52,'mimic69':69}[skill.observation]).astype(np.float32)
            started=time.monotonic();action=vector(infer(obs));duration=time.monotonic()-started
            target=action_target(skill,action,float(request.get('phase',0.)),request.get('reference'))
            history=history_action(skill,action,float(request.get('phase',0.)))
            pipe.send(dict(epoch=request['epoch'],stamp=request['stamp'],target=target.tolist(),raw_action=action.tolist(),
                history_action=history.tolist(),inference_s=duration))
    except (EOFError,BrokenPipeError):return
    except Exception as exc:
        try:pipe.send(dict(error=repr(exc)))
        except (EOFError,BrokenPipeError):pass
    finally:pipe.close()


class PolicyProcess:
    """At most one outstanding observation. poll/submit never waits for inference."""
    def __init__(self,manifest,root):
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe();self.process=ctx.Process(target=_worker,args=(child,str(manifest),str(root)),daemon=True)
        self.process.start();child.close();self.inflight=False;self.ready=False;self.error=None
    def poll(self):
        if not self.pipe.poll():return None
        result=self.pipe.recv()
        if result.get('ready'):self.ready=True
        elif 'error' in result:self.error=result['error'];self.inflight=False
        else:self.inflight=False
        return result
    def submit(self,obs,epoch,stamp,phase=0.,reference=None):
        if not self.ready or self.inflight or self.error:return False
        self.pipe.send(dict(observation=np.asarray(obs).tolist(),epoch=epoch,stamp=stamp,phase=phase,
            reference=reference));self.inflight=True;return True
    def close(self):
        if self.process.is_alive():self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():self.process.kill();self.process.join(timeout=2)
        self.pipe.close()
