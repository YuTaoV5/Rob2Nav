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


def observation(skill,s,command,last,previous=None,phase=0.,reference=None,reference_yaw=0.):
    q=skill.from_sdk(s.q)-vector(skill.default);dq=skill.from_sdk(s.dq)
    last=skill.from_sdk(last);previous=last if previous is None else skill.from_sdk(previous)
    command=vector(command,3)
    if skill.observation=='stand':return np.zeros(1,dtype=np.float32)
    if skill.observation=='robogauge45':
        out=np.r_[s.gyro*.25,s.gravity,np.clip(command,[-1,-1,-2],[1,1,2])*[2,2,.25],q,dq*.05,last]
    elif skill.observation=='backflip60':
        theta=np.pi*min(max(phase,0.),2.)/2
        features=np.ravel([[np.sin(theta/d),np.cos(theta/d)] for d in [1,2,4]])
        out=np.clip(np.r_[s.gyro*.25,s.gravity,q,dq*.05,last,previous,features],-100,100)
    elif skill.observation=='upright48':
        if s.velocity is None:raise ValueError('upright48 requires measured/estimated BODY COM linear velocity; SDK LowState alone is insufficient')
        out=np.r_[s.velocity,s.gyro,s.gravity,command,q,dq,last]
    else:
        if reference is None or s.quaternion is None:raise ValueError('mimic69 requires pinned motion and measured orientation')
        index=min(round(max(phase,0.)*float(reference['fps'][0])),len(reference['joint_pos'])-1)
        anchor=list(reference['body_names']).index('base') if 'body_names' in reference else 0
        z=reference_yaw;align=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
        ori=rotation(s.quaternion).T@align@rotation(reference['body_quat_w'][index,anchor])
        out=np.r_[reference['joint_pos'][index],reference['joint_vel'][index],ori[:,:2].reshape(-1),s.gyro,q,dq,last]
    expected={'robogauge45':45,'backflip60':60,'upright48':48,'mimic69':69}[skill.observation]
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
            obs=vector(request['observation'],{'robogauge45':45,'backflip60':60,'upright48':48,'mimic69':69}[skill.observation]).astype(np.float32)
            started=time.monotonic();action=vector(infer(obs));duration=time.monotonic()-started
            target=skill.to_sdk(vector(skill.default)+skill.scale*action)
            pipe.send(dict(epoch=request['epoch'],stamp=request['stamp'],target=target.tolist(),inference_s=duration))
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
    def submit(self,obs,epoch,stamp):
        if not self.ready or self.inflight or self.error:return False
        self.pipe.send(dict(observation=np.asarray(obs).tolist(),epoch=epoch,stamp=stamp));self.inflight=True;return True
    def close(self):
        if self.process.is_alive():self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():self.process.kill();self.process.join(timeout=2)
        self.pipe.close()
