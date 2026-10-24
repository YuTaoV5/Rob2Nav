"""Reviewable positive-pitch reward adaptation; no robot or policy mirroring."""
import hashlib
import inspect
from pathlib import Path
import subprocess
import textwrap

PIN='954aa61ceb62c515689677f44a21e32109f6b62d'


def adapt_method(name,text):
    if name in ('_update_flip_state','_reward_ang_vel_y'):
        old='-self._robot.data.root_ang_vel_b[:, 1]';new='self._robot.data.root_ang_vel_b[:, 1]'
    elif name=='_reward_orientation_control':
        old='desired_angle = -4.0 * torch.pi * phase';new='desired_angle = 4.0 * torch.pi * phase'
    else:raise ValueError('Unreviewed forward-flip method')
    if text.count(old)!=1:raise ValueError('Pinned rotation reward changed: '+name)
    return text.replace(old,new)


def roll_rate_penalty(self):
    """Non-target body roll rate; positive magnitude, use a negative weight."""
    return abs(self._robot.data.root_ang_vel_b[:, 0])


def configure(cfg,source):
    from isaaclab_backflip.tasks.go2_backflip.go2_backflip_env import Go2BackflipEnv
    source=Path(source);file=Path(inspect.getfile(Go2BackflipEnv)).resolve()
    relative=file.relative_to(source.resolve()).as_posix()
    pristine=subprocess.check_output(['git','-C',str(source),'show',PIN+':'+relative])
    if file.read_bytes()!=pristine:raise ValueError('Frontflip base environment is not the pinned source')
    if getattr(Go2BackflipEnv,'_rob2nav_forward_rotation',False):raise ValueError('Forward task already configured')
    for name in ('_update_flip_state','_reward_ang_vel_y','_reward_orientation_control'):
        original=getattr(Go2BackflipEnv,name);scope=dict(original.__globals__)
        code=adapt_method(name,textwrap.dedent(inspect.getsource(original)))
        exec(compile(code,'<rob2nav-positive-pitch-task>','exec'),scope)
        setattr(Go2BackflipEnv,name,scope[name])
    Go2BackflipEnv._rob2nav_forward_rotation=True
    Go2BackflipEnv._reward_ang_vel_x=roll_rate_penalty
    cfg.rewards.scales.setdefault('ang_vel_x',0.0)
    cfg.flip_direction='front'
    cfg.frontflip_reward_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg.frontflip_base_sha256=hashlib.sha256(pristine).hexdigest()
    print('[Rob2Nav] Independent positive-pitch frontflip rewards; Go2 morphology unchanged',flush=True)
