"""External MLP teacher control for a diagnostic baseline, not AME success."""
import json
import torch
from torch import nn


def add_scanner(cfg):
    from isaaclab.sensors import RayCasterCfg,patterns
    cfg.scene.diagnostic_teacher_scan=RayCasterCfg(
        prim_path='{ENV_REGEX_NS}/Robot/base',update_period=.02,ray_alignment='yaw',
        offset=RayCasterCfg.OffsetCfg(pos=(.25,0.,20.)),
        pattern_cfg=patterns.GridPatternCfg(resolution=.1,size=(1.6,1.0),ordering='xy'),
        mesh_prim_paths=['/World/ground'])


class Teacher:
    def __init__(self,env,checkpoint,io):
        self.env=env;self.io=json.load(open(io));self.robot=env.scene['robot']
        self.action=env.action_manager.get_term('joint_pos')
        names=self.io['joint_names'];current=self.action._joint_names
        self.asset_indices=[self.robot.joint_names.index(n) for n in names]
        self.teacher_to_action=[names.index(n) for n in current]
        self.action_to_teacher=[current.index(n) for n in names]
        self.default=torch.tensor(self.io['default_joint_pos'],device=env.device)
        self.scale=float(self.io['action_scale'])
        state=torch.load(checkpoint,map_location=env.device,weights_only=False)['model_state_dict']
        self.actor=nn.Sequential(nn.Linear(235,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
                                 nn.Linear(256,128),nn.ELU(),nn.Linear(128,12)).to(env.device)
        self.actor.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith('actor.')})
        self.actor.eval()

    def __call__(self,obs):
        r=self.robot;e=self.env
        previous_targets=self.action._offset+self.action._scale*e.action_manager.action
        previous=(previous_targets[:,self.action_to_teacher]-self.default)/self.scale
        heights=(r.data.root_pos_w[:,2,None]-e.scene['diagnostic_teacher_scan'].data.ray_hits_w[...,2]-.5).clamp(-1,1)
        if not torch.isfinite(heights).all():raise FloatingPointError('Diagnostic teacher scan missed')
        inputs=torch.cat((r.data.root_lin_vel_b,r.data.root_ang_vel_b,r.data.projected_gravity_b,
            e.command_manager.get_command('base_velocity'),r.data.joint_pos[:,self.asset_indices]-self.default,
            r.data.joint_vel[:,self.asset_indices],previous,heights),-1)
        target=self.default+self.scale*self.actor(inputs).clamp(-10,10)
        return (target[:,self.teacher_to_action]-self.action._offset)/self.action._scale
