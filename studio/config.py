"""A typed, allowlisted job contract. No user commands or executable rewards."""
import math
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

REWARDS = {
    'forward_progress':{'label':'目标方向进度','weight':5.,'unit':'m/s','description':'速度或朝目标投影；具体公式随训练模板。'},
    'goal_motion':{'label':'目标速度跟踪','weight':5.,'unit':'无量纲','description':'目标运动奖励；减速模板在目标附近降低期望速度。'},
    'goal_position':{'label':'目标位置','weight':100.,'unit':'无量纲','description':'回合末段目标接近奖励。'},
    'route_bonus':{'label':'完整通过','weight':500.,'unit':'事件','description':'达到严格完整通过条件的终止奖励。'},
    'termination':{'label':'失败终止','weight':-100.,'unit':'事件','description':'失败终止惩罚。'},
    'lane_center':{'label':'横向偏移','weight':-2.,'unit':'m²','description':'机身相对路线中心的横向偏差平方。'},
    'anchor_cost':{'label':'原MoE保持','weight':-.2,'unit':'rad²','description':'策略动作相对冻结原MoE的偏差。'},
    'supported_contact':{'label':'落脚支撑','weight':1.,'unit':'比例','description':'首次接触时足下局部支撑比例。'},
    'heading_cost':{'label':'航向偏差','weight':-2.,'unit':'无量纲','description':'1-cos(yaw)，仅wide_deep配置存在。'},
    'descending_foot_risk':{'label':'下降脚风险','weight':-2.,'unit':'比例','description':'下降低脚下方缺失支撑；仅风险实验模板存在。'},
}

class RunConfig(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    name:str=Field(default='Go2 感知实验',min_length=1,max_length=80)
    adapter:Literal['moe','ame','nju','robogauge']='moe'
    task:Literal['train','simulation']='train'
    simulation_policy:Literal['baseline','checkpoint']='baseline'
    difficulty:float=Field(default=1.,ge=0,le=1)
    scene:Literal['empty','slalom','random','blocked','stairs','platform','gap','mixed','course']='course'
    jump_backend:Literal['disabled','nju','nju_runup','primitive']='nju'
    sensors:Literal['both','depth','lidar']='both'
    goal_x:float|None=Field(default=None,ge=-20,le=100)
    goal_y:float|None=Field(default=None,ge=-20,le=20)
    terrain_height:float=Field(default=.08,ge=.02,le=.5)
    gap_width:float=Field(default=.3,ge=.05,le=1.)
    max_seconds:float=Field(default=45,ge=1,le=120)
    mode:Literal['teacher','student']='teacher'
    course:Literal['original','wide_deep']='wide_deep'
    kind:Literal['mixed','flat','stairs','platform','ame_stakes','ame_alternate']='mixed'
    num_envs:int=Field(default=512,ge=8,le=4096)
    iterations:int=Field(default=200,ge=1,le=50000)
    seed:int=Field(default=42,ge=0,le=2147483647)
    warmup:int=Field(default=0,ge=0,le=50000)
    fusion:int=Field(default=0,ge=0,le=50000)
    actor:int=Field(default=0,ge=0,le=50000)
    memory_lifetime:float=Field(default=1.,gt=0,le=30)
    arrival_aware:bool=True
    foothold_aware:bool=False
    spatial_attention:bool=False
    numeric_debug:bool=True
    sensor_rig:Literal['original','near_down60','near_down60_front_lidar']='near_down60'
    initialization:Literal['base','initialize','resume']='base'
    checkpoint:str=''
    learning_rate:float=Field(default=.0002,ge=.000001,le=.01)
    actor_learning_rate:float=Field(default=.00002,ge=.0000001,le=.001)
    encoder_learning_rate:float=Field(default=.00001,ge=.0000001,le=.001)
    entropy_coef:float=Field(default=.001,ge=0,le=.1)
    gamma:float=Field(default=.99,ge=.8,le=.9999)
    lam:float=Field(default=.95,ge=.8,le=1.)
    clip_param:float=Field(default=.2,gt=0,le=.5)
    epochs:int=Field(default=4,ge=1,le=20)
    minibatches:int=Field(default=4,ge=1,le=64)
    max_grad_norm:float=Field(default=1.,gt=0,le=10)
    rewards:dict[str,float]=Field(default_factory=dict)
    auto_evaluate:bool=True

    @field_validator('rewards')
    @classmethod
    def reward_values(cls,v):
        if set(v)-set(REWARDS):raise ValueError('Unknown reward term')
        if any(not math.isfinite(x) or abs(x)>1000 for x in v.values()):raise ValueError('Invalid reward weight')
        return v

    @model_validator(mode='after')
    def consistency(self):
        if self.task=='simulation':
            if self.simulation_policy=='checkpoint' and (not self.checkpoint or self.kind=='mixed'):raise ValueError('Checkpoint simulation requires a weight and a specific terrain')
            if (self.goal_x is None)!=(self.goal_y is None):raise ValueError('Both target coordinates must be provided')
            if self.adapter not in ('moe','nju'):raise ValueError('Simulation supports MoE or NJU skill adapters')
            return self
        if self.num_envs%8:raise ValueError('Environment count must be a multiple of 8')
        if self.foothold_aware and self.mode!='teacher':raise ValueError('Foothold risk probe requires privileged teacher')
        if self.initialization!='base' and not self.checkpoint:raise ValueError('Select a checkpoint')
        if self.initialization=='base' and self.checkpoint:raise ValueError('Base initialization cannot select a checkpoint')
        if 'descending_foot_risk' in self.rewards and not self.foothold_aware:raise ValueError('Risk reward requires foothold-aware mode')
        if 'heading_cost' in self.rewards and self.course!='wide_deep':raise ValueError('Heading reward requires wide_deep')
        return self

    def overrides(self):
        return {'rewards':self.rewards,'ppo':{
            'learning_rate':self.learning_rate,'actor_learning_rate':self.actor_learning_rate,
            'encoder_learning_rate':self.encoder_learning_rate,'entropy_coef':self.entropy_coef,
            'gamma':self.gamma,'lam':self.lam,'clip_param':self.clip_param,
            'num_learning_epochs':self.epochs,'num_mini_batches':self.minibatches,'max_grad_norm':self.max_grad_norm}}


def resolve_checkpoint(root, value):
    root=Path(root).resolve();p=Path(value)
    if not p.is_absolute():p=root/p
    p=p.resolve()
    allowed=[root/'artifacts',root/'outputs',root.parent/'MoE-runs']
    if not any(p.is_relative_to(r.resolve()) for r in allowed) or p.suffix!='.pt' or not p.is_file():
        raise ValueError('Checkpoint must be an existing trusted project artifact')
    return p


def command(config,root,output,config_path,python):
    if config.adapter!='moe':raise ValueError('This adapter is not enabled for training yet')
    root=Path(root)
    cmd=[str(python),'-m','moe_perceptive.train','--headless','--moe',str(root/'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt'),
         '--navigation','--course',config.course,'--kind',config.kind,'--num-envs',str(config.num_envs),
         '--iterations',str(config.iterations),'--seed',str(config.seed),'--warmup',str(config.warmup),
         '--fusion',str(config.fusion),'--actor',str(config.actor),'--memory-lifetime',str(config.memory_lifetime),
         '--sensor-rig',config.sensor_rig,'--studio-config',str(config_path),'--output',str(output)]
    cmd.extend(['--stop-file',str(Path(config_path).parent/'cancel.flag')])
    if config.mode=='teacher':cmd.append('--teacher-map')
    for name in ('arrival_aware','foothold_aware','spatial_attention','numeric_debug'):
        if getattr(config,name):cmd.append('--'+name.replace('_','-'))
    if config.initialization!='base':cmd.extend(['--resume' if config.initialization=='resume' else '--initialize-from',str(resolve_checkpoint(root,config.checkpoint))])
    return cmd


def simulation_command(config,root,output,python):
    root=Path(root)
    if config.simulation_policy=='checkpoint':
        return [str(python),'-m','studio.simulate','--checkpoint',str(resolve_checkpoint(root,config.checkpoint)),
            '--kind',config.kind,'--course',config.course,'--difficulty',str(config.difficulty),'--seed',str(config.seed),
            '--max-seconds',str(config.max_seconds),'--output',str(output)]
    cmd=[str(python),'-m','go2_nav.active_run','--scene',config.scene,'--episodes','1','--seed',str(config.seed),
         '--height',str(config.terrain_height),'--gap-width',str(config.gap_width),
         '--max-seconds',str(config.max_seconds),'--jump-backend',config.jump_backend,
         '--sensors',config.sensors,'--video','--output',str(output)]
    if config.scene=='course':cmd.extend(['--config',str(root/'configs/mission_runup.json')])
    elif config.jump_backend.startswith('nju'):cmd.extend(['--config',str(root/'configs/nju_active.json')])
    if config.goal_x is not None:cmd.extend(['--goal',str(config.goal_x),str(config.goal_y)])
    return cmd
