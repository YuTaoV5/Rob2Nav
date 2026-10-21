"""A typed, allowlisted job contract. No user commands or executable rewards."""
import math
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

from rob2nav.experiment_schema import REWARDS, reward_schema

class RunConfig(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    name:str=Field(default='Go2 感知实验',min_length=1,max_length=80)
    adapter:Literal['moe','ame','nju','robogauge','pie','him','wtw','wmp','mid360','leggedskilldeploy']='moe'
    recipe_id:str=Field(default='',max_length=80)
    capability_id:str=Field(default='',max_length=80)
    robot_id:Literal['go2','go1','g1_29dof','custom_dog']='go2'
    algorithm_id:str=Field(default='moe_terrain_ppo',max_length=80)
    task:Literal['train','simulation']='train'
    simulation_policy:Literal['baseline','checkpoint','pie','lsd_loco','lsd_jump','lsd_backflip','lsd_sideflip']='baseline'
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
    navigation:bool=True
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
    body_height_target_m:float|None=Field(default=None,ge=.18,le=.5)
    swing_clearance_target_m:float|None=Field(default=None,ge=.04,le=.25)
    rewards:dict[str,float]=Field(default_factory=dict)
    auto_evaluate:bool=True
    recipe_options:dict[str,str|int|float|bool]=Field(default_factory=dict)

    @field_validator('rewards')
    @classmethod
    def reward_values(cls,v):
        if set(v)-set(REWARDS):raise ValueError('Unknown reward term')
        if any(not math.isfinite(x) or abs(x)>1000 for x in v.values()):raise ValueError('Invalid reward weight')
        return v

    @model_validator(mode='after')
    def consistency(self):
        from .registry import resolve_recipe
        recipe=resolve_recipe(self)
        if recipe and recipe['execution']=='manual_cli' and self.task=='train':
            from .recipe_training import validate_options
            validate_options(recipe['id'],self.recipe_options)
            return self
        if self.task=='simulation' and (self.body_height_target_m is not None or self.swing_clearance_target_m is not None):raise ValueError('Height targets are training objectives, not runtime simulation commands')
        if self.task=='simulation':
            if self.simulation_policy=='checkpoint' and (not self.checkpoint or self.kind=='mixed'):raise ValueError('Checkpoint simulation requires a weight and a specific terrain')
            if (self.goal_x is None)!=(self.goal_y is None):raise ValueError('Both target coordinates must be provided')
            if self.adapter not in ('moe','nju','pie','leggedskilldeploy'):raise ValueError('This simulation adapter is not connected')
            return self
        if self.num_envs%8:raise ValueError('Environment count must be a multiple of 8')
        if not self.navigation and (self.arrival_aware or self.mode=='teacher' or self.foothold_aware):raise ValueError('Goal/teacher experiments require the navigation observation interface')
        if not self.navigation and set(self.rewards)&{'goal_motion','goal_position','route_bonus'}:raise ValueError('Goal rewards require the navigation interface')
        if self.foothold_aware and self.mode!='teacher':raise ValueError('Foothold risk probe requires privileged teacher')
        if self.initialization!='base' and not self.checkpoint:raise ValueError('Select a checkpoint')
        if self.initialization=='base' and self.checkpoint:raise ValueError('Base initialization cannot select a checkpoint')
        if 'descending_foot_risk' in self.rewards and not self.foothold_aware:raise ValueError('Risk reward requires foothold-aware mode')
        if 'heading_cost' in self.rewards and self.course!='wide_deep':raise ValueError('Heading reward requires wide_deep')
        for name in self.rewards:
            if not reward_schema(self)[name]['available']:raise ValueError('Reward not available in this mode: '+name)
        return self

    def overrides(self):
        result={'rewards':self.rewards,'ppo':{
            'learning_rate':self.learning_rate,'actor_learning_rate':self.actor_learning_rate,
            'encoder_learning_rate':self.encoder_learning_rate,'entropy_coef':self.entropy_coef,
            'gamma':self.gamma,'lam':self.lam,'clip_param':self.clip_param,
            'num_learning_epochs':self.epochs,'num_mini_batches':self.minibatches,'max_grad_norm':self.max_grad_norm}}
        motion={k:getattr(self,k) for k in ('body_height_target_m','swing_clearance_target_m') if getattr(self,k) is not None}
        if motion:result['motion']=motion
        return result


def resolve_checkpoint(root, value):
    root=Path(root).resolve();p=Path(value)
    if not p.is_absolute():p=root/p
    p=p.resolve()
    allowed=[root/'artifacts',root/'outputs',root.parent/'MoE-runs']
    if not any(p.is_relative_to(r.resolve()) for r in allowed) or p.suffix!='.pt' or not p.is_file():
        raise ValueError('Checkpoint must be an existing trusted project artifact')
    return p


def command(config,root,output,config_path,python,mujoco_python=None):
    from .registry import resolve_recipe
    recipe=resolve_recipe(config,require_ready=True)
    if recipe and recipe['execution']=='manual_cli':
        from .recipe_training import recipe_command
        return recipe_command(config,root,output,python,mujoco_python)
    if config.adapter!='moe':raise ValueError('This adapter is not enabled for training yet')
    root=Path(root)
    cmd=[str(python),'-m','moe_perceptive.train','--headless','--moe',str(root/'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt'),
         '--course',config.course,'--kind',config.kind,'--num-envs',str(config.num_envs),
         '--iterations',str(config.iterations),'--seed',str(config.seed),'--warmup',str(config.warmup),
         '--fusion',str(config.fusion),'--actor',str(config.actor),'--memory-lifetime',str(config.memory_lifetime),
         '--sensor-rig',config.sensor_rig,'--studio-config',str(config_path),'--output',str(output)]
    cmd.extend(['--stop-file',str(Path(config_path).parent/'cancel.flag')])
    if config.navigation:cmd.append('--navigation')
    if config.mode=='teacher':cmd.append('--teacher-map')
    for name in ('arrival_aware','foothold_aware','spatial_attention','numeric_debug'):
        if getattr(config,name):cmd.append('--'+name.replace('_','-'))
    if config.initialization!='base':cmd.extend(['--resume' if config.initialization=='resume' else '--initialize-from',str(resolve_checkpoint(root,config.checkpoint))])
    return cmd


def simulation_command(config,root,output,python):
    if config.robot_id!='go2':raise ValueError('Only the Go2 dynamics adapter is currently validated for Studio simulation')
    root=Path(root)
    if config.simulation_policy in ('pie','lsd_loco','lsd_jump','lsd_backflip','lsd_sideflip'):
        from .upstreams import source
        source(root,'pie' if config.simulation_policy=='pie' else 'lsd')
        if config.scene not in ('empty','stairs') or (config.simulation_policy.startswith('lsd_') and config.scene!='empty'):raise ValueError('This pinned inference adapter does not support the selected terrain')
        return [str(python),'-m','studio.external_sim','--policy',config.simulation_policy,'--root',str(root),'--scene',config.scene,
            '--seconds',str(config.max_seconds),'--seed',str(config.seed),'--output',str(output)]
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
