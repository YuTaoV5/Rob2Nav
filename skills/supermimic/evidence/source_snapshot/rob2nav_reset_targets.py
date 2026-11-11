"""Initialize controller targets after reset; never write qpos/qvel or solver state."""
from mjlab.managers.event_manager import EventTermCfg

def hold_current_joint_targets(env,env_ids):
 robot=env.scene['robot']
 ids=slice(None) if env_ids is None else env_ids
 robot.set_joint_position_target(robot.data.joint_pos[ids].clone(),env_ids=env_ids)

def install(cfg):
 cfg.events['initialize_hold_targets']=EventTermCfg(mode='reset',func=hold_current_joint_targets,params={})
 return cfg
