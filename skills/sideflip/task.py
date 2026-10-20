"""Shared native Go2 motion-imitation configuration (training and diagnostic playback)."""
import re
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg


def motion_finished(env):
    cmd = env.command_manager.get_term('motion')
    return cmd.time_steps >= cmd.motion.time_step_total-2


def make_cfg(source, motion, contract, num_envs, seed, device):
    from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import RobotEnvCfg
    from .common import robot_config
    cfg = RobotEnvCfg()
    cfg.seed = seed
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    cfg.scene.robot = robot_config(source)
    cfg.scene.terrain.visual_material = None
    cfg.scene.contact_forces.debug_vis = False
    bodies = contract['body_names']
    anchor = 'base' if 'base' in bodies else 'base_link'
    feet = []
    for leg in ('FL','FR','RL','RR'):
        candidates = [n for n in bodies if n.startswith(leg+'_') and ('foot' in n.lower() or 'calf' in n.lower())]
        if not candidates:
            raise RuntimeError('No Go2 distal leg body for '+leg)
        feet.append(candidates[-1])
    cfg.commands.motion.motion_file = str(motion.resolve())
    cfg.commands.motion.anchor_body_name = anchor
    cfg.commands.motion.body_names = bodies
    cfg.commands.motion.debug_vis = False
    cfg.actions.JointPositionAction.scale = .5
    cfg.actions.JointPositionAction.joint_names = contract['joint_names']
    cfg.actions.JointPositionAction.preserve_order = True
    cfg.events.base_com.params['asset_cfg'] = SceneEntityCfg('robot', body_names=anchor)
    cfg.events.base_com.params['com_range'] = {axis:(-.015,.015) for axis in ('x','y','z')}
    cfg.events.physics_material.params['restitution_range'] = (0., .05)
    cfg.events.push_robot = None
    cfg.rewards.undesired_contacts.params['sensor_cfg'] = SceneEntityCfg('contact_forces',
        body_names=['^(?!(?:'+'|'.join(map(re.escape,feet))+')$).+$'])
    cfg.terminations.ee_body_pos.params['body_names'] = feet
    cfg.terminations.motion_end = TerminationTermCfg(func=motion_finished, time_out=True)
    cfg.episode_length_s = 4.
    return cfg
